from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from universal_agent_policy.data import load_hidden_tensor
from universal_agent_policy.runtime.policy_runtime import FrozenPolicyRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-jsonl", required=True)
    parser.add_argument("--hidden-tensor", required=True)
    parser.add_argument("--source-policy", required=True)
    parser.add_argument("--target-adapter")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def is_tool_action(name: str) -> bool:
    return name.startswith("ACT_") or name in {"THINK", "TRANSFER"}


def action_names_from_rows(rows: list[dict[str, Any]]) -> list[str] | None:
    for row in rows:
        names = row.get("action_names")
        if names:
            return list(names)
    return None


def main() -> None:
    args = parse_args()
    rows = load_rows(args.decision_jsonl)
    tensor = load_hidden_tensor(args.hidden_tensor)
    if len(rows) != tensor["h"].shape[0]:
        raise ValueError("decision rows and hidden tensor must have the same length")

    tensor_ids = tensor.get("sample_id")
    if tensor_ids:
        row_ids = [row.get("sample_id", "") for row in rows]
        if list(tensor_ids)[: len(row_ids)] != row_ids:
            raise ValueError("decision rows and hidden tensor sample_id order differ")

    if args.target_adapter:
        policy = FrozenPolicyRuntime.from_target_adapter(args.source_policy, args.target_adapter, args.device)
    else:
        policy = FrozenPolicyRuntime.from_source_policy(args.source_policy, args.device)
    fallback_action_names = action_names_from_rows(rows)
    checkpoint_has_names = not policy.action_space.names[0].startswith("NO_TOOL") or policy.action_space.names == fallback_action_names

    records = []
    by_expected: dict[str, list[bool]] = defaultdict(list)
    by_domain: dict[str, list[bool]] = defaultdict(list)
    tool_action_results: list[bool] = []
    for idx, row in enumerate(rows):
        prediction = policy.predict_from_hidden(tensor["h"][idx])
        expected_action = str(row["correct_action"])
        predicted_id = prediction.action.action_id
        expected_id = int(row.get("action_id", tensor["y"][idx].item()))
        if fallback_action_names and 0 <= predicted_id < len(fallback_action_names):
            predicted_action = fallback_action_names[predicted_id]
        else:
            predicted_action = prediction.action.name
        if checkpoint_has_names:
            correct = expected_action == predicted_action
        else:
            correct = expected_id == predicted_id
        by_expected[expected_action].append(correct)
        by_domain[str(row.get("domain", "unknown"))].append(correct)
        if is_tool_action(expected_action):
            tool_action_results.append(correct)
        records.append(
            {
                "sample_id": row.get("sample_id"),
                "domain": row.get("domain"),
                "reward": row.get("reward"),
                "turn_index": row.get("turn_index"),
                "expected_action": expected_action,
                "predicted_action": predicted_action,
                "correct": correct,
            }
        )

    total = max(1, len(records))
    per_action = {key: sum(vals) / max(1, len(vals)) for key, vals in sorted(by_expected.items())}
    per_domain = {key: sum(vals) / max(1, len(vals)) for key, vals in sorted(by_domain.items())}
    summary = {
        "num_samples": len(records),
        "accuracy": sum(1 for row in records if row["correct"]) / total,
        "macro_action_accuracy": sum(per_action.values()) / max(1, len(per_action)),
        "tool_action_accuracy": sum(tool_action_results) / max(1, len(tool_action_results)),
        "per_action_accuracy": per_action,
        "per_domain_accuracy": per_domain,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
