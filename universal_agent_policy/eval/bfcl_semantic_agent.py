from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from universal_agent_policy.data import load_hidden_tensor
from universal_agent_policy.runtime.policy_runtime import FrozenPolicyRuntime
from universal_agent_policy.runtime.semantic_grounding import SemanticActionGrounder


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


def expected_tool_name(row: dict[str, Any]) -> str | None:
    name = row.get("expected_tool_name")
    if name:
        return str(name)
    action_id = int(row.get("slot_action_id", row.get("action_id", 0)))
    tools = list(row.get("tools", []))
    if action_id > 0 and action_id - 1 < len(tools):
        return str(tools[action_id - 1].get("name", ""))
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
    grounder = SemanticActionGrounder()

    records = []
    for idx, row in enumerate(rows):
        prediction = policy.predict_from_hidden(tensor["h"][idx])
        predicted_action = prediction.action.name
        expected_action = str(row["correct_action"])
        expected_tool = expected_tool_name(row)

        grounded = grounder.ground(predicted_action, row)
        oracle_grounded = grounder.ground(expected_action, row)
        predicted_tool = None if grounded.tool is None else grounded.tool.name
        oracle_tool = None if oracle_grounded.tool is None else oracle_grounded.tool.name

        expected_calls_tool = expected_tool is not None
        predicted_calls_tool = predicted_tool is not None
        records.append(
            {
                "sample_id": row.get("sample_id"),
                "category": row.get("category"),
                "expected_action": expected_action,
                "predicted_action": predicted_action,
                "policy_action_correct": predicted_action == expected_action,
                "expected_tool": expected_tool,
                "predicted_tool": predicted_tool,
                "oracle_grounded_tool": oracle_tool,
                "calls_tool_correct": predicted_calls_tool == expected_calls_tool,
                "tool_grounding_correct": predicted_tool == expected_tool,
                "oracle_grounding_correct": oracle_tool == expected_tool,
                "grounding_reason": grounded.reason,
            }
        )

    def avg(key: str) -> float:
        return sum(1.0 for row in records if row[key]) / max(1, len(records))

    tool_records = [row for row in records if row["expected_tool"] is not None]
    summary = {
        "num_samples": len(records),
        "policy_action_accuracy": avg("policy_action_correct"),
        "calls_tool_accuracy": avg("calls_tool_correct"),
        "tool_grounding_accuracy_all": avg("tool_grounding_correct"),
        "tool_grounding_accuracy_tool_only": sum(1.0 for row in tool_records if row["tool_grounding_correct"])
        / max(1, len(tool_records)),
        "oracle_grounding_accuracy_all": avg("oracle_grounding_correct"),
        "oracle_grounding_accuracy_tool_only": sum(1.0 for row in tool_records if row["oracle_grounding_correct"])
        / max(1, len(tool_records)),
        "end_to_end_tool_decision_rate": sum(
            1.0
            for row in records
            if row["policy_action_correct"] and row["calls_tool_correct"] and row["tool_grounding_correct"]
        )
        / max(1, len(records)),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
