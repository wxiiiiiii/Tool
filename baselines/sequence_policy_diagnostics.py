from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from baselines.tool_policy_utils import load_prediction_file, load_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute trajectory-level diagnostics for precomputed action predictions. "
            "These metrics expose whether a static action-transfer baseline preserves "
            "sequential agent policy dynamics, not only per-state action accuracy."
        )
    )
    parser.add_argument("--decision-jsonl", required=True)
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--replay-json", nargs="*", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown-out")
    parser.add_argument("--max-samples", type=int, default=-1)
    return parser.parse_args()


def task_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row.get("domain")), int(row.get("task_id", -1)), int(row.get("trial", 0)))


def group_indices(rows: list[dict[str, Any]]) -> dict[tuple[str, int, int], list[int]]:
    groups: dict[tuple[str, int, int], list[tuple[int, int]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[task_key(row)].append((int(row.get("turn_index", idx)), idx))
    return {key: [idx for _, idx in sorted(items)] for key, items in groups.items()}


def avg(values: list[bool]) -> float:
    return sum(1.0 for value in values if value) / max(1, len(values))


def load_replay_summaries(paths: list[str]) -> dict[str, dict[str, Any]]:
    summaries = {}
    for path in paths:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        method = str(obj.get("summary", {}).get("method", Path(path).stem))
        summaries[method] = obj.get("summary", {})
    return summaries


def first_divergence(expected: list[str], predicted: list[str]) -> int:
    for idx, (exp, pred) in enumerate(zip(expected, predicted, strict=True)):
        if exp != pred:
            return idx
    return len(expected)


def diagnose(rows: list[dict[str, Any]], prediction_path: str, replay_summary: dict[str, Any] | None) -> dict[str, Any]:
    obj = load_prediction_file(prediction_path)
    records = obj["records"][: len(rows)]
    if len(records) != len(rows):
        raise ValueError(f"{prediction_path}: prediction record count does not match decision rows")
    for row, record in zip(rows, records, strict=True):
        if str(row.get("sample_id")) != str(record.get("sample_id")):
            raise ValueError(f"{prediction_path}: sample_id order does not match decision rows")

    method = str(obj.get("summary", {}).get("method", Path(prediction_path).stem))
    expected = [str(row.get("correct_action", "")) for row in rows]
    predicted = [str(record.get("predicted_action", "")) for record in records]
    groups = group_indices(rows)

    action_correct = [exp == pred for exp, pred in zip(expected, predicted, strict=True)]
    seq_exact = []
    first_steps = []
    transition_exact = []
    transition_state_exact = []
    expected_flip_pairs = []
    predicted_flip_on_expected_flip = []
    correct_next_on_expected_flip = []
    correct_next_after_correct_prev_flip = []

    for indices in groups.values():
        exp_seq = [expected[idx] for idx in indices]
        pred_seq = [predicted[idx] for idx in indices]
        seq_exact.append(exp_seq == pred_seq)
        first_steps.append(first_divergence(exp_seq, pred_seq))

        for left, right in zip(indices, indices[1:]):
            exp_pair = (expected[left], expected[right])
            pred_pair = (predicted[left], predicted[right])
            transition_exact.append(exp_pair == pred_pair)
            transition_state_exact.append(action_correct[left] and action_correct[right])
            if expected[left] != expected[right]:
                expected_flip_pairs.append(True)
                predicted_flip_on_expected_flip.append(predicted[left] != predicted[right])
                correct_next_on_expected_flip.append(predicted[right] == expected[right])
                if predicted[left] == expected[left]:
                    correct_next_after_correct_prev_flip.append(predicted[right] == expected[right])

    result = {
        "method": method,
        "prediction_file": prediction_path,
        "num_samples": len(rows),
        "num_tasks": len(groups),
        "action_accuracy": avg(action_correct),
        "sequence_exact_match": avg(seq_exact),
        "transition_pair_exact": avg(transition_exact),
        "transition_state_accuracy": avg(transition_state_exact),
        "expected_action_flip_pairs": len(expected_flip_pairs),
        "predicted_flip_rate_on_expected_flip": avg(predicted_flip_on_expected_flip),
        "next_action_accuracy_on_expected_flip": avg(correct_next_on_expected_flip),
        "next_action_accuracy_after_correct_prev_flip": avg(correct_next_after_correct_prev_flip),
        "mean_first_divergence_step": sum(first_steps) / max(1, len(first_steps)),
    }
    if replay_summary:
        for key in (
            "tool_execution_ok_rate_predicted_tool",
            "tool_grounding_accuracy_all",
            "tool_grounding_accuracy_tool_only",
            "db_hash_match_rate",
            "formal_gt_hash_match_rate",
        ):
            if key in replay_summary:
                result[key] = replay_summary[key]
    return result


def pct(value: Any) -> str:
    return "" if value is None else f"{100 * float(value):.2f}%"


def main() -> None:
    args = parse_args()
    rows = load_rows(args.decision_jsonl, args.max_samples)
    replay_summaries = load_replay_summaries(args.replay_json)
    results = []
    for prediction_path in args.predictions:
        obj = load_prediction_file(prediction_path)
        method = str(obj.get("summary", {}).get("method", Path(prediction_path).stem))
        results.append(diagnose(rows, prediction_path, replay_summaries.get(method)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results}, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.markdown_out:
        md = Path(args.markdown_out)
        md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Sequence Policy Diagnostics",
            "",
            f"- decision states: `{args.decision_jsonl}`",
            f"- samples: {len(rows)}",
            "",
            "| Method | Action Acc | Seq Exact | Transition Pair Exact | Next Acc On Expected Flip | Mean First Div Step | Tool Exec OK | DB Hash |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for item in results:
            lines.append(
                "| {method} | {action} | {seq} | {trans} | {flip} | {first:.2f} | {exec_ok} | {db} |".format(
                    method=item["method"],
                    action=pct(item["action_accuracy"]),
                    seq=pct(item["sequence_exact_match"]),
                    trans=pct(item["transition_pair_exact"]),
                    flip=pct(item["next_action_accuracy_on_expected_flip"]),
                    first=float(item["mean_first_divergence_step"]),
                    exec_ok=pct(item.get("tool_execution_ok_rate_predicted_tool")),
                    db=pct(item.get("db_hash_match_rate")),
                )
            )
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"results": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
