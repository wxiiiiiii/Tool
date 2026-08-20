from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WRITE_MARKERS = ("UPDATE", "CANCEL", "BOOK", "SEND_CERTIFICATE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown-out")
    return parser.parse_args()


def transition_name(prev_action: str | None, action: str) -> str:
    return f"{prev_action or 'START'}->{action}"


def is_write_action(action: str | None) -> bool:
    if not action:
        return False
    return any(marker in action for marker in WRITE_MARKERS)


def classify_record(record: dict[str, Any]) -> str:
    expected_action = str(record.get("expected_action"))
    predicted_action = str(record.get("predicted_action"))
    expected_tool = record.get("expected_tool")
    predicted_tool = record.get("predicted_tool")
    if record.get("action_correct") and record.get("tool_grounding_correct") and record.get("execution_ok"):
        return "ok"
    if predicted_action in {"ANSWER", "TRANSFER"} and predicted_action != expected_action:
        return "stop_or_transfer_too_early"
    if expected_action in {"ANSWER", "TRANSFER"} and predicted_action != expected_action:
        return "missed_stop_or_transfer"
    if is_write_action(expected_action) or is_write_action(predicted_action):
        if not record.get("action_correct"):
            return "wrong_write_action"
        if expected_tool != predicted_tool:
            return "wrong_write_tool_grounding"
        if not record.get("execution_ok"):
            return "write_execution_error"
    if not record.get("action_correct"):
        return "wrong_policy_action"
    if expected_tool != predicted_tool:
        if expected_tool is None:
            return "extra_tool_call"
        if predicted_tool is None:
            return "missed_tool_call"
        return "wrong_tool_grounding"
    if not record.get("args_valid"):
        return "invalid_arguments"
    if not record.get("execution_ok"):
        return "execution_error"
    return "other"


def task_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row.get("domain")), int(row.get("task_id", -1)), int(row.get("trial", 0)))


def avg(values: list[bool]) -> float:
    return sum(1.0 for value in values if value) / max(1, len(values))


def main() -> None:
    args = parse_args()
    replay = json.loads(Path(args.replay_json).read_text(encoding="utf-8"))
    task_db_match = {task_key(row): bool(row.get("db_hash_match")) for row in replay.get("tasks", [])}
    records_by_task: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in replay.get("records", []):
        records_by_task[task_key(record)].append(record)
    for records in records_by_task.values():
        records.sort(key=lambda row: int(row.get("turn_index") or 0))

    transition_fail: dict[str, list[bool]] = defaultdict(list)
    first_error_types = Counter()
    first_error_transitions = Counter()
    failed_examples: list[dict[str, Any]] = []

    for key, records in sorted(records_by_task.items()):
        db_ok = task_db_match.get(key, False)
        prev_expected: str | None = None
        first_error: dict[str, Any] | None = None
        first_error_type = "none"
        first_error_transition = "none"
        for record in records:
            expected_action = str(record.get("expected_action"))
            trans = transition_name(prev_expected, expected_action)
            transition_fail[trans].append(not db_ok)
            error_type = classify_record(record)
            if first_error is None and error_type != "ok":
                first_error = record
                first_error_type = error_type
                first_error_transition = trans
            prev_expected = expected_action
        if not db_ok:
            first_error_types[first_error_type] += 1
            first_error_transitions[first_error_transition] += 1
            if first_error is not None and len(failed_examples) < 50:
                failed_examples.append(
                    {
                        "task": {"domain": key[0], "task_id": key[1], "trial": key[2]},
                        "first_error_type": first_error_type,
                        "first_error_transition": first_error_transition,
                        "turn_index": first_error.get("turn_index"),
                        "expected_action": first_error.get("expected_action"),
                        "predicted_action": first_error.get("predicted_action"),
                        "expected_tool": first_error.get("expected_tool"),
                        "predicted_tool": first_error.get("predicted_tool"),
                        "execution_ok": first_error.get("execution_ok"),
                        "observation": first_error.get("observation"),
                    }
                )

    transition_rows = [
        {
            "transition_type": trans,
            "count": len(values),
            "db_fail_probability": avg(values),
        }
        for trans, values in transition_fail.items()
    ]
    transition_rows.sort(key=lambda row: (row["db_fail_probability"], row["count"]), reverse=True)
    num_failed = sum(1 for matched in task_db_match.values() if not matched)
    result = {
        "summary": {
            "num_tasks": len(task_db_match),
            "num_db_failed_tasks": num_failed,
            "db_fail_rate": num_failed / max(1, len(task_db_match)),
        },
        "first_error_types": dict(first_error_types.most_common()),
        "first_error_transitions": dict(first_error_transitions.most_common()),
        "transition_db_fail_probability": transition_rows,
        "failed_examples": failed_examples,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.markdown_out:
        md = Path(args.markdown_out)
        md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Trajectory error decomposition",
            "",
            f"- replay: `{args.replay_json}`",
            f"- tasks: {result['summary']['num_tasks']}",
            f"- DB failed tasks: {result['summary']['num_db_failed_tasks']} ({result['summary']['db_fail_rate']:.2%})",
            "",
            "## First Error Type",
            "",
            "| Type | Count |",
            "|---|---:|",
        ]
        lines.extend(f"| {key} | {value} |" for key, value in first_error_types.most_common(20))
        lines.extend(["", "## First Error Transition", "", "| Transition | Count |", "|---|---:|"])
        lines.extend(f"| {key} | {value} |" for key, value in first_error_transitions.most_common(20))
        lines.extend(["", "## P(DB fail | transition type)", "", "| Transition | Count | P(DB fail) |", "|---|---:|---:|"])
        lines.extend(
            f"| {row['transition_type']} | {row['count']} | {row['db_fail_probability']:.2%} |"
            for row in transition_rows[:30]
        )
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
