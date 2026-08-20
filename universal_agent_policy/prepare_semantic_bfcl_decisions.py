from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from universal_agent_policy.runtime.semantic_grounding import (
    ANSWER,
    SEMANTIC_ACTION_NAMES,
    infer_tool_action,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-per-action", type=int, default=-1)
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def expected_tool_name(row: dict[str, Any]) -> str | None:
    action_id = int(row.get("action_id", 0))
    tools = list(row.get("tools", []))
    if action_id > 0 and action_id - 1 < len(tools):
        return str(tools[action_id - 1].get("name", ""))
    for call in row.get("ground_truth", []):
        if call:
            return next(iter(call.keys()))
    return None


def expected_tool(row: dict[str, Any]) -> dict[str, Any] | None:
    name = expected_tool_name(row)
    if not name:
        return None
    for tool in row.get("tools", []):
        if str(tool.get("name", "")) == name:
            return tool
    return None


def semantic_action_for_row(row: dict[str, Any]) -> str:
    if int(row.get("action_id", 0)) == 0:
        return ANSWER
    tool = expected_tool(row)
    if tool is None:
        return ANSWER
    return infer_tool_action(tool)


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    action_name = semantic_action_for_row(row)
    action_id = SEMANTIC_ACTION_NAMES.index(action_name)
    converted = dict(row)
    converted["slot_action_id"] = row.get("action_id")
    converted["slot_correct_action"] = row.get("correct_action")
    converted["expected_tool_name"] = expected_tool_name(row)
    converted["correct_action"] = action_name
    converted["action_id"] = action_id
    converted["action_names"] = SEMANTIC_ACTION_NAMES
    return converted


def balance_rows(rows: list[dict[str, Any]], max_per_action: int) -> list[dict[str, Any]]:
    if max_per_action <= 0:
        return rows
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[int(row["action_id"])].append(row)
    balanced = []
    for action_id in sorted(buckets):
        balanced.extend(buckets[action_id][:max_per_action])
    return balanced


def main() -> None:
    args = parse_args()
    rows = [convert_row(row) for row in load_jsonl(args.input_jsonl)]
    rows = balance_rows(rows, args.max_per_action)
    counts = Counter(row["correct_action"] for row in rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "out": str(out),
                "num_rows": len(rows),
                "action_counts": dict(sorted(counts.items())),
                "action_names": SEMANTIC_ACTION_NAMES,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
