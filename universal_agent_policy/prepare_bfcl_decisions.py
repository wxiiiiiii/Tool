from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ACTION_NAMES = ["NO_TOOL", "SELECT_TOOL_0", "SELECT_TOOL_1", "SELECT_TOOL_2", "SELECT_TOOL_3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-per-action", type=int, default=80)
    parser.add_argument("--max-tool-slots", type=int, default=2)
    parser.add_argument("--include", nargs="+", default=["multiple", "live_multiple", "irrelevance", "live_irrelevance"])
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def flatten_messages(question: Any) -> list[dict[str, str]]:
    if isinstance(question, list) and question and isinstance(question[0], list):
        return question[0]
    if isinstance(question, list):
        return question
    return [{"role": "user", "content": str(question)}]


def tool_names(row: dict[str, Any]) -> list[str]:
    return [fn["name"] for fn in row.get("function", [])]


def ground_truth_names(answer: dict[str, Any] | None) -> list[str]:
    if not answer:
        return []
    names = []
    for call in answer.get("ground_truth", []):
        names.extend(call.keys())
    return names


def prompt_content(messages: list[dict[str, str]], tools: list[dict[str, Any]]) -> str:
    user_text = "\n".join(f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages)
    compact_tools = [
        {
            "slot": idx,
            "name": tool.get("name"),
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {}),
        }
        for idx, tool in enumerate(tools)
    ]
    return (
        "Decide which tool slot should be used for the user request. "
        "Return NO_TOOL if none of the tools are relevant.\n\n"
        f"Conversation:\n{user_text}\n\n"
        f"Tools:\n{json.dumps(compact_tools, ensure_ascii=False, sort_keys=True)}"
    )


def build_rows(raw_dir: Path, include: list[str], max_tool_slots: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category in include:
        data_path = raw_dir / f"BFCL_v3_{category}.json"
        if not data_path.exists():
            raise FileNotFoundError(data_path)
        answer_path = raw_dir / "possible_answer" / f"BFCL_v3_{category}.json"
        answers = {}
        if answer_path.exists():
            answers = {row["id"]: row for row in load_jsonl(answer_path)}

        for row in load_jsonl(data_path):
            names = tool_names(row)
            if len(names) > max_tool_slots:
                continue

            answer = answers.get(row["id"])
            gt_names = ground_truth_names(answer)
            if category.endswith("irrelevance"):
                action_id = 0
            elif len(gt_names) != 1 or gt_names[0] not in names:
                continue
            else:
                action_id = names.index(gt_names[0]) + 1

            if action_id >= len(ACTION_NAMES):
                continue

            messages = flatten_messages(row.get("question", ""))
            tools = row.get("function", [])
            rows.append(
                {
                    "sample_id": row["id"],
                    "category": category,
                    "original_question": row.get("question", ""),
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt_content(messages, tools),
                        }
                    ],
                    "prompt": prompt_content(messages, tools),
                    "tools": tools,
                    "tool_names": names,
                    "ground_truth": (answer or {}).get("ground_truth", []),
                    "correct_action": ACTION_NAMES[action_id],
                    "action_id": action_id,
                    "action_names": ACTION_NAMES[: max_tool_slots + 1],
                }
            )
    return rows


def balance_rows(rows: list[dict[str, Any]], max_per_action: int) -> list[dict[str, Any]]:
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[int(row["action_id"])].append(row)

    balanced = []
    for action_id in sorted(buckets):
        balanced.extend(buckets[action_id][:max_per_action])
    return balanced


def main() -> None:
    args = parse_args()
    rows = build_rows(Path(args.raw_dir), args.include, args.max_tool_slots)
    rows = balance_rows(rows, args.max_per_action)
    counts = Counter(row["action_id"] for row in rows)

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
                "action_counts": {ACTION_NAMES[key]: value for key, value in sorted(counts.items())},
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
