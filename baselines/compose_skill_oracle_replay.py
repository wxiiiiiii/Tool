from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from baselines import eval_protocol
from baselines.tool_policy_utils import action_names_from_rows, load_prediction_file, load_rows
from universal_agent_policy.eval.tau_replay_agent import (
    data_hash,
    ensure_tau_import,
    execute_tool,
    load_domain_runtime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compose single-class oracle-skill replays from valid static and full-oracle replay records. "
            "This is used when the original tau trajectories are unavailable, but previous valid replay "
            "records already contain the constrained tool arguments for static and oracle choices."
        )
    )
    parser.add_argument("--decision-jsonl", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--static-replay", required=True)
    parser.add_argument("--oracle-replay", required=True)
    parser.add_argument("--tau-bench-path", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def avg(items: list[dict[str, Any]], key: str) -> float:
    return sum(1.0 for item in items if item[key]) / max(1, len(items))


def task_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row.get("domain")), int(row.get("task_id", -1)), int(row.get("trial", 0)))


def load_replay(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    ensure_tau_import(args.tau_bench_path)
    all_rows = load_rows(args.decision_jsonl)
    _, val_idx, split_metadata = eval_protocol.load_split(args.split_json, len(all_rows))
    selected = [int(idx) for idx in val_idx.tolist()]
    rows = [all_rows[idx] for idx in selected]
    predictions_obj = load_prediction_file(args.predictions)
    all_predictions = predictions_obj["records"]
    predictions = [all_predictions[idx] for idx in selected]

    static_replay = load_replay(args.static_replay)
    oracle_replay = load_replay(args.oracle_replay)
    static_records = {str(record["sample_id"]): record for record in static_replay["records"]}
    oracle_records = {str(record["sample_id"]): record for record in oracle_replay["records"]}
    static_tasks = {task_key(task): task for task in static_replay["tasks"]}

    action_names = action_names_from_rows(rows)
    action_to_id = {name: idx for idx, name in enumerate(action_names)}
    domains = {domain: load_domain_runtime(domain) for domain in sorted({str(row.get("domain")) for row in rows})}
    task_states: dict[tuple[str, int, int], dict[str, Any]] = {}
    task_step_counts: dict[tuple[str, int, int], int] = defaultdict(int)
    records: list[dict[str, Any]] = []

    for row, prediction in zip(rows, predictions, strict=True):
        sample_id = str(row.get("sample_id"))
        predicted_action = str(prediction.get("predicted_action", ""))
        expected_action = str(row.get("correct_action", ""))
        predicted_id = action_to_id.get(predicted_action, -1)
        expected_id = int(row.get("action_id", action_to_id.get(expected_action, -1)))
        action_correct = predicted_id == expected_id
        domain = str(row.get("domain"))
        runtime = domains[domain]
        key = task_key(row)
        if key not in task_states:
            task_states[key] = runtime.load_data()

        static_record = static_records[sample_id]
        oracle_record = oracle_records[sample_id]
        if predicted_action == str(oracle_record.get("predicted_action")):
            chosen = oracle_record
            source = "oracle"
        else:
            chosen = static_record
            source = "static"

        predicted_tool = chosen.get("predicted_tool")
        arguments = chosen.get("arguments")
        observation = None
        execution_ok = False
        if predicted_tool is not None and predicted_tool in runtime.tool_map and predicted_tool not in runtime.terminate_tools:
            try:
                observation = execute_tool(runtime, task_states[key], str(predicted_tool), arguments or {})
                execution_ok = not str(observation).startswith("Error:")
                task_step_counts[key] += 1
            except Exception as exc:
                observation = f"Error: {exc}"
        elif predicted_tool is None:
            execution_ok = row.get("expected_tool_name") is None

        records.append(
            {
                "sample_id": sample_id,
                "domain": domain,
                "task_id": row.get("task_id"),
                "trial": row.get("trial", 0),
                "turn_index": row.get("turn_index"),
                "expected_action": expected_action,
                "predicted_action": predicted_action,
                "action_correct": action_correct,
                "expected_tool": row.get("expected_tool_name"),
                "predicted_tool": predicted_tool,
                "tool_grounding_correct": predicted_tool == row.get("expected_tool_name"),
                "args_valid": arguments is not None or predicted_tool is None,
                "execution_ok": execution_ok,
                "observation": observation,
                "arguments": arguments,
                "raw_args": chosen.get("raw_args"),
                "decode_errors": chosen.get("decode_errors", []),
                "grounding_reason": f"composed_from_{source}_valid_replay",
            }
        )

    task_records = []
    for key, data in sorted(task_states.items()):
        domain, task_id, trial = key
        predicted_hash = data_hash(data)
        reference = static_tasks[key]
        expected_hash = reference["expected_hash"]
        formal_gt_hash = reference.get("formal_gt_hash")
        task_records.append(
            {
                "domain": domain,
                "task_id": task_id,
                "trial": trial,
                "num_executed_steps": task_step_counts[key],
                "num_expected_steps": reference.get("num_expected_steps", 0),
                "predicted_hash": predicted_hash,
                "expected_hash": expected_hash,
                "formal_gt_hash": formal_gt_hash,
                "db_hash_match": predicted_hash == expected_hash,
                "formal_gt_hash_match": formal_gt_hash is not None and predicted_hash == formal_gt_hash,
            }
        )

    tool_records = [row for row in records if row["expected_tool"] is not None]
    predicted_tool_records = [row for row in records if row["predicted_tool"] is not None]
    summary = {
        "method": predictions_obj.get("summary", {}).get("method", "composed_predictions"),
        "prediction_file": args.predictions,
        "split": split_metadata,
        "num_samples": len(records),
        "num_tasks": len(task_records),
        "action_accuracy": avg(records, "action_correct"),
        "tool_grounding_accuracy_all": avg(records, "tool_grounding_correct"),
        "tool_grounding_accuracy_tool_only": avg(tool_records, "tool_grounding_correct"),
        "args_valid_rate_predicted_tool": avg(predicted_tool_records, "args_valid"),
        "tool_execution_ok_rate_predicted_tool": avg(predicted_tool_records, "execution_ok"),
        "db_hash_match_rate": avg(task_records, "db_hash_match"),
        "formal_gt_hash_match_rate": avg(task_records, "formal_gt_hash_match"),
        "composition_note": "arguments/tools are selected from valid static or full-oracle replay records",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "tasks": task_records, "records": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
