from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from baselines.tool_policy_utils import action_names_from_rows, load_prediction_file, load_rows
from universal_agent_policy.eval.tau_replay_agent import (
    build_arg_generator,
    data_hash,
    ensure_tau_import,
    execute_tool,
    historical_tool_call_map,
    load_domain_runtime,
    oracle_ground_tool,
    task_gt_hash,
)
from universal_agent_policy.runtime.constrained_decode import JsonSchemaConstrainedDecoder
from universal_agent_policy.runtime.tau_grounding import TauActionGrounder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay tau-bench with precomputed baseline action predictions. "
            "This keeps grounding, argument generation, constrained decoding, execution, and DB-hash "
            "evaluation identical to the frozen-policy replay path."
        )
    )
    parser.add_argument("--decision-jsonl", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--trajectories", nargs="+", required=True)
    parser.add_argument("--tau-bench-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arg-generator", choices=["empty", "oracle", "hf"], default="oracle")
    parser.add_argument("--grounder", choices=["semantic", "oracle"], default="semantic")
    parser.add_argument("--arg-model-id")
    parser.add_argument("--arg-device", default="cpu")
    parser.add_argument("--arg-torch-dtype", default="float32", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--arg-max-new-tokens", type=int, default=192)
    parser.add_argument("--max-samples", type=int, default=-1)
    return parser.parse_args()


def avg(items: list[dict[str, Any]], key: str) -> float:
    return sum(1.0 for item in items if item[key]) / max(1, len(items))


def main() -> None:
    args = parse_args()
    ensure_tau_import(args.tau_bench_path)
    rows = load_rows(args.decision_jsonl, args.max_samples)
    predictions_obj = load_prediction_file(args.predictions)
    prediction_rows = predictions_obj["records"][: len(rows)]
    if len(prediction_rows) != len(rows):
        raise ValueError("prediction records and decision rows must have the same length")

    historical_calls = historical_tool_call_map(args.trajectories)
    for row in rows:
        row["historical_tool_calls"] = historical_calls.get(str(row.get("sample_id", "")), [])

    action_names = action_names_from_rows(rows)
    action_to_id = {name: idx for idx, name in enumerate(action_names)}
    grounder = TauActionGrounder()
    decoder = JsonSchemaConstrainedDecoder()
    arg_generator = build_arg_generator(args)
    domains = {domain: load_domain_runtime(domain) for domain in sorted({str(row.get("domain")) for row in rows})}

    task_states: dict[tuple[str, int, int], dict[str, Any]] = {}
    expected_task_states: dict[tuple[str, int, int], dict[str, Any]] = {}
    task_step_counts: dict[tuple[str, int, int], int] = defaultdict(int)
    expected_step_counts: dict[tuple[str, int, int], int] = defaultdict(int)
    records: list[dict[str, Any]] = []

    for row, prediction in zip(rows, prediction_rows, strict=True):
        if str(row.get("sample_id")) != str(prediction.get("sample_id")):
            raise ValueError("prediction sample_id order does not match decision rows")
        domain = str(row.get("domain"))
        runtime = domains[domain]
        task_key = (domain, int(row.get("task_id", -1)), int(row.get("trial", 0)))
        if task_key not in task_states:
            task_states[task_key] = runtime.load_data()
            expected_task_states[task_key] = runtime.load_data()

        for historical_call in row.get("historical_tool_calls", []):
            historical_tool = historical_call.get("name")
            if historical_tool in runtime.tool_map and historical_tool not in runtime.terminate_tools:
                try:
                    execute_tool(runtime, expected_task_states[task_key], historical_tool, historical_call.get("arguments", {}))
                    expected_step_counts[task_key] += 1
                except Exception:
                    pass

        predicted_action = str(prediction.get("predicted_action", ""))
        expected_action = str(row.get("correct_action", ""))
        predicted_id = action_to_id.get(predicted_action, -1)
        expected_id = int(row.get("action_id", action_to_id.get(expected_action, -1)))
        action_correct = predicted_id == expected_id

        if args.grounder == "oracle":
            oracle_tool = oracle_ground_tool(row, action_correct, runtime)
            grounded = type(
                "OracleGrounding",
                (),
                {
                    "tool": oracle_tool,
                    "reason": "oracle_grounding" if oracle_tool is not None else "oracle_no_tool",
                },
            )()
        else:
            grounded = grounder.ground(predicted_action, runtime.tool_specs, row)

        expected_tool = row.get("expected_tool_name")
        predicted_tool = None if grounded.tool is None else grounded.tool.name
        report = None
        execution_ok = False
        observation = None
        if grounded.tool is not None:
            raw_args = arg_generator.generate(row, grounded.tool)
            report = decoder.decode(raw_args, grounded.tool.parameters)
            if report.valid:
                try:
                    observation = execute_tool(runtime, task_states[task_key], grounded.tool.name, report.constrained)
                    execution_ok = not observation.startswith("Error:")
                    task_step_counts[task_key] += 1
                except Exception as exc:
                    observation = f"Error: {exc}"

        records.append(
            {
                "sample_id": row.get("sample_id"),
                "domain": domain,
                "task_id": row.get("task_id"),
                "trial": row.get("trial", 0),
                "turn_index": row.get("turn_index"),
                "expected_action": expected_action,
                "predicted_action": predicted_action,
                "action_correct": action_correct,
                "expected_tool": expected_tool,
                "predicted_tool": predicted_tool,
                "tool_grounding_correct": predicted_tool == expected_tool,
                "args_valid": True if report is None else report.valid,
                "execution_ok": execution_ok if predicted_tool is not None else expected_tool is None,
                "observation": observation,
                "arguments": None if report is None else report.constrained,
                "raw_args": None if report is None else report.raw_text,
                "decode_errors": [] if report is None else report.errors,
                "grounding_reason": grounded.reason,
            }
        )

    task_records = []
    for task_key, data in sorted(task_states.items()):
        domain, task_id, trial = task_key
        runtime = domains[domain]
        predicted_hash = data_hash(data)
        expected_hash = data_hash(expected_task_states[task_key])
        formal_gt_hash = task_gt_hash(runtime, task_id)
        task_records.append(
            {
                "domain": domain,
                "task_id": task_id,
                "trial": trial,
                "num_executed_steps": task_step_counts[task_key],
                "num_expected_steps": expected_step_counts[task_key],
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
        "method": predictions_obj.get("summary", {}).get("method", "precomputed_predictions"),
        "prediction_file": args.predictions,
        "num_samples": len(records),
        "num_tasks": len(task_records),
        "action_accuracy": avg(records, "action_correct"),
        "tool_grounding_accuracy_all": avg(records, "tool_grounding_correct"),
        "tool_grounding_accuracy_tool_only": avg(tool_records, "tool_grounding_correct"),
        "args_valid_rate_predicted_tool": avg(predicted_tool_records, "args_valid"),
        "tool_execution_ok_rate_predicted_tool": avg(predicted_tool_records, "execution_ok"),
        "db_hash_match_rate": avg(task_records, "db_hash_match"),
        "formal_gt_hash_match_rate": avg(task_records, "formal_gt_hash_match"),
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
