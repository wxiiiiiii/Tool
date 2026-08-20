from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from universal_agent_policy.data import load_hidden_tensor
from universal_agent_policy.runtime.agent import UniversalAgent
from universal_agent_policy.runtime.argument_generation import (
    EmptyArgumentGenerator,
    HFJsonArgumentGenerator,
    OracleArgumentGenerator,
)
from universal_agent_policy.runtime.policy_runtime import FrozenPolicyRuntime, dtype_from_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-jsonl", required=True)
    parser.add_argument("--hidden-tensor", required=True)
    parser.add_argument("--source-policy", required=True)
    parser.add_argument("--target-adapter")
    parser.add_argument("--out", required=True)
    parser.add_argument("--arg-generator", choices=["empty", "oracle", "hf"], default="oracle")
    parser.add_argument("--arg-model-id")
    parser.add_argument("--arg-device", default="cpu")
    parser.add_argument("--arg-torch-dtype", default="float32", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--arg-max-new-tokens", type=int, default=192)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_rows(path: str | Path, max_samples: int) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def build_arg_generator(args: argparse.Namespace):
    if args.arg_generator == "empty":
        return EmptyArgumentGenerator()
    if args.arg_generator == "oracle":
        return OracleArgumentGenerator()

    if not args.arg_model_id:
        raise ValueError("--arg-model-id is required when --arg-generator hf")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.arg_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.arg_model_id,
        torch_dtype=dtype_from_name(args.arg_torch_dtype),
        device_map=args.arg_device if args.arg_device != "cpu" else None,
    )
    if args.arg_device == "cpu":
        model.to("cpu")
    model.eval()
    return HFJsonArgumentGenerator(model, tokenizer, max_new_tokens=args.arg_max_new_tokens)


def possible_value_match(predicted: Any, possible: Any) -> bool:
    if isinstance(possible, list):
        if isinstance(predicted, list):
            if predicted == possible:
                return True
            for item in possible:
                if isinstance(item, (dict, list)) and possible_value_match(predicted, item):
                    return True
            return False
        return any(not isinstance(item, (dict, list)) and possible_value_match(predicted, item) for item in possible)
    if isinstance(possible, dict):
        if not isinstance(predicted, dict):
            return False
        return all(key in predicted and possible_value_match(predicted[key], value) for key, value in possible.items())
    return predicted == possible


def args_match_possible(predicted: dict[str, Any], ground_truth: list[dict[str, Any]], tool_name: str) -> bool:
    for call in ground_truth:
        if tool_name not in call:
            continue
        possible_args = call[tool_name]
        if all(key in predicted and possible_value_match(predicted[key], value) for key, value in possible_args.items()):
            return True
    return False


def main() -> None:
    args = parse_args()
    rows = load_rows(args.decision_jsonl, args.max_samples)
    tensor = load_hidden_tensor(args.hidden_tensor)
    if args.max_samples > 0:
        tensor["h"] = tensor["h"][: args.max_samples]
        tensor["y"] = tensor["y"][: args.max_samples]
        if tensor.get("sample_id"):
            tensor["sample_id"] = list(tensor["sample_id"])[: args.max_samples]
    if len(rows) != tensor["h"].shape[0]:
        raise ValueError("decision rows and hidden tensor must have the same length")

    row_ids = [row.get("sample_id", "") for row in rows]
    tensor_ids = tensor.get("sample_id")
    if tensor_ids and list(tensor_ids)[: len(row_ids)] != row_ids:
        raise ValueError("decision rows and hidden tensor sample_id order differ")

    if args.target_adapter:
        policy = FrozenPolicyRuntime.from_target_adapter(args.source_policy, args.target_adapter, args.device)
    else:
        policy = FrozenPolicyRuntime.from_source_policy(args.source_policy, args.device)
    agent = UniversalAgent(policy, build_arg_generator(args))

    records = []
    for idx, row in enumerate(rows):
        decision = agent.decide_from_hidden(row, tensor["h"][idx])
        expected_action = int(row["action_id"])
        predicted_action = int(decision.prediction.action.action_id)
        expected_tool_name = None
        if expected_action > 0 and expected_action - 1 < len(row.get("tools", [])):
            expected_tool_name = row["tools"][expected_action - 1].get("name")

        grounded_tool_name = None if decision.grounded.tool is None else decision.grounded.tool.name
        tool_name_correct = grounded_tool_name == expected_tool_name
        args_correct = False
        if decision.tool_call is not None:
            args_correct = args_match_possible(
                decision.tool_call.arguments,
                row.get("ground_truth", []),
                decision.tool_call.name,
            )

        records.append(
            {
                "sample_id": row.get("sample_id"),
                "category": row.get("category"),
                "expected_action": expected_action,
                "predicted_action": predicted_action,
                "action_correct": predicted_action == expected_action,
                "tool_name_correct": tool_name_correct,
                "args_valid": decision.argument_report.valid if decision.argument_report else expected_action == 0,
                "args_correct": args_correct if expected_action > 0 else decision.tool_call is None,
                "predicted_call": None
                if decision.tool_call is None
                else {"name": decision.tool_call.name, "arguments": decision.tool_call.arguments},
                "raw_args": None if decision.argument_report is None else decision.argument_report.raw_text,
                "parsed_args": None if decision.argument_report is None else decision.argument_report.parsed,
                "grounding_reason": decision.grounded.reason,
                "decode_errors": [] if decision.argument_report is None else decision.argument_report.errors,
            }
        )

    def avg(key: str) -> float:
        return sum(1.0 for row in records if row[key]) / max(1, len(records))

    summary = {
        "num_samples": len(records),
        "action_accuracy": avg("action_correct"),
        "tool_name_accuracy": avg("tool_name_correct"),
        "args_valid_rate": avg("args_valid"),
        "args_correct_rate": avg("args_correct"),
        "end_to_end_rate": sum(
            1.0 for row in records if row["action_correct"] and row["tool_name_correct"] and row["args_correct"]
        )
        / max(1, len(records)),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
