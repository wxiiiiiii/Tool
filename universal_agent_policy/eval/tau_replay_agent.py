from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from universal_agent_policy.data import load_hidden_tensor
from universal_agent_policy.runtime.argument_generation import (
    EmptyArgumentGenerator,
    HFJsonArgumentGenerator,
)
from universal_agent_policy.runtime.constrained_decode import JsonSchemaConstrainedDecoder
from universal_agent_policy.runtime.policy_runtime import FrozenPolicyRuntime, dtype_from_name
from universal_agent_policy.runtime.tau_grounding import TauActionGrounder, tau_tools_to_specs


@dataclass
class DomainRuntime:
    tools: list[type]
    tools_info: list[dict[str, Any]]
    tool_specs: list[Any]
    tool_map: dict[str, type]
    load_data: Any
    tasks: list[Any]
    terminate_tools: set[str]


class TauOracleArgumentGenerator:
    def generate(self, row: dict[str, Any], tool: Any) -> str:
        calls = row.get("historical_tool_calls", [])
        for call in calls:
            if call.get("name") == tool.name:
                return json.dumps(call.get("arguments", {}), ensure_ascii=False)
        return "{}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-jsonl", required=True)
    parser.add_argument("--hidden-tensor", required=True)
    parser.add_argument("--source-policy", required=True)
    parser.add_argument("--target-adapter")
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
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def ensure_tau_import(tau_bench_path: str | Path) -> None:
    path = Path(tau_bench_path).resolve()
    package_parent = path if (path / "tau_bench").exists() else path.parent
    sys.path.insert(0, str(package_parent))
    if "pydantic" not in sys.modules:
        pydantic_stub = types.ModuleType("pydantic")

        class BaseModel:
            def __init__(self, **kwargs: Any) -> None:
                for key, value in kwargs.items():
                    setattr(self, key, value)

            def model_dump(self) -> dict[str, Any]:
                return dict(self.__dict__)

        pydantic_stub.BaseModel = BaseModel
        sys.modules["pydantic"] = pydantic_stub
    if "litellm" not in sys.modules:
        litellm_stub = types.ModuleType("litellm")
        litellm_stub.provider_list = []

        def _completion(*_: Any, **__: Any) -> Any:
            raise RuntimeError("litellm completion is unavailable in tau replay mode")

        litellm_stub.completion = _completion
        sys.modules["litellm"] = litellm_stub


def load_rows(path: str | Path, max_samples: int) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def domain_from_path(path: str | Path) -> str:
    name = Path(path).name.lower()
    if "airline" in name:
        return "airline"
    if "retail" in name:
        return "retail"
    return "unknown"


def parse_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function", call)
        raw_args = function.get("arguments", {})
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
        calls.append({"name": str(function.get("name", call.get("name", ""))), "arguments": args})
    return calls


def historical_tool_call_map(paths: list[str]) -> dict[str, list[dict[str, Any]]]:
    mapping: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        domain = domain_from_path(path)
        with Path(path).open("r", encoding="utf-8") as handle:
            trajectories = json.load(handle)
        if isinstance(trajectories, dict) and isinstance(trajectories.get("records"), list):
            trajectories = trajectories["records"]
        for item in trajectories:
            task_id = item.get("task_id")
            trial = item.get("trial", 0)
            for idx, message in enumerate(item.get("traj", [])):
                if message.get("role") != "assistant":
                    continue
                sample_id = f"{domain}_{task_id}_{trial}_{idx}"
                mapping[sample_id] = parse_tool_calls(message)
    return mapping


def load_domain_runtime(domain: str) -> DomainRuntime:
    if domain == "retail":
        from tau_bench.envs.retail.data import load_data
        from tau_bench.envs.retail.tools import ALL_TOOLS

        task_candidates = [
            ("tau_bench.envs.retail.tasks_test", "TASKS_TEST"),
            ("tau_bench.envs.retail.tasks_dev", "TASKS_DEV"),
            ("tau_bench.envs.retail.tasks", "TASKS"),
            ("tau_bench.envs.retail.tasks_train", "TASKS_TRAIN"),
        ]
    elif domain == "airline":
        from tau_bench.envs.airline.data import load_data
        from tau_bench.envs.airline.tools import ALL_TOOLS

        task_candidates = [
            ("tau_bench.envs.airline.tasks_test", "TASKS"),
            ("tau_bench.envs.airline.tasks", "TASKS"),
        ]
    else:
        raise ValueError(f"Unknown tau domain: {domain}")

    tasks: list[Any] = []
    for module_name, attr in task_candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        candidate = list(getattr(module, attr, []))
        if len(candidate) > len(tasks):
            tasks = candidate

    tools_info = [tool.get_info() for tool in ALL_TOOLS]
    tool_specs = tau_tools_to_specs(tools_info)
    tool_map = {spec.name: tool for spec, tool in zip(tool_specs, ALL_TOOLS, strict=True)}
    return DomainRuntime(
        tools=list(ALL_TOOLS),
        tools_info=tools_info,
        tool_specs=tool_specs,
        tool_map=tool_map,
        load_data=load_data,
        tasks=tasks,
        terminate_tools={"transfer_to_human_agents"},
    )


def action_names_from_rows(rows: list[dict[str, Any]]) -> list[str] | None:
    for row in rows:
        names = row.get("action_names")
        if names:
            return list(names)
    return None


def build_arg_generator(args: argparse.Namespace):
    if args.arg_generator == "empty":
        return EmptyArgumentGenerator()
    if args.arg_generator == "oracle":
        return TauOracleArgumentGenerator()

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


def oracle_ground_tool(row: dict[str, Any], action_correct: bool, runtime: DomainRuntime) -> Any | None:
    if not action_correct:
        return None
    expected_tool = row.get("expected_tool_name")
    if not expected_tool:
        return None
    for spec in runtime.tool_specs:
        if spec.name == expected_tool:
            return spec
    return None


def execute_tool(runtime: DomainRuntime, data: dict[str, Any], tool_name: str, arguments: dict[str, Any]) -> str:
    tool = runtime.tool_map[tool_name]
    return str(tool.invoke(data=data, **arguments))


def data_hash(data: dict[str, Any]) -> str:
    from tau_bench.envs.base import consistent_hash, to_hashable

    return consistent_hash(to_hashable(data))


def task_gt_hash(runtime: DomainRuntime, task_id: int) -> str | None:
    if task_id < 0 or task_id >= len(runtime.tasks):
        return None
    data = runtime.load_data()
    for action in runtime.tasks[task_id].actions:
        if action.name == "respond" or action.name in runtime.terminate_tools:
            continue
        if action.name not in runtime.tool_map:
            continue
        execute_tool(runtime, data, action.name, dict(action.kwargs))
    return data_hash(data)


def main() -> None:
    args = parse_args()
    ensure_tau_import(args.tau_bench_path)

    rows = load_rows(args.decision_jsonl, args.max_samples)
    historical_calls = historical_tool_call_map(args.trajectories)
    for row in rows:
        row["historical_tool_calls"] = historical_calls.get(str(row.get("sample_id", "")), [])

    tensor = load_hidden_tensor(args.hidden_tensor)
    if args.max_samples > 0:
        tensor["h"] = tensor["h"][: args.max_samples]
        tensor["y"] = tensor["y"][: args.max_samples]
        if tensor.get("sample_id"):
            tensor["sample_id"] = list(tensor["sample_id"])[: args.max_samples]
    if len(rows) != tensor["h"].shape[0]:
        raise ValueError("decision rows and hidden tensor must have the same length")

    tensor_ids = tensor.get("sample_id")
    if tensor_ids:
        row_ids = [row.get("sample_id", "") for row in rows]
        if list(tensor_ids)[: len(row_ids)] != row_ids:
            raise ValueError("decision rows and hidden tensor sample_id order differ")

    policy = (
        FrozenPolicyRuntime.from_target_adapter(args.source_policy, args.target_adapter, args.device)
        if args.target_adapter
        else FrozenPolicyRuntime.from_source_policy(args.source_policy, args.device)
    )
    action_names = action_names_from_rows(rows) or list(policy.action_space.names)
    grounder = TauActionGrounder()
    decoder = JsonSchemaConstrainedDecoder()
    arg_generator = build_arg_generator(args)
    domains = {domain: load_domain_runtime(domain) for domain in sorted({str(row.get("domain")) for row in rows})}

    task_states: dict[tuple[str, int, int], dict[str, Any]] = {}
    expected_task_states: dict[tuple[str, int, int], dict[str, Any]] = {}
    task_step_counts: dict[tuple[str, int, int], int] = defaultdict(int)
    expected_step_counts: dict[tuple[str, int, int], int] = defaultdict(int)
    records = []
    for idx, row in enumerate(rows):
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

        prediction = policy.predict_from_hidden(tensor["h"][idx])
        predicted_id = int(prediction.action.action_id)
        expected_id = int(row.get("action_id", tensor["y"][idx].item()))
        predicted_action = action_names[predicted_id] if 0 <= predicted_id < len(action_names) else prediction.action.name
        expected_action = str(row.get("correct_action", action_names[expected_id]))

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

    def avg(items: list[dict[str, Any]], key: str) -> float:
        return sum(1.0 for item in items if item[key]) / max(1, len(items))

    tool_records = [row for row in records if row["expected_tool"] is not None]
    predicted_tool_records = [row for row in records if row["predicted_tool"] is not None]
    summary = {
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
