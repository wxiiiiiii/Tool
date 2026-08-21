from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from baselines import eval_protocol
from baselines.hidden_policy_baselines import RidgeProjection, load_source_parts
from universal_agent_policy.data import load_hidden_tensor


OBSERVATIONS = {
    "success": {
        "status": "success",
        "message": "The tool completed successfully and the requested state change is now committed.",
        "result": {"ok": True, "updated": True},
    },
    "failure": {
        "status": "error",
        "error": "tool_execution_failed",
        "message": "The tool failed and no state change was committed.",
    },
    "empty": {
        "status": "success",
        "message": "The tool returned no matching records.",
        "result": [],
    },
    "conflict": {
        "status": "conflict",
        "message": "The tool result conflicts with the requested operation and requires verification.",
        "result": {"ok": False, "requires_verification": True},
    },
}

BRANCH_NAMES = ["finish", "recover", "acquire", "verify", "unsafe"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-jsonl", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--source-policy", required=True)
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--target-tensor", required=True)
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM3-3B")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-prefixes", type=int, default=-1)
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument(
        "--existing-hidden-observation-audit",
        action="store_true",
        help=(
            "Use already-extracted target hidden states for real held-out observations. "
            "This is a fallback audit, not the fixed-prefix counterfactual benchmark."
        ),
    )
    return parser.parse_args()


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pct(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{100.0 * value:.2f}%"


def action_names_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    for row in rows:
        names = row.get("action_names")
        if names:
            return list(names)
    raise ValueError("No action_names found in rows")


def conversation_bounds(prompt: str) -> tuple[int, int] | None:
    marker = "Conversation state before the next assistant decision:\n"
    end_marker = "\n\nNext policy action:"
    start = prompt.find(marker)
    end = prompt.find(end_marker, start + len(marker))
    if start < 0 or end < 0:
        return None
    return start + len(marker), end


def last_tool_line_info(prompt: str) -> tuple[int, int, str] | None:
    bounds = conversation_bounds(prompt)
    if bounds is None:
        return None
    start, end = bounds
    context = prompt[start:end]
    line_start = 0
    found: tuple[int, int, str] | None = None
    for line in context.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.lower().startswith("tool "):
            absolute_start = start + line_start
            found = (absolute_start, absolute_start + len(line.rstrip("\n")), stripped)
        line_start += len(line)
    return found


def last_tool_call_kind(prompt: str) -> str:
    matches = re.findall(r"assistant tool_call:\s*([A-Za-z0-9_]+)\(", prompt)
    if not matches:
        return "none"
    name = matches[-1].lower()
    if name.startswith(("book_", "cancel_", "exchange_", "modify_", "return_", "send_", "update_")):
        return "write"
    if name.startswith(("find_", "get_", "list_")):
        return "retrieve"
    if name.startswith("search_"):
        return "search"
    if name == "calculate":
        return "compute"
    if name == "think":
        return "think"
    if "transfer" in name:
        return "transfer"
    return "other"


def replace_last_tool_observation(prompt: str, observation_type: str) -> str:
    info = last_tool_line_info(prompt)
    if info is None:
        raise ValueError("Prompt has no tool observation line")
    start, end, line = info
    prefix = line.split(":", 1)[0] if ":" in line else "tool counterfactual"
    payload = json.dumps(OBSERVATIONS[observation_type], ensure_ascii=False, sort_keys=True)
    replacement = f"{prefix}: {payload}"
    return prompt[:start] + replacement + prompt[end:]


def infer_observation_type(prompt: str) -> str:
    info = last_tool_line_info(prompt)
    if info is None:
        return "none"
    text = info[2].lower()
    if any(term in text for term in ["conflict", "inconsistent", "requires verification"]):
        return "conflict"
    if any(term in text for term in ["error", "failed", "failure", "exception", "invalid", "denied"]):
        return "failure"
    if any(term in text for term in ["not found", "no matching", "no result", "empty", "[]"]):
        return "empty"
    return "success"


def build_existing_hidden_audit_rows(
    rows: list[dict[str, Any]],
    val_idx: torch.Tensor,
    max_prefixes: int,
) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    selected = []
    for idx in val_idx.tolist():
        row = rows[int(idx)]
        prompt = str(row.get("prompt", ""))
        if last_tool_line_info(prompt) is None:
            continue
        selected.append(int(idx))
    if max_prefixes > 0:
        selected = selected[:max_prefixes]

    audit_rows = []
    for original_idx in selected:
        row = dict(rows[original_idx])
        prompt = str(row["prompt"])
        row["original_index"] = original_idx
        row["counterfactual_observation"] = infer_observation_type(prompt)
        row["last_tool_kind"] = last_tool_call_kind(prompt)
        audit_rows.append(row)
    meta = {
        "mode": "existing_hidden_observation_audit",
        "not_counterfactual": True,
        "num_candidate_prefixes": len(selected),
        "num_counterfactual_rows": len(audit_rows),
        "tool_kind_counts": dict(Counter(row["last_tool_kind"] for row in audit_rows)),
        "observation_type_counts": dict(Counter(row["counterfactual_observation"] for row in audit_rows)),
    }
    return audit_rows, selected, meta


def build_counterfactual_rows(
    rows: list[dict[str, Any]],
    val_idx: torch.Tensor,
    max_prefixes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = []
    for idx in val_idx.tolist():
        row = rows[int(idx)]
        if last_tool_line_info(str(row.get("prompt", ""))) is None:
            continue
        selected.append(int(idx))
    if max_prefixes > 0:
        selected = selected[:max_prefixes]

    cf_rows = []
    for original_idx in selected:
        row = rows[original_idx]
        prompt = str(row["prompt"])
        tool_kind = last_tool_call_kind(prompt)
        for obs_type in OBSERVATIONS:
            cf_row = dict(row)
            cf_row["original_index"] = original_idx
            cf_row["sample_id"] = f"{row.get('sample_id', original_idx)}__cf_{obs_type}"
            cf_row["counterfactual_observation"] = obs_type
            cf_row["last_tool_kind"] = tool_kind
            cf_row["prompt"] = replace_last_tool_observation(prompt, obs_type)
            cf_rows.append(cf_row)
    meta = {
        "num_candidate_prefixes": len(selected),
        "num_counterfactual_rows": len(cf_rows),
        "tool_kind_counts": dict(Counter(last_tool_call_kind(str(rows[i].get("prompt", ""))) for i in selected)),
    }
    return cf_rows, meta


def dtype_from_name(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def extract_hidden_states(
    rows: list[dict[str, Any]],
    model_id: str,
    layer: int,
    batch_size: int,
    max_input_tokens: int,
    torch_dtype: str,
) -> torch.Tensor:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=dtype_from_name(torch_dtype),
    )
    model.eval()

    outputs: list[torch.Tensor] = []
    prompts = [str(row["prompt"]) for row in rows]
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_tokens,
            )
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            result = model(**inputs, output_hidden_states=True)
            hidden = result.hidden_states[layer]
            last_idx = inputs["attention_mask"].sum(dim=1) - 1
            states = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_idx]
            outputs.append(states.detach().float().cpu())
    return torch.cat(outputs, dim=0)


def branch_for_action(action: str) -> str:
    if action in {"ANSWER", "STOP"}:
        return "finish"
    if action in {"ASK_USER", "TRANSFER", "THINK"}:
        return "recover"
    if action == "VERIFY":
        return "verify"
    if action.startswith(("ACT_RETRIEVE", "ACT_SEARCH", "ACT_COMPUTE")):
        return "acquire"
    if action.startswith(("ACT_UPDATE", "ACT_SEND")):
        return "unsafe"
    return "recover"


def expected_branches(observation_type: str, tool_kind: str) -> set[str]:
    if observation_type == "success":
        if tool_kind == "write":
            return {"finish"}
        if tool_kind in {"retrieve", "search", "compute"}:
            return {"verify", "unsafe", "finish", "acquire"}
        return {"finish", "verify"}
    if observation_type == "failure":
        return {"recover", "verify", "acquire"}
    if observation_type == "empty":
        return {"recover", "verify", "acquire"}
    if observation_type == "conflict":
        return {"verify", "recover"}
    return set(BRANCH_NAMES)


def branch_allowed_action_ids(action_names: list[str], observation_type: str, tool_kind: str) -> list[int]:
    branches = expected_branches(observation_type, tool_kind)
    return [idx for idx, action in enumerate(action_names) if branch_for_action(action) in branches]


def constrained_branch_predictions(logits: torch.Tensor, rows: list[dict[str, Any]], action_names: list[str]) -> torch.Tensor:
    adjusted = logits.clone()
    for idx, row in enumerate(rows):
        allowed = branch_allowed_action_ids(action_names, row["counterfactual_observation"], row["last_tool_kind"])
        mask = torch.ones(len(action_names), dtype=torch.bool)
        mask[allowed] = False
        adjusted[idx, mask] = -1e9
    return adjusted.argmax(dim=-1)


def predictions_to_actions(pred: torch.Tensor, action_names: list[str]) -> list[str]:
    return [action_names[int(item)] for item in pred.tolist()]


def branch_accuracy(rows: list[dict[str, Any]], actions: list[str], only_tool_kind: str | None = None) -> float:
    total = 0
    correct = 0
    for row, action in zip(rows, actions):
        if only_tool_kind is not None and row["last_tool_kind"] != only_tool_kind:
            continue
        total += 1
        correct += int(branch_for_action(action) in expected_branches(row["counterfactual_observation"], row["last_tool_kind"]))
    return correct / total if total else float("nan")


def action_sensitivity(rows: list[dict[str, Any]], actions: list[str], only_tool_kind: str | None = None) -> dict[str, float]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for row, action in zip(rows, actions):
        if only_tool_kind is not None and row["last_tool_kind"] != only_tool_kind:
            continue
        grouped[int(row["original_index"])].append(action)
    if not grouped:
        return {"unique_action_rate": float("nan"), "avg_unique_actions": float("nan")}
    unique_counts = [len(set(items)) for items in grouped.values()]
    return {
        "unique_action_rate": sum(1 for count in unique_counts if count > 1) / len(unique_counts),
        "avg_unique_actions": sum(unique_counts) / len(unique_counts),
    }


def pair_branch_rate(rows: list[dict[str, Any]], actions: list[str], only_tool_kind: str | None = None) -> float:
    grouped: dict[int, dict[str, str]] = defaultdict(dict)
    tool_kind_by_prefix: dict[int, str] = {}
    for row, action in zip(rows, actions):
        original = int(row["original_index"])
        grouped[original][row["counterfactual_observation"]] = branch_for_action(action)
        tool_kind_by_prefix[original] = row["last_tool_kind"]
    total = 0
    correct = 0
    for original, obs_to_branch in grouped.items():
        if only_tool_kind is not None and tool_kind_by_prefix.get(original) != only_tool_kind:
            continue
        if "success" not in obs_to_branch or "failure" not in obs_to_branch:
            continue
        total += 1
        success_ok = obs_to_branch["success"] == "finish" if tool_kind_by_prefix.get(original) == "write" else obs_to_branch["success"] != "recover"
        failure_ok = obs_to_branch["failure"] in {"recover", "verify", "acquire"}
        correct += int(success_ok and failure_ok)
    return correct / total if total else float("nan")


def observed_branch_accuracy(rows: list[dict[str, Any]], actions: list[str], only_tool_kind: str | None = None) -> float:
    total = 0
    correct = 0
    for row, action in zip(rows, actions):
        if only_tool_kind is not None and row["last_tool_kind"] != only_tool_kind:
            continue
        obs_type = row["counterfactual_observation"]
        if obs_type not in OBSERVATIONS:
            continue
        total += 1
        correct += int(branch_for_action(action) in expected_branches(obs_type, row["last_tool_kind"]))
    return correct / total if total else float("nan")


def distribution_by_observation(rows: list[dict[str, Any]], actions: list[str]) -> dict[str, dict[str, int]]:
    dist: dict[str, Counter[str]] = defaultdict(Counter)
    for row, action in zip(rows, actions):
        dist[row["counterfactual_observation"]][action] += 1
    return {key: dict(counter.most_common()) for key, counter in sorted(dist.items())}


def summarize_method(name: str, rows: list[dict[str, Any]], actions: list[str]) -> dict[str, Any]:
    all_sens = action_sensitivity(rows, actions)
    write_sens = action_sensitivity(rows, actions, "write")
    return {
        "method": name,
        "branch_accuracy_all": branch_accuracy(rows, actions),
        "branch_accuracy_write_tools": branch_accuracy(rows, actions, "write"),
        "observed_branch_accuracy_all": observed_branch_accuracy(rows, actions),
        "observed_branch_accuracy_write_tools": observed_branch_accuracy(rows, actions, "write"),
        "success_failure_pair_accuracy_all": pair_branch_rate(rows, actions),
        "success_failure_pair_accuracy_write_tools": pair_branch_rate(rows, actions, "write"),
        "unique_action_rate_all": all_sens["unique_action_rate"],
        "avg_unique_actions_all": all_sens["avg_unique_actions"],
        "unique_action_rate_write_tools": write_sens["unique_action_rate"],
        "avg_unique_actions_write_tools": write_sens["avg_unique_actions"],
        "action_distribution_by_observation": distribution_by_observation(rows, actions),
    }


def write_report(out_dir: Path, metadata: dict[str, Any], results: list[dict[str, Any]]) -> None:
    title = (
        "Existing-Hidden Observation Branch Audit"
        if metadata.get("not_counterfactual")
        else "Counterfactual Observation Branch Benchmark"
    )
    description = (
        "Held-out tau-bench states with real observations are evaluated from existing target hidden states."
        if metadata.get("not_counterfactual")
        else "Fixed held-out tau-bench prefixes are paired with synthetic success/failure/empty/conflict tool observations."
    )
    lines = [
        f"# {title}",
        "",
        description,
        "The main metric is whether the frozen transferred policy enters the expected response branch.",
        "",
        "## Setup",
        "",
        f"- Prefixes: {metadata['num_candidate_prefixes']}",
        f"- Counterfactual rows: {metadata['num_counterfactual_rows']}",
        f"- Tool kinds: `{metadata['tool_kind_counts']}`",
        f"- Observation counts: `{metadata.get('observation_type_counts', {})}`",
        f"- Existing-hidden fallback audit: `{metadata.get('not_counterfactual', False)}`",
        f"- Split: `{metadata.get('split_metadata', {})}`",
        "",
        "## Results",
        "",
        "| Method | Branch Acc All | Branch Acc Write | Observed Branch All | Observed Branch Write | Success/Failure Pair All | Pair Write | Unique Action Prefixes All | Unique Action Prefixes Write | Avg Unique Actions All | Avg Unique Actions Write |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| {method} | {branch_all} | {branch_write} | {observed_all} | {observed_write} | {pair_all} | {pair_write} | {uniq_all} | {uniq_write} | {avg_all:.2f} | {avg_write:.2f} |".format(
                method=result["method"],
                branch_all=pct(result["branch_accuracy_all"]),
                branch_write=pct(result["branch_accuracy_write_tools"]),
                observed_all=pct(result["observed_branch_accuracy_all"]),
                observed_write=pct(result["observed_branch_accuracy_write_tools"]),
                pair_all=pct(result["success_failure_pair_accuracy_all"]),
                pair_write=pct(result["success_failure_pair_accuracy_write_tools"]),
                uniq_all=pct(result["unique_action_rate_all"]),
                uniq_write=pct(result["unique_action_rate_write_tools"]),
                avg_all=result["avg_unique_actions_all"],
                avg_write=result["avg_unique_actions_write_tools"],
            )
        )
    lines.extend(["", "## Action Distributions", ""])
    for result in results:
        lines.append(f"### {result['method']}")
        lines.append("")
        for obs_type, dist in result["action_distribution_by_observation"].items():
            top = ", ".join(f"{name}:{count}" for name, count in list(dist.items())[:8])
            lines.append(f"- `{obs_type}`: {top}")
        lines.append("")
    (out_dir / "counterfactual_observation_branch_results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.decision_jsonl)
    _, val_idx, split_metadata = eval_protocol.load_split(args.split_json, len(rows))
    if args.existing_hidden_observation_audit:
        cf_rows, source_indices, metadata = build_existing_hidden_audit_rows(rows, val_idx, args.max_prefixes)
    else:
        cf_rows, metadata = build_counterfactual_rows(rows, val_idx, args.max_prefixes)
        source_indices = []
    if not cf_rows:
        raise ValueError("No held-out prefixes with a tool observation were found")
    metadata["split_metadata"] = split_metadata

    action_names = action_names_from_rows(rows)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    source_parts = load_source_parts(args.source_policy, device)
    alignment = torch.load(args.alignment, map_location="cpu")
    projection = RidgeProjection(alignment["coef"]).to(device).eval()
    policy_head = source_parts.head.to(device).eval()

    # Touch target tensor only to verify dimensional compatibility with the saved alignment.
    target_obj = load_hidden_tensor(args.target_tensor)
    if projection.coef.shape[0] != target_obj["h"].shape[1] + 1:
        raise ValueError(
            f"Alignment expects target hidden dim {projection.coef.shape[0] - 1}, "
            f"but target tensor has {target_obj['h'].shape[1]}"
        )

    if args.existing_hidden_observation_audit:
        hidden = target_obj["h"][torch.tensor(source_indices, dtype=torch.long)]
    else:
        hidden = extract_hidden_states(
            cf_rows,
            args.model_id,
            args.layer,
            args.batch_size,
            args.max_input_tokens,
            args.torch_dtype,
        )
    with torch.no_grad():
        logits = policy_head(projection(hidden.to(device))).cpu()

    static_pred = logits.argmax(dim=-1)
    static_actions = predictions_to_actions(static_pred, action_names)
    branch_constrained_actions = predictions_to_actions(
        constrained_branch_predictions(logits, cf_rows, action_names),
        action_names,
    )

    results = [
        summarize_method("static_lowrank_policy", cf_rows, static_actions),
        summarize_method("oracle_branch_constrained_policy", cf_rows, branch_constrained_actions),
    ]
    payload = {
        "metadata": metadata,
        "results": results,
        "rows": [
            {
                "sample_id": row["sample_id"],
                "original_index": row["original_index"],
                "domain": row.get("domain"),
                "task_id": row.get("task_id"),
                "turn_index": row.get("turn_index"),
                "last_tool_kind": row["last_tool_kind"],
                "counterfactual_observation": row["counterfactual_observation"],
                "static_action": static_actions[idx],
                "static_branch": branch_for_action(static_actions[idx]),
                "oracle_branch_action": branch_constrained_actions[idx],
                "oracle_branch": branch_for_action(branch_constrained_actions[idx]),
                "expected_branches": sorted(expected_branches(row["counterfactual_observation"], row["last_tool_kind"])),
            }
            for idx, row in enumerate(cf_rows)
        ],
    }
    (out_dir / "counterfactual_observation_branch_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(out_dir, metadata, results)


if __name__ == "__main__":
    main()
