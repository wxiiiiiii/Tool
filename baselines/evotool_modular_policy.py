from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch

from baselines.tool_policy_utils import (
    action_names_from_rows,
    add_reference_agreement,
    load_rows,
    prediction_records,
    predictions_by_sample_id,
    save_json,
    torch_dtype_from_name,
)
from universal_agent_policy.eval.tau_replay_agent import ensure_tau_import, load_domain_runtime
from universal_agent_policy.runtime.tau_grounding import tau_tool_actions


NO_TOOL_MARKERS = {"", "NONE", "NO_TOOL", "STOP", "DONE", "FINISH", "SKIP"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paper-faithful EvoTool modular policy baseline for tau decision states. "
            "It loads official theta_star.json, runs Planner then Selector, maps the selected "
            "concrete tau tool to the repository's universal action vocabulary, and emits "
            "precomputed action predictions for shared replay/evaluation."
        )
    )
    parser.add_argument("--decision-jsonl", required=True)
    parser.add_argument("--theta-star", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--tau-bench-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference-predictions")
    parser.add_argument("--max-samples", type=int, default=60)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-plan-steps", type=int, default=3)
    return parser.parse_args()


def load_theta(path: str | Path) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    modules = obj.get("modules", obj)
    required = {"planner", "selector", "caller", "synthesizer"}
    missing = required - set(modules)
    if missing:
        raise ValueError(f"theta_star missing modules: {sorted(missing)}")
    return {key: str(modules[key]) for key in sorted(required)}


def json_from_text(text: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?think>", " ", text, flags=re.IGNORECASE)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def generate_json(model: Any, tokenizer: Any, messages: list[dict[str, str]], max_new_tokens: int) -> tuple[dict[str, Any], str]:
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    return json_from_text(text), text


def compact_state(prompt: str) -> str:
    markers = [
        "Conversation state before the next assistant decision:",
        "Conversation:",
    ]
    for marker in markers:
        if marker in prompt:
            text = prompt.split(marker, 1)[1]
            end = "Next policy action:"
            if end in text:
                text = text.split(end, 1)[0]
            return text.strip()
    return prompt[-5000:]


def tools_brief(tool_specs: list[Any]) -> str:
    lines = []
    for spec in tool_specs:
        params = spec.parameters.get("properties", {}) if isinstance(spec.parameters, dict) else {}
        lines.append(f"- {spec.name}: {spec.description} params={list(params)}")
    return "\n".join(lines)


def run_planner(
    model: Any,
    tokenizer: Any,
    spec: str,
    query: str,
    max_new_tokens: int,
) -> tuple[list[str], str]:
    fmt = (
        '\n\nReturn JSON only: {"plan": ["subgoal 1", "subgoal 2", ...]}. '
        "Each subgoal is one short ABSTRACT step describing WHAT to accomplish. "
        "Do NOT name specific tools or APIs."
    )
    messages = [
        {"role": "system", "content": spec + fmt},
        {"role": "user", "content": f"USER_TASK:\n{query}"},
    ]
    obj, raw = generate_json(model, tokenizer, messages, max_new_tokens)
    plan = obj.get("plan") or []
    plan = [str(item).strip() for item in plan if str(item).strip()]
    return plan, raw


def run_selector(
    model: Any,
    tokenizer: Any,
    spec: str,
    query: str,
    subgoal: str,
    state: dict[str, Any],
    tool_specs: list[Any],
    max_new_tokens: int,
) -> tuple[str, str]:
    fmt = (
        '\n\nChoose the single best tool to accomplish the CURRENT_SUBGOAL, using '
        "the steps done so far for context. "
        'Return JSON only: {"tool": "<EXACT name from AVAILABLE_TOOLS, or NONE if this '
        'subgoal needs no tool>"}.'
    )
    state_brief = "\n".join(f"{k}: {v}" for k, v in state.items()) or "(nothing done yet)"
    messages = [
        {"role": "system", "content": spec + fmt},
        {
            "role": "user",
            "content": (
                f"USER_TASK:\n{query}\n\nCURRENT_SUBGOAL:\n{subgoal}\n\n"
                f"STEPS DONE SO FAR:\n{state_brief}\n\nAVAILABLE_TOOLS:\n{tools_brief(tool_specs)}"
            ),
        },
    ]
    obj, raw = generate_json(model, tokenizer, messages, max_new_tokens)
    return str(obj.get("tool") or obj.get("next_tool") or "NONE").strip(), raw


def normalize_tool_name(name: str, tool_specs: list[Any]) -> str:
    exact = {spec.name: spec.name for spec in tool_specs}
    if name in exact:
        return name
    lower = name.lower()
    for spec in tool_specs:
        if spec.name.lower() == lower:
            return spec.name
    for spec in tool_specs:
        if spec.name.lower() in lower:
            return spec.name
    return name


def non_tool_action_from_state(query: str, action_names: list[str]) -> str:
    lower = query.lower()
    if "confirm" in lower or "confirmation" in lower or "explicit user confirmation" in lower:
        return "VERIFY" if "VERIFY" in action_names else "ANSWER"
    if "?" in query[-500:] or "please provide" in lower or "need" in lower and "information" in lower:
        return "ASK_USER" if "ASK_USER" in action_names else "ANSWER"
    return "ANSWER" if "ANSWER" in action_names else action_names[0]


def tool_to_action(tool_name: str, action_names: list[str], query: str) -> str:
    if tool_name.upper() in NO_TOOL_MARKERS:
        return non_tool_action_from_state(query, action_names)
    candidates = tau_tool_actions(tool_name)
    for action in sorted(candidates, key=len, reverse=True):
        if action in action_names:
            return action
    return "THINK" if "THINK" in action_names else non_tool_action_from_state(query, action_names)


def main() -> None:
    args = parse_args()
    ensure_tau_import(args.tau_bench_path)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_rows(args.decision_jsonl, args.max_samples)
    action_names = action_names_from_rows(rows)
    theta = load_theta(args.theta_star)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype_from_name(args.torch_dtype),
        device_map=args.device if args.device != "cpu" else None,
    )
    if args.device == "cpu":
        model.to("cpu")
    model.eval()

    domains = {domain: load_domain_runtime(domain) for domain in sorted({str(row.get("domain")) for row in rows})}
    predictions: list[str] = []
    traces: list[dict[str, Any]] = []
    for row in rows:
        query = compact_state(str(row["prompt"]))
        runtime = domains[str(row.get("domain"))]
        plan, planner_raw = run_planner(model, tokenizer, theta["planner"], query, args.max_new_tokens)
        if not plan:
            predictions.append(non_tool_action_from_state(query, action_names))
            traces.append({"plan": [], "selected_tool": "NONE", "planner_raw": planner_raw, "selector_raw": ""})
            continue
        state: dict[str, Any] = {}
        selected_tool = "NONE"
        selector_raw = ""
        for subgoal in plan[: args.max_plan_steps]:
            selected_tool, selector_raw = run_selector(
                model,
                tokenizer,
                theta["selector"],
                query,
                subgoal,
                state,
                runtime.tool_specs,
                args.max_new_tokens,
            )
            selected_tool = normalize_tool_name(selected_tool, runtime.tool_specs)
            if selected_tool.upper() not in NO_TOOL_MARKERS:
                break
            state[f"subgoal_{len(state) + 1}"] = f"{subgoal} -> no tool"
        predictions.append(tool_to_action(selected_tool, action_names, query))
        traces.append(
            {
                "plan": plan,
                "selected_tool": selected_tool,
                "planner_raw": planner_raw,
                "selector_raw": selector_raw,
            }
        )

    artifact = prediction_records(
        rows,
        predictions,
        method="evotool_official_theta_modular_policy",
        metadata={
            "model_id": args.model_id,
            "theta_star": args.theta_star,
            "target_trainable_parameters": 0,
            "target_training_labels": 0,
            "target_training_states": 0,
            "onboarding": "official_theta_star_zero_target_adaptation",
            "bridge": "planner_selector_concrete_tool_to_universal_action",
        },
    )
    for record, trace in zip(artifact["records"], traces, strict=True):
        record["evotool_trace"] = trace
    if args.reference_predictions:
        add_reference_agreement(artifact, rows, predictions_by_sample_id(args.reference_predictions), "source")
    save_json(artifact, args.out)
    print(json.dumps(artifact["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
