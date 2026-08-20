from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from baselines.tool_policy_utils import (
    action_names_from_rows,
    add_reference_agreement,
    load_rows,
    parse_action,
    prediction_records,
    predictions_by_sample_id,
    save_json,
    torch_dtype_from_name,
)


DEFAULT_POLICY = {
    "name": "evotool_zero_adapt_prompt_policy",
    "planner": (
        "You are the frozen Planner from a source-trained EvoTool policy. "
        "Infer the next agent intent from the dialogue state."
    ),
    "selector": (
        "Select exactly one universal policy action from the allowed action list. "
        "Prefer ASK_USER when required information is missing, VERIFY before irreversible updates, "
        "ANSWER when the task is complete, and ACT_* when a tool should be called."
    ),
    "caller": "If an ACT_* action is selected, choose the abstract action category only, not tool arguments.",
    "synthesizer": "Return only the selected universal action name. No reasoning, no explanation.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "EvoTool-style zero-adaptation prompt-policy transfer baseline. "
            "The evolved source policy is represented as frozen external prompts; "
            "the target backbone is not trained."
        )
    )
    parser.add_argument("--decision-jsonl", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--policy-artifact", help="JSON file containing frozen planner/selector/caller/synthesizer prompts")
    parser.add_argument("--reference-predictions", help="Optional source-policy prediction file for policy agreement")
    parser.add_argument("--max-samples", type=int, default=60)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=24)
    return parser.parse_args()


def load_policy_artifact(path: str | None) -> dict[str, Any]:
    if not path:
        return dict(DEFAULT_POLICY)
    with Path(path).open("r", encoding="utf-8") as handle:
        policy = json.load(handle)
    merged = dict(DEFAULT_POLICY)
    merged.update(policy)
    return merged


def prompt_for(row: dict[str, Any], action_names: list[str], policy: dict[str, Any]) -> str:
    return (
        f"{policy['planner']}\n"
        f"{policy['selector']}\n"
        f"{policy['caller']}\n"
        f"{policy['synthesizer']}\n\n"
        "Allowed universal actions:\n"
        f"{', '.join(action_names)}\n\n"
        "Decision state:\n"
        f"{row['prompt']}\n\n"
        "Selected action:"
    )


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_rows(args.decision_jsonl, args.max_samples)
    action_names = action_names_from_rows(rows)
    policy = load_policy_artifact(args.policy_artifact)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype_from_name(args.torch_dtype),
        device_map=args.device if args.device != "cpu" else None,
    )
    if args.device == "cpu":
        model.to("cpu")
    model.eval()

    predictions: list[str] = []
    raw_outputs: list[str] = []
    for row in rows:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are executing a frozen source-trained EvoTool external policy. "
                    "Return only one action label."
                ),
            },
            {"role": "user", "content": prompt_for(row, action_names, policy)},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        raw_outputs.append(text)
        predictions.append(parse_action(text, action_names))

    artifact = prediction_records(
        rows,
        predictions,
        method="evotool_zero_adapt_prompt_policy",
        metadata={
            "model_id": args.model_id,
            "policy_artifact": args.policy_artifact,
            "target_trainable_parameters": 0,
            "target_training_labels": 0,
            "target_training_states": 0,
            "onboarding": "zero_adaptation_external_prompt_policy_transfer",
        },
    )
    for record, raw in zip(artifact["records"], raw_outputs, strict=True):
        record["raw_output"] = raw
    if args.reference_predictions:
        add_reference_agreement(artifact, rows, predictions_by_sample_id(args.reference_predictions), "source")
    save_json(artifact, args.out)
    print(json.dumps(artifact["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
