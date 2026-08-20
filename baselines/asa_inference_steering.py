from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from baselines.tool_policy_utils import (
    action_names_from_rows,
    add_reference_agreement,
    is_tool_action,
    load_rows,
    parse_action,
    prediction_records,
    predictions_by_sample_id,
    save_json,
    torch_dtype_from_name,
)


@dataclass
class ASAInferenceConfig:
    method: str
    model_id: str
    layer: int
    intervention_mode: str
    decode_mode: str
    calibration_samples: int
    max_samples: int
    alpha: float
    beta_domain: float
    abstain_margin: float
    target_trainable_parameters: int
    stored_parameters: int
    uses_target_action_labels: bool
    uses_source_policy: bool
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ASA-style inference-time activation steering baseline. This script keeps the "
            "target HF model frozen, builds tool-boundary/domain steering directions from a "
            "small target calibration split, hooks a middle transformer layer during generate, "
            "and lets the target model emit an action from the universal vocabulary."
        )
    )
    parser.add_argument("--decision-jsonl", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference-predictions", help="Optional source-policy prediction file for agreement metrics")
    parser.add_argument("--max-samples", type=int, default=60)
    parser.add_argument("--calibration-samples", type=int, default=240)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--layer", type=int, default=-8, help="Transformer block index; negative indexes from the end")
    parser.add_argument("--intervention-mode", choices=["prefill", "cascade"], default="cascade")
    parser.add_argument(
        "--decode-mode",
        choices=["probe", "constrained_score", "generate"],
        default="probe",
        help=(
            "probe uses the ASA lightweight action probe over steered states; constrained_score "
            "ranks valid action labels with the LM; generate lets the model freely emit text"
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--beta-domain", type=float, default=0.3)
    parser.add_argument("--abstain-margin", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser.parse_args()


def prompt_for(row: dict[str, Any], action_names: list[str]) -> str:
    return (
        f"{row['prompt']}\n\n"
        "You are a strict tool-policy controller. Do not reason. Do not output <think> tags.\n"
        "Return exactly one action name from this list and no other text:\n"
        f"{', '.join(action_names)}"
    )


def messages_for(row: dict[str, Any], action_names: list[str]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Return only the requested action label. No reasoning, no explanation."},
        {"role": "user", "content": prompt_for(row, action_names)},
    ]


def find_transformer_layers(model: Any) -> Any:
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("backbone", "layers"),
        ("language_model", "model", "layers"),
    ]
    for path in candidates:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise ValueError("Could not locate transformer layers for activation hook")


def resolve_layer_index(layer: int, num_layers: int) -> int:
    idx = layer if layer >= 0 else num_layers + layer
    if not 0 <= idx < num_layers:
        raise ValueError(f"Layer index {layer} resolved to {idx}, outside 0..{num_layers - 1}")
    return idx


def render_inputs(tokenizer: Any, row: dict[str, Any], action_names: list[str], device: torch.device, max_input_tokens: int) -> Any:
    messages = messages_for(row, action_names)
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    if encoded["input_ids"].shape[-1] > max_input_tokens:
        encoded["input_ids"] = encoded["input_ids"][:, -max_input_tokens:]
        if "attention_mask" in encoded:
            encoded["attention_mask"] = encoded["attention_mask"][:, -max_input_tokens:]
    return encoded.to(device)


def extract_layer_state(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    action_names: list[str],
    hidden_state_index: int,
    max_input_tokens: int,
) -> torch.Tensor:
    inputs = render_inputs(tokenizer, row, action_names, model.device, max_input_tokens)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    return outputs.hidden_states[hidden_state_index][0, -1].detach().float().cpu()


def mean_or_zero(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.numel() == 0:
        return torch.zeros(dim)
    return x.float().mean(dim=0)


def normalize_or_zero(x: torch.Tensor) -> torch.Tensor:
    norm = x.norm()
    if float(norm) == 0.0 or math.isnan(float(norm)):
        return torch.zeros_like(x)
    return x / norm


def choose_threshold(scores: torch.Tensor, labels: torch.Tensor) -> float:
    tool_scores = scores[labels]
    non_tool_scores = scores[~labels]
    if tool_scores.numel() == 0 or non_tool_scores.numel() == 0:
        return float(scores.median())
    return float((tool_scores.mean() + non_tool_scores.mean()) / 2.0)


def calibration_indices(num_rows: int, max_samples: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    indices = list(range(num_rows))
    rng.shuffle(indices)
    return sorted(indices[: min(max_samples, num_rows)])


def fit_asa_controller(
    states: torch.Tensor,
    rows: list[dict[str, Any]],
    indices: list[int],
    action_names: list[str],
) -> dict[str, Any]:
    dim = states.shape[1]
    h_cal = states[indices].float()
    mean = h_cal.mean(dim=0)
    std = h_cal.std(dim=0).clamp_min(1e-6)
    z_cal = (h_cal - mean) / std

    labels = torch.tensor([is_tool_action(str(rows[idx]["correct_action"])) for idx in indices], dtype=torch.bool)
    action_to_id = {name: idx for idx, name in enumerate(action_names)}
    action_labels = torch.tensor(
        [action_to_id.get(str(rows[idx]["correct_action"]), -1) for idx in indices],
        dtype=torch.long,
    )
    boundary = normalize_or_zero(mean_or_zero(z_cal[labels], dim) - mean_or_zero(z_cal[~labels], dim))
    scores = z_cal @ boundary
    threshold = choose_threshold(scores, labels)

    global_mean = z_cal.mean(dim=0)
    action_centroids = []
    for action_id in range(len(action_names)):
        mask = action_labels == action_id
        action_centroids.append(mean_or_zero(z_cal[mask], dim) if bool(mask.any()) else global_mean)

    domain_residuals: dict[str, torch.Tensor] = {}
    for domain in sorted({str(row.get("domain", "unknown")) for row in rows}):
        local_positions = [
            pos for pos, row_idx in enumerate(indices) if str(rows[row_idx].get("domain", "unknown")) == domain
        ]
        if not local_positions:
            domain_residuals[domain] = torch.zeros(dim)
            continue
        local_labels = labels[torch.tensor(local_positions)]
        local_states = z_cal[torch.tensor(local_positions)]
        local_dir = mean_or_zero(local_states[local_labels], dim) - mean_or_zero(local_states[~local_labels], dim)
        residual = local_dir - torch.dot(local_dir, boundary) * boundary
        domain_residuals[domain] = normalize_or_zero(residual)

    return {
        "mean": mean,
        "std": std,
        "boundary": boundary,
        "threshold": threshold,
        "domain_residuals": domain_residuals,
        "action_centroids": F.normalize(torch.stack(action_centroids), dim=-1),
    }


def gate_for_state(state: torch.Tensor, controller: dict[str, Any], abstain_margin: float) -> tuple[int, float]:
    z = (state.float() - controller["mean"]) / controller["std"].clamp_min(1e-6)
    score = float(torch.dot(z, controller["boundary"]))
    threshold = float(controller["threshold"])
    if score >= threshold + abstain_margin:
        return 1, score
    if score <= threshold - abstain_margin:
        return -1, score
    return 0, score


def delta_for_row(
    row: dict[str, Any],
    state: torch.Tensor,
    controller: dict[str, Any],
    alpha: float,
    beta_domain: float,
    abstain_margin: float,
) -> tuple[torch.Tensor, int, float]:
    gate, score = gate_for_state(state, controller, abstain_margin)
    domain = str(row.get("domain", "unknown"))
    direction = controller["boundary"] + beta_domain * controller["domain_residuals"].get(
        domain, torch.zeros_like(controller["boundary"])
    )
    direction = normalize_or_zero(direction)
    delta = float(alpha) * float(gate) * direction * controller["std"]
    return delta, gate, score


def generate_with_intervention(
    model: Any,
    tokenizer: Any,
    layer_module: Any,
    row: dict[str, Any],
    action_names: list[str],
    delta: torch.Tensor,
    intervention_mode: str,
    max_input_tokens: int,
    max_new_tokens: int,
) -> str:
    inputs = render_inputs(tokenizer, row, action_names, model.device, max_input_tokens)
    current_delta = delta.to(model.device)
    calls = {"n": 0}

    def hook(_: Any, __: Any, output: Any) -> Any:
        apply_now = intervention_mode == "cascade" or calls["n"] == 0
        calls["n"] += 1
        if not apply_now:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        delta_t = current_delta.to(device=hidden.device, dtype=hidden.dtype)
        steered = hidden.clone()
        steered[:, -1, :] = steered[:, -1, :] + delta_t
        if isinstance(output, tuple):
            return (steered, *output[1:])
        return steered

    handle = layer_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        handle.remove()
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)


def score_action_with_intervention(
    model: Any,
    tokenizer: Any,
    layer_module: Any,
    row: dict[str, Any],
    action_names: list[str],
    action_name: str,
    delta: torch.Tensor,
    max_input_tokens: int,
) -> float:
    prompt_inputs = render_inputs(tokenizer, row, action_names, model.device, max_input_tokens)
    candidate = tokenizer(action_name, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    if candidate.numel() == 0:
        return float("-inf")

    input_ids = torch.cat([prompt_inputs["input_ids"], candidate], dim=-1)
    attention_mask = torch.ones_like(input_ids)
    if "attention_mask" in prompt_inputs:
        attention_mask[:, : prompt_inputs["attention_mask"].shape[-1]] = prompt_inputs["attention_mask"]
    prompt_len = int(prompt_inputs["input_ids"].shape[-1])
    label_len = int(candidate.shape[-1])
    current_delta = delta.to(model.device)

    def hook(_: Any, __: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        delta_t = current_delta.to(device=hidden.device, dtype=hidden.dtype)
        steered = hidden.clone()
        start = max(0, prompt_len - 1)
        end = max(start + 1, prompt_len + label_len - 1)
        steered[:, start:end, :] = steered[:, start:end, :] + delta_t
        if isinstance(output, tuple):
            return (steered, *output[1:])
        return steered

    handle = layer_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()

    logits = outputs.logits[:, prompt_len - 1 : prompt_len + label_len - 1, :]
    targets = candidate
    token_logprobs = F.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(token_logprobs.mean())


def constrained_decode_with_intervention(
    model: Any,
    tokenizer: Any,
    layer_module: Any,
    row: dict[str, Any],
    action_names: list[str],
    delta: torch.Tensor,
    max_input_tokens: int,
) -> tuple[str, str]:
    scored = [
        (action_name, score_action_with_intervention(model, tokenizer, layer_module, row, action_names, action_name, delta, max_input_tokens))
        for action_name in action_names
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    raw = " | ".join(f"{name}:{score:.4f}" for name, score in scored[: min(5, len(scored))])
    return scored[0][0], raw


def probe_decode(
    state: torch.Tensor,
    delta: torch.Tensor,
    action_names: list[str],
    controller: dict[str, Any],
) -> tuple[str, str]:
    steered = (state.float() + delta.float() - controller["mean"]) / controller["std"].clamp_min(1e-6)
    steered = F.normalize(steered, dim=0)
    action_scores = controller["action_centroids"] @ steered
    topk = torch.topk(action_scores, k=min(5, len(action_names)))
    raw = " | ".join(
        f"{action_names[int(idx)]}:{float(score):.4f}"
        for score, idx in zip(topk.values, topk.indices, strict=True)
    )
    return action_names[int(topk.indices[0])], raw


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_rows(args.decision_jsonl, max_samples=-1)
    if not rows:
        raise ValueError("No decision rows loaded")
    eval_rows = rows[: args.max_samples] if args.max_samples > 0 else rows
    action_names = action_names_from_rows(rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype_from_name(args.torch_dtype),
        device_map=args.device if args.device != "cpu" else None,
    )
    if args.device == "cpu":
        model.to("cpu")
    model.eval()

    layers = find_transformer_layers(model)
    layer_idx = resolve_layer_index(args.layer, len(layers))
    hidden_state_index = layer_idx + 1
    layer_module = layers[layer_idx]

    cal_idx = calibration_indices(len(rows), args.calibration_samples, args.seed)
    cal_states = []
    for idx in cal_idx:
        cal_states.append(extract_layer_state(model, tokenizer, rows[idx], action_names, hidden_state_index, args.max_input_tokens))
    states = torch.zeros(len(rows), cal_states[0].shape[0], dtype=torch.float32)
    states[cal_idx] = torch.stack(cal_states)
    controller = fit_asa_controller(states, rows, cal_idx, action_names)

    predictions: list[str] = []
    raw_outputs: list[str] = []
    gates: list[int] = []
    scores: list[float] = []
    for row in eval_rows:
        state = extract_layer_state(model, tokenizer, row, action_names, hidden_state_index, args.max_input_tokens)
        delta, gate, score = delta_for_row(
            row,
            state,
            controller,
            args.alpha,
            args.beta_domain,
            args.abstain_margin,
        )
        if args.decode_mode == "probe":
            predicted, text = probe_decode(state, delta, action_names, controller)
        elif args.decode_mode == "constrained_score":
            predicted, text = constrained_decode_with_intervention(
                model,
                tokenizer,
                layer_module,
                row,
                action_names,
                delta,
                args.max_input_tokens,
            )
        else:
            text = generate_with_intervention(
                model,
                tokenizer,
                layer_module,
                row,
                action_names,
                delta,
                args.intervention_mode,
                args.max_input_tokens,
                args.max_new_tokens,
            )
            predicted = parse_action(text, action_names)
        predictions.append(predicted)
        raw_outputs.append(text)
        gates.append(gate)
        scores.append(score)

    stored_params = (
        int(controller["mean"].numel())
        + int(controller["std"].numel())
        + int(controller["boundary"].numel())
        + int(controller["action_centroids"].numel())
        + sum(int(value.numel()) for value in controller["domain_residuals"].values())
        + 1
    )
    metadata = ASAInferenceConfig(
        method="asa_inference_time_activation_steering",
        model_id=args.model_id,
        layer=layer_idx,
        intervention_mode=args.intervention_mode,
        decode_mode=args.decode_mode,
        calibration_samples=len(cal_idx),
        max_samples=len(eval_rows),
        alpha=args.alpha,
        beta_domain=args.beta_domain,
        abstain_margin=args.abstain_margin,
        target_trainable_parameters=0,
        stored_parameters=stored_params,
        uses_target_action_labels=True,
        uses_source_policy=False,
        note=(
            "Paper-faithful ASA-style baseline: target-side calibration labels build a "
            "tool boundary, domain residuals, and signed gate; the frozen target model is "
            "steered at inference through a transformer-layer hook before generating the action."
        ),
    )
    artifact = prediction_records(eval_rows, predictions, method=metadata.method, metadata=asdict(metadata))
    calibration_set = set(cal_idx)
    for idx, record in enumerate(artifact["records"]):
        source_row_idx = idx
        record["calibration_state"] = source_row_idx in calibration_set
        record["gate"] = gates[idx]
        record["probe_score"] = scores[idx]
        record["raw_output"] = raw_outputs[idx]
    if args.reference_predictions:
        add_reference_agreement(artifact, eval_rows, predictions_by_sample_id(args.reference_predictions), "source")
    artifact["controller"] = {
        "action_names": action_names,
        "calibration_indices": cal_idx,
        "boundary_threshold": controller["threshold"],
        "domain_names": sorted(controller["domain_residuals"].keys()),
        "note": "Controller tensor values are not serialized; rerun the script to rebuild them.",
    }
    save_json(artifact, args.out)
    print(json.dumps(artifact["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
