from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from baselines import eval_protocol
from baselines.hidden_policy_baselines import RidgeProjection, load_source_parts, source_latents
from baselines.tool_policy_utils import prediction_records, save_json
from universal_agent_policy.data import load_hidden_tensor


COLLECT_PREFIXES = ("ACT_RETRIEVE", "ACT_SEARCH", "ACT_COMPUTE")
EXECUTE_PREFIXES = ("ACT_UPDATE", "ACT_SEND")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate whether cross-backbone policy transfer is more stable at a "
            "coarse multi-step Skill level than at the concrete action level."
        )
    )
    parser.add_argument("--source-tensor", required=True)
    parser.add_argument("--target-tensor", required=True)
    parser.add_argument("--source-policy", required=True)
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--decision-jsonl", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--skill-prior-beta", type=float, default=1.0)
    parser.add_argument("--skill-confidence-threshold", type=float, default=0.55)
    parser.add_argument("--temporal-skill-weight", type=float, default=0.75)
    parser.add_argument("--temporal-skill-smoothing", type=float, default=0.5)
    parser.add_argument("--skill-binder-ridge", type=float, default=1e-2)
    parser.add_argument("--skill-binder-beta", type=float, default=1.0)
    parser.add_argument("--response-operator-clusters", type=int, default=8)
    parser.add_argument("--response-residual-alpha", type=float, default=0.6)
    parser.add_argument("--write-pair-calibration-margin", type=float, default=1.0)
    parser.add_argument("--write-pair-max-calibration", type=float, default=8.0)
    parser.add_argument("--outcome-execute-skill-weight", type=float, default=8.0)
    parser.add_argument("--outcome-critical-confidence-threshold", type=float, default=0.35)
    parser.add_argument("--selective-action-margin-threshold", type=float, default=0.08)
    parser.add_argument("--selective-skill-margin-threshold", type=float, default=0.10)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def action_to_skill(action: str) -> str:
    if action.startswith(COLLECT_PREFIXES):
        return "COLLECT_INFO"
    if action == "VERIFY":
        return "CONFIRM"
    if action.startswith(EXECUTE_PREFIXES):
        return "EXECUTE"
    if action in {"ASK_USER", "THINK", "TRANSFER"}:
        return "RECOVER_OR_ELICIT"
    if action in {"ANSWER", "STOP"}:
        return "FINISH"
    return "OTHER"


def skill_groups(action_names: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, action in enumerate(action_names):
        groups[action_to_skill(action)].append(idx)
    return dict(groups)


def random_action_groups(action_names: list[str], reference_groups: dict[str, list[int]], seed: int = 13) -> dict[str, str]:
    sizes = [len(reference_groups[name]) for name in sorted(reference_groups)]
    indices = list(range(len(action_names)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    mapping: dict[str, str] = {}
    cursor = 0
    for group_idx, size in enumerate(sizes):
        group_name = f"RANDOM_{group_idx}"
        for action_idx in indices[cursor : cursor + size]:
            mapping[action_names[action_idx]] = group_name
        cursor += size
    return mapping


def labels_from_action_map(actions: list[str], action_to_label: dict[str, str]) -> list[str]:
    return [action_to_label[action] for action in actions]


def conversation_segment(prompt: str) -> str:
    marker = "Conversation state before the next assistant decision:"
    if marker not in prompt:
        return prompt
    tail = prompt.split(marker, 1)[1]
    return tail.split("Next policy action:", 1)[0]


def last_tool_name(prompt: str) -> str:
    convo = conversation_segment(prompt)
    for line in reversed([line.strip() for line in convo.splitlines() if line.strip()]):
        match = re.search(r"assistant tool_call:\s*([A-Za-z0-9_]+)\(", line)
        if match:
            return match.group(1)
    return "unknown_tool"


def tool_family(tool_name: str) -> str:
    lower = tool_name.lower()
    if lower.startswith(("book_", "cancel_", "exchange_", "modify_", "return_", "send_", "update_")):
        return "write"
    if lower.startswith(("find_", "get_", "list_")):
        return "retrieve"
    if lower.startswith("search_"):
        return "search"
    if lower == "calculate":
        return "compute"
    if lower == "think":
        return "think"
    if "transfer" in lower:
        return "transfer"
    return "other"


def prefix_features(row: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Features available before the current action; no GT action or future observation."""
    prompt = str(row.get("prompt") or "")
    convo = conversation_segment(prompt)
    lower = convo.lower()
    lines = [line.strip() for line in convo.splitlines() if line.strip()]
    last = lines[-1].lower() if lines else ""
    last_tool_line = ""
    for line in reversed(lines):
        if line.lower().startswith("tool "):
            last_tool_line = line.lower()
            break

    user_count = sum(line.lower().startswith("user:") for line in lines)
    assistant_count = sum(line.lower().startswith("assistant:") for line in lines)
    tool_call_count = sum("assistant tool_call:" in line.lower() for line in lines)
    tool_obs_count = sum(line.lower().startswith("tool ") for line in lines)
    total = max(1.0, float(len(lines)))
    tool_names = [
        "find_user",
        "get_user",
        "get_order",
        "get_product",
        "get_reservation",
        "search",
        "update",
        "cancel",
        "return",
        "book",
        "send",
        "flight",
    ]
    tool_hits = [1.0 if name in last_tool_line else 0.0 for name in tool_names]
    history_tool_hits = [min(1.0, lower.count(name) / 3.0) for name in tool_names]
    confirmation_terms = (
        "confirm",
        "confirmation",
        "proceed",
        "go ahead",
        "yes",
        "fine",
        "sounds good",
        "do it",
    )
    error_terms = ("error", "not found", "invalid", "cannot", "can't", "not possible", "permission")
    empty_terms = ("[]", "empty", "no result", "none found", "not found")
    frustration_terms = ("damn", "upset", "angry", "frustrated", "unacceptable")
    id_terms = ("user_id", "order_id", "reservation_id", "flight_number", "payment_method_id", "item_id")
    values = [
        1.0 if str(row.get("domain")) == "retail" else 0.0,
        1.0 if str(row.get("domain")) == "airline" else 0.0,
        min(user_count / 8.0, 1.0),
        min(assistant_count / 8.0, 1.0),
        min(tool_call_count / 8.0, 1.0),
        min(tool_obs_count / 8.0, 1.0),
        1.0 if last.startswith("user:") else 0.0,
        1.0 if last.startswith("assistant:") else 0.0,
        1.0 if last.startswith("tool ") else 0.0,
        1.0 if "assistant tool_call:" in last else 0.0,
        1.0 if any(term in last for term in error_terms) else 0.0,
        1.0 if any(term in last_tool_line for term in error_terms) else 0.0,
        1.0 if any(term in last_tool_line for term in empty_terms) else 0.0,
        1.0 if "{" in last_tool_line and "}" in last_tool_line else 0.0,
        1.0 if "authenticated" in lower or "identity has been authenticated" in lower else 0.0,
        1.0 if any(term in last for term in confirmation_terms) else 0.0,
        1.0 if any(term in lower for term in confirmation_terms) else 0.0,
        1.0 if "could you" in last or "please provide" in last else 0.0,
        1.0 if any(term in last for term in frustration_terms) else 0.0,
        1.0 if "transfer" in lower or "human agent" in lower else 0.0,
        1.0 if any(term in lower for term in id_terms) else 0.0,
        min(len(re.findall(r"#[A-Z]\\d+", convo)) / 5.0, 1.0),
        min(len(re.findall(r"\\b[a-z_]+_\\d+\\b", convo.lower())) / 5.0, 1.0),
        min(total / 20.0, 1.0),
    ]
    values.extend(tool_hits)
    values.extend(history_tool_hits)
    return torch.tensor(values, dtype=torch.float32, device=device)


def prefix_defined_skill(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt") or "")
    convo = conversation_segment(prompt)
    lower = convo.lower()
    lines = [line.strip() for line in convo.splitlines() if line.strip()]
    last = lines[-1].lower() if lines else ""
    last_tool = ""
    for line in reversed(lines):
        if line.lower().startswith("tool "):
            last_tool = line.lower()
            break

    has_auth = "authenticated" in lower or "identity has been authenticated" in lower
    has_user_lookup = "find_user_id" in lower or "get_user_details" in lower
    has_confirmation_request = "please confirm" in lower or "confirm if" in lower or "would you like to proceed" in lower
    last_user_confirms = (
        last.startswith("user:")
        and any(term in last for term in ("yes", "confirm", "proceed", "go ahead", "fine", "do it", "usual way"))
    )
    last_tool_failure = any(term in last_tool for term in ("error", "not found", "invalid", "cannot", "can't", "permission"))
    last_tool_commit = any(
        name in last_tool
        for name in (
            "update_",
            "cancel_",
            "return_",
            "book_",
            "send_",
            "modify_",
        )
    )
    last_tool_success = bool(last_tool) and not last_tool_failure
    user_needs_help = last.startswith("user:")

    if last_tool_failure:
        return "RECOVER_AFTER_FAILURE"
    if last_tool_commit and last_tool_success:
        return "FINISH_AFTER_COMMIT"
    if last_user_confirms and has_confirmation_request:
        return "EXECUTE_AFTER_CONFIRM"
    if has_confirmation_request and user_needs_help and not last_user_confirms:
        return "CONFIRM_OR_CLARIFY"
    if not has_auth or not has_user_lookup:
        return "AUTHENTICATE"
    if user_needs_help and any(term in last for term in ("can't", "cannot", "upset", "damn", "angry")):
        return "RECOVER_AFTER_FAILURE"
    if user_needs_help and any(term in last for term in ("yes", "fine", "do it", "proceed", "confirm")):
        return "EXECUTE_AFTER_CONFIRM"
    if last_tool_success:
        return "COLLECT_OR_VERIFY"
    return "COLLECT_OR_VERIFY"


def prefix_skill_allowed_actions(action_names: list[str]) -> dict[str, list[int]]:
    allowed: dict[str, set[int]] = {
        "AUTHENTICATE": set(),
        "COLLECT_OR_VERIFY": set(),
        "CONFIRM_OR_CLARIFY": set(),
        "EXECUTE_AFTER_CONFIRM": set(),
        "RECOVER_AFTER_FAILURE": set(),
        "FINISH_AFTER_COMMIT": set(),
    }
    for idx, action in enumerate(action_names):
        if action in {"ASK_USER", "THINK"} or action.startswith("ACT_RETRIEVE_USER"):
            allowed["AUTHENTICATE"].add(idx)
        if action.startswith(COLLECT_PREFIXES) or action in {"ASK_USER", "VERIFY", "THINK"}:
            allowed["COLLECT_OR_VERIFY"].add(idx)
        if action in {"VERIFY", "ASK_USER", "ANSWER", "THINK"}:
            allowed["CONFIRM_OR_CLARIFY"].add(idx)
        if action.startswith(EXECUTE_PREFIXES) or action in {"VERIFY", "ASK_USER", "THINK"}:
            allowed["EXECUTE_AFTER_CONFIRM"].add(idx)
        if action in {"ASK_USER", "THINK", "TRANSFER", "VERIFY"} or action.startswith(COLLECT_PREFIXES):
            allowed["RECOVER_AFTER_FAILURE"].add(idx)
        if action in {"ANSWER", "STOP", "ASK_USER", "THINK", "TRANSFER"}:
            allowed["FINISH_AFTER_COMMIT"].add(idx)
    return {name: sorted(idxs) for name, idxs in allowed.items()}


def random_allowed_actions(
    action_names: list[str],
    reference_allowed: dict[str, list[int]],
    seed: int = 17,
) -> dict[str, list[int]]:
    rng = random.Random(seed)
    all_indices = list(range(len(action_names)))
    out = {}
    for name, idxs in sorted(reference_allowed.items()):
        sample_size = min(len(all_indices), max(1, len(idxs)))
        out[name] = sorted(rng.sample(all_indices, sample_size))
    return out


def logits_to_actions(logits: torch.Tensor, action_names: list[str]) -> list[str]:
    pred = logits.argmax(dim=1).cpu().tolist()
    return [action_names[int(idx)] for idx in pred]


def skill_logits_from_action_logits(logits: torch.Tensor, action_names: list[str]) -> tuple[list[str], torch.Tensor]:
    groups = skill_groups(action_names)
    group_names = sorted(groups)
    skill_scores = []
    for skill in group_names:
        idx = torch.tensor(groups[skill], dtype=torch.long, device=logits.device)
        skill_scores.append(torch.logsumexp(logits.index_select(1, idx), dim=1))
    return group_names, torch.stack(skill_scores, dim=1)


def logsumexp_skill_predictions(logits: torch.Tensor, action_names: list[str]) -> tuple[list[str], list[str]]:
    groups = skill_groups(action_names)
    group_names, skill_logits = skill_logits_from_action_logits(logits, action_names)
    pred_skill_ids = skill_logits.argmax(dim=1).cpu().tolist()
    pred_skills = [group_names[int(idx)] for idx in pred_skill_ids]
    actions: list[str] = []
    for row_idx, skill in enumerate(pred_skills):
        idxs = groups[skill]
        local_logits = logits[row_idx, torch.tensor(idxs, dtype=torch.long, device=logits.device)]
        actions.append(action_names[idxs[int(local_logits.argmax().item())]])
    return pred_skills, actions


def soft_skill_conditioned_actions(
    logits: torch.Tensor,
    action_names: list[str],
    skill_log_probs: torch.Tensor,
    skill_names: list[str],
    beta: float,
    gate_mask: torch.Tensor | None = None,
) -> list[str]:
    skill_to_pos = {skill: idx for idx, skill in enumerate(skill_names)}
    adjusted = logits.clone()
    for action_idx, action in enumerate(action_names):
        skill_idx = skill_to_pos[action_to_skill(action)]
        adjusted[:, action_idx] = adjusted[:, action_idx] + beta * skill_log_probs[:, skill_idx].to(adjusted.device)
    if gate_mask is not None:
        adjusted = torch.where(gate_mask[:, None].to(adjusted.device), adjusted, logits)
    return logits_to_actions(adjusted, action_names)


def selective_hard_skill_actions(
    logits: torch.Tensor,
    action_names: list[str],
    skills: list[str],
    gate_mask: torch.Tensor,
) -> list[str]:
    static_actions = logits_to_actions(logits, action_names)
    reranked_actions = constrained_actions_by_skill(logits, action_names, skills)
    gates = gate_mask.cpu().tolist()
    return [reranked_actions[idx] if bool(gates[idx]) else static_actions[idx] for idx in range(len(static_actions))]


def source_skill_transition_log_probs(
    rows: list[dict[str, Any]],
    train_idx: list[int],
    source_skills: list[str],
    skill_names: list[str],
    smoothing: float,
) -> dict[str, torch.Tensor]:
    selected = set(train_idx)
    groups: dict[tuple[str, int, int], list[tuple[int, int]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        if idx not in selected:
            continue
        key = (str(row.get("domain")), int(row.get("task_id", -1)), int(row.get("trial", 0)))
        groups[key].append((int(row.get("turn_index") or 0), idx))
    counts: dict[str, torch.Tensor] = {}
    for skill in ["START", *skill_names]:
        counts[skill] = torch.full((len(skill_names),), float(smoothing))
    skill_to_pos = {skill: idx for idx, skill in enumerate(skill_names)}
    for items in groups.values():
        prev = "START"
        for _, idx in sorted(items):
            current = source_skills[idx]
            counts[prev][skill_to_pos[current]] += 1.0
            prev = current
    return {skill: torch.log(vec / vec.sum()) for skill, vec in counts.items()}


def temporal_soft_skill_actions(
    logits: torch.Tensor,
    rows: list[dict[str, Any]],
    action_names: list[str],
    base_skill_log_probs: torch.Tensor,
    skill_names: list[str],
    transition_log_probs: dict[str, torch.Tensor],
    beta: float,
    temporal_weight: float,
) -> list[str]:
    groups: dict[tuple[str, int, int], list[tuple[int, int]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        key = (str(row.get("domain")), int(row.get("task_id", -1)), int(row.get("trial", 0)))
        groups[key].append((int(row.get("turn_index") or 0), idx))

    adjusted_skill_log_probs = base_skill_log_probs.clone()
    predicted_skills: dict[int, str] = {}
    for items in groups.values():
        prev_skill = "START"
        for _, idx in sorted(items):
            prior = transition_log_probs.get(prev_skill, transition_log_probs["START"]).to(base_skill_log_probs.device)
            adjusted_skill_log_probs[idx] = base_skill_log_probs[idx] + temporal_weight * prior
            predicted_skill = skill_names[int(adjusted_skill_log_probs[idx].argmax().item())]
            predicted_skills[idx] = predicted_skill
            prev_skill = predicted_skill
    return soft_skill_conditioned_actions(logits, action_names, adjusted_skill_log_probs, skill_names, beta)


def one_hot(index: int, size: int, device: torch.device) -> torch.Tensor:
    out = torch.zeros(size, device=device)
    out[index] = 1.0
    return out


def row_feature(
    action_logits: torch.Tensor,
    skill_log_probs: torch.Tensor,
    prev_skill_idx: int,
    turn_index: int,
    max_turn_index: int,
    num_prev_labels: int,
    prompt_feature: torch.Tensor | None = None,
    *,
    use_action_logits: bool = True,
    use_turn_index: bool = True,
    use_prev_label: bool = True,
    use_prompt_features: bool = False,
) -> torch.Tensor:
    parts = []
    if use_action_logits:
        centered_actions = action_logits - action_logits.mean()
        parts.append(centered_actions / centered_actions.std().clamp_min(1e-6))
    action_top2 = torch.topk(torch.softmax(action_logits, dim=0), k=min(2, action_logits.numel())).values
    if action_top2.numel() == 1:
        action_margin = action_top2.new_tensor([float(action_top2[0])])
    else:
        action_margin = (action_top2[0] - action_top2[1]).view(1)
    skill_probs = torch.softmax(skill_log_probs, dim=0)
    skill_top2 = torch.topk(skill_probs, k=min(2, skill_probs.numel())).values
    if skill_top2.numel() == 1:
        skill_margin = skill_top2.new_tensor([float(skill_top2[0])])
    else:
        skill_margin = (skill_top2[0] - skill_top2[1]).view(1)
    parts.extend([skill_log_probs, action_margin, skill_margin])
    if use_turn_index:
        parts.append(
            action_logits.new_tensor(
                [
                    float(turn_index) / max(1, max_turn_index),
                    1.0 if turn_index == 0 else 0.0,
                ]
            )
        )
    if use_prev_label:
        parts.append(one_hot(prev_skill_idx, num_prev_labels + 1, action_logits.device))
    if use_prompt_features and prompt_feature is not None:
        parts.append(prompt_feature.to(action_logits.device))
    return torch.cat(parts, dim=0)


def grouped_indices(rows: list[dict[str, Any]], selected: set[int] | None = None) -> list[list[int]]:
    grouped: dict[tuple[str, int, int], list[tuple[int, int]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        if selected is not None and idx not in selected:
            continue
        key = (str(row.get("domain")), int(row.get("task_id", -1)), int(row.get("trial", 0)))
        grouped[key].append((int(row.get("turn_index") or 0), idx))
    return [[idx for _, idx in sorted(items)] for items in grouped.values()]


def fit_label_binder(
    rows: list[dict[str, Any]],
    train_idx: list[int],
    target_logits: torch.Tensor,
    skill_log_probs: torch.Tensor,
    labels: list[str],
    label_names: list[str],
    ridge: float,
    *,
    use_action_logits: bool = True,
    use_turn_index: bool = True,
    use_prev_label: bool = True,
    use_prompt_features: bool = False,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    label_to_pos = {label: idx for idx, label in enumerate(label_names)}
    max_turn = max(int(row.get("turn_index") or 0) for row in rows)
    features = []
    targets = []
    for group in grouped_indices(rows, set(train_idx)):
        prev_label_idx = len(label_names)
        for idx in group:
            features.append(
                row_feature(
                    target_logits[idx],
                    skill_log_probs[idx],
                    prev_label_idx,
                    int(rows[idx].get("turn_index") or 0),
                    max_turn,
                    len(label_names),
                    prefix_features(rows[idx], target_logits.device),
                    use_action_logits=use_action_logits,
                    use_turn_index=use_turn_index,
                    use_prev_label=use_prev_label,
                    use_prompt_features=use_prompt_features,
                )
            )
            current = label_to_pos[labels[idx]]
            targets.append(current)
            prev_label_idx = current
    x = torch.stack(features, dim=0).float()
    y = torch.zeros((len(targets), len(label_names)), dtype=torch.float32)
    y[torch.arange(len(targets)), torch.tensor(targets, dtype=torch.long)] = 1.0
    x_aug = torch.cat([x, torch.ones((x.shape[0], 1), dtype=x.dtype)], dim=1)
    if sample_weights is not None:
        if sample_weights.numel() != x_aug.shape[0]:
            raise ValueError("sample_weights must match the number of training examples")
        scale = sample_weights.float().clamp_min(1e-6).sqrt().view(-1, 1)
        x_fit = x_aug * scale
        y_fit = y * scale
    else:
        x_fit = x_aug
        y_fit = y
    reg = torch.eye(x_aug.shape[1], dtype=x.dtype) * float(ridge)
    reg[-1, -1] = 0.0
    return torch.linalg.solve(x_fit.T @ x_fit + reg, x_fit.T @ y_fit)


def fit_skill_binder(
    rows: list[dict[str, Any]],
    train_idx: list[int],
    target_logits: torch.Tensor,
    skill_log_probs: torch.Tensor,
    label_skills: list[str],
    skill_names: list[str],
    ridge: float,
    *,
    use_action_logits: bool = True,
    use_turn_index: bool = True,
    use_prev_label: bool = True,
    use_prompt_features: bool = False,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    return fit_label_binder(
        rows,
        train_idx,
        target_logits,
        skill_log_probs,
        label_skills,
        skill_names,
        ridge,
        use_action_logits=use_action_logits,
        use_turn_index=use_turn_index,
        use_prev_label=use_prev_label,
        use_prompt_features=use_prompt_features,
        sample_weights=sample_weights,
    )


def training_order_weights(
    rows: list[dict[str, Any]],
    train_idx: list[int],
    label_skills: list[str],
    *,
    execute_weight: float,
) -> torch.Tensor:
    """Outcome-weight proxy from source-derived skills.

    The outcome audit showed that only the Commit/Execute oracle changed DB hash
    on SmolLM3. We therefore upweight Source-predicted EXECUTE states without
    using target action labels.
    """
    weights = []
    for group in grouped_indices(rows, set(train_idx)):
        for idx in group:
            weight = float(execute_weight) if label_skills[idx] == "EXECUTE" else 1.0
            weights.append(weight)
    values = torch.tensor(weights, dtype=torch.float32)
    return values / values.mean().clamp_min(1e-6)


def decode_label_binder(
    rows: list[dict[str, Any]],
    target_logits: torch.Tensor,
    skill_log_probs: torch.Tensor,
    label_names: list[str],
    coef: torch.Tensor,
    *,
    use_action_logits: bool = True,
    use_turn_index: bool = True,
    use_prev_label: bool = True,
    use_prompt_features: bool = False,
) -> tuple[list[str], torch.Tensor]:
    max_turn = max(int(row.get("turn_index") or 0) for row in rows)
    pred_scores = torch.empty((len(rows), len(label_names)), dtype=torch.float32)
    pred_labels = [""] * len(rows)
    for group in grouped_indices(rows):
        prev_label_idx = len(label_names)
        for idx in group:
            feat = row_feature(
                target_logits[idx],
                skill_log_probs[idx],
                prev_label_idx,
                int(rows[idx].get("turn_index") or 0),
                max_turn,
                len(label_names),
                prefix_features(rows[idx], target_logits.device),
                use_action_logits=use_action_logits,
                use_turn_index=use_turn_index,
                use_prev_label=use_prev_label,
                use_prompt_features=use_prompt_features,
            ).float()
            feat = torch.cat([feat, feat.new_ones(1)], dim=0)
            scores = feat @ coef
            pred_scores[idx] = scores
            pred_idx = int(scores.argmax().item())
            pred_labels[idx] = label_names[pred_idx]
            prev_label_idx = pred_idx
    return pred_labels, pred_scores


def decode_skill_binder(
    rows: list[dict[str, Any]],
    target_logits: torch.Tensor,
    skill_log_probs: torch.Tensor,
    skill_names: list[str],
    coef: torch.Tensor,
    *,
    use_action_logits: bool = True,
    use_turn_index: bool = True,
    use_prev_label: bool = True,
    use_prompt_features: bool = False,
) -> tuple[list[str], torch.Tensor]:
    return decode_label_binder(
        rows,
        target_logits,
        skill_log_probs,
        skill_names,
        coef,
        use_action_logits=use_action_logits,
        use_turn_index=use_turn_index,
        use_prev_label=use_prev_label,
        use_prompt_features=use_prompt_features,
    )


def selective_critical_skill_actions(
    logits: torch.Tensor,
    action_names: list[str],
    base_actions: list[str],
    skills: list[str],
    skill_scores: torch.Tensor,
    skill_names: list[str],
    *,
    critical_skill: str = "EXECUTE",
    confidence_threshold: float = 0.35,
) -> list[str]:
    """Only allow skill override on outcome-critical skill states.

    This follows the outcome audit: do not perturb every decision state; only
    constrain actions when the binder predicts a high-confidence EXECUTE skill.
    """
    constrained = constrained_actions_by_skill(logits, action_names, skills)
    skill_to_pos = {skill: idx for idx, skill in enumerate(skill_names)}
    critical_pos = skill_to_pos.get(critical_skill)
    if critical_pos is None:
        return list(base_actions)
    probs = torch.softmax(skill_scores, dim=1)
    out = list(base_actions)
    for idx, skill in enumerate(skills):
        if skill == critical_skill and float(probs[idx, critical_pos].item()) >= confidence_threshold:
            out[idx] = constrained[idx]
    return out


def selective_skill_set_actions(
    logits: torch.Tensor,
    action_names: list[str],
    base_actions: list[str],
    skills: list[str],
    allowed_skills: set[str],
) -> list[str]:
    """Override the base policy only for a small set of outcome-critical skills."""
    constrained = constrained_actions_by_skill(logits, action_names, skills)
    out = list(base_actions)
    for idx, skill in enumerate(skills):
        if skill in allowed_skills:
            out[idx] = constrained[idx]
    return out


def soft_actions_from_skill_scores(
    logits: torch.Tensor,
    action_names: list[str],
    skill_scores: torch.Tensor,
    skill_names: list[str],
    beta: float,
) -> list[str]:
    return soft_skill_conditioned_actions(
        logits,
        action_names,
        torch.log_softmax(skill_scores, dim=1),
        skill_names,
        beta,
    )


def constrained_actions_by_skill(logits: torch.Tensor, action_names: list[str], skills: list[str]) -> list[str]:
    return constrained_actions_by_label(
        logits,
        action_names,
        skills,
        {action: action_to_skill(action) for action in action_names},
    )


def constrained_actions_by_label(
    logits: torch.Tensor,
    action_names: list[str],
    labels: list[str],
    action_to_label: dict[str, str],
) -> list[str]:
    groups = skill_groups(action_names)
    if any(action_to_label[action] != action_to_skill(action) for action in action_names):
        groups = defaultdict(list)
        for idx, action in enumerate(action_names):
            groups[action_to_label[action]].append(idx)
        groups = dict(groups)
    fallback = logits.argmax(dim=1).cpu().tolist()
    actions: list[str] = []
    for row_idx, label in enumerate(labels):
        idxs = groups.get(label)
        if not idxs:
            actions.append(action_names[int(fallback[row_idx])])
            continue
        local_logits = logits[row_idx, torch.tensor(idxs, dtype=torch.long, device=logits.device)]
        actions.append(action_names[idxs[int(local_logits.argmax().item())]])
    return actions


def constrained_actions_by_allowed_sets(
    logits: torch.Tensor,
    action_names: list[str],
    labels: list[str],
    allowed_sets: dict[str, list[int]],
) -> list[str]:
    fallback = logits.argmax(dim=1).cpu().tolist()
    actions: list[str] = []
    for row_idx, label in enumerate(labels):
        idxs = allowed_sets.get(label)
        if not idxs:
            actions.append(action_names[int(fallback[row_idx])])
            continue
        local_logits = logits[row_idx, torch.tensor(idxs, dtype=torch.long, device=logits.device)]
        actions.append(action_names[idxs[int(local_logits.argmax().item())]])
    return actions


def latent_prefix_feature(
    latent: torch.Tensor,
    row: dict[str, Any],
    prev_label_idx: int,
    num_prev_labels: int,
    *,
    use_prev_label: bool = True,
) -> torch.Tensor:
    centered = latent - latent.mean()
    parts = [centered / centered.std().clamp_min(1e-6), prefix_features(row, latent.device)]
    if use_prev_label:
        parts.append(one_hot(prev_label_idx, num_prev_labels + 1, latent.device))
    return torch.cat(parts, dim=0)


def fit_latent_prefix_binder(
    rows: list[dict[str, Any]],
    train_idx: list[int],
    target_latents: torch.Tensor,
    labels: list[str],
    label_names: list[str],
    ridge: float,
    *,
    use_prev_label: bool = True,
) -> torch.Tensor:
    label_to_pos = {label: idx for idx, label in enumerate(label_names)}
    features = []
    targets = []
    for group in grouped_indices(rows, set(train_idx)):
        prev_label_idx = len(label_names)
        for idx in group:
            features.append(
                latent_prefix_feature(
                    target_latents[idx],
                    rows[idx],
                    prev_label_idx,
                    len(label_names),
                    use_prev_label=use_prev_label,
                )
            )
            current = label_to_pos[labels[idx]]
            targets.append(current)
            prev_label_idx = current
    x = torch.stack(features, dim=0).float()
    y = torch.zeros((len(targets), len(label_names)), dtype=torch.float32)
    y[torch.arange(len(targets)), torch.tensor(targets, dtype=torch.long)] = 1.0
    x_aug = torch.cat([x, torch.ones((x.shape[0], 1), dtype=x.dtype)], dim=1)
    reg = torch.eye(x_aug.shape[1], dtype=x.dtype) * float(ridge)
    reg[-1, -1] = 0.0
    return torch.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)


def decode_latent_prefix_binder(
    rows: list[dict[str, Any]],
    target_latents: torch.Tensor,
    label_names: list[str],
    coef: torch.Tensor,
    *,
    use_prev_label: bool = True,
) -> tuple[list[str], torch.Tensor]:
    pred_scores = torch.empty((len(rows), len(label_names)), dtype=torch.float32)
    pred_labels = [""] * len(rows)
    for group in grouped_indices(rows):
        prev_label_idx = len(label_names)
        for idx in group:
            feat = latent_prefix_feature(
                target_latents[idx],
                rows[idx],
                prev_label_idx,
                len(label_names),
                use_prev_label=use_prev_label,
            ).float()
            feat = torch.cat([feat, feat.new_ones(1)], dim=0)
            scores = feat @ coef
            pred_scores[idx] = scores
            pred_idx = int(scores.argmax().item())
            pred_labels[idx] = label_names[pred_idx]
            prev_label_idx = pred_idx
    return pred_labels, pred_scores


def transition_delta_features(rows: list[dict[str, Any]], logits: torch.Tensor) -> torch.Tensor:
    deltas = torch.zeros_like(logits)
    for group in grouped_indices(rows):
        prev = None
        for idx in group:
            if prev is not None:
                deltas[idx] = logits[idx] - logits[prev]
            prev = idx
    return deltas


def source_discovery_features(rows: list[dict[str, Any]], source_logits: torch.Tensor) -> torch.Tensor:
    centered = source_logits - source_logits.mean(dim=1, keepdim=True)
    scaled = centered / centered.std(dim=1, keepdim=True).clamp_min(1e-6)
    delta = transition_delta_features(rows, source_logits)
    delta = delta / delta.std(dim=1, keepdim=True).clamp_min(1e-6)
    prompts = torch.stack([prefix_features(row, source_logits.device) for row in rows], dim=0)
    return torch.cat([scaled, delta, prompts], dim=1).float()


def fit_kmeans(features: torch.Tensor, train_idx: list[int], k: int, iters: int = 30) -> torch.Tensor:
    train = features[torch.tensor(train_idx, dtype=torch.long)]
    if train.shape[0] < k:
        raise ValueError("not enough training rows for k-means")
    init_pos = torch.linspace(0, train.shape[0] - 1, steps=k).round().long()
    centroids = train.index_select(0, init_pos).clone()
    for _ in range(iters):
        dist = torch.cdist(train, centroids)
        labels = dist.argmin(dim=1)
        new_centroids = centroids.clone()
        for cluster_id in range(k):
            mask = labels == cluster_id
            if bool(mask.any()):
                new_centroids[cluster_id] = train[mask].mean(dim=0)
        if torch.allclose(new_centroids, centroids, atol=1e-5):
            break
        centroids = new_centroids
    return centroids


def assign_clusters(features: torch.Tensor, centroids: torch.Tensor) -> list[str]:
    labels = torch.cdist(features, centroids).argmin(dim=1).cpu().tolist()
    return [f"DISCOVERED_{int(label)}" for label in labels]


def discovered_cluster_allowed_actions(
    train_idx: list[int],
    cluster_labels: list[str],
    source_actions: list[str],
    action_names: list[str],
    min_actions: int = 2,
    coverage: float = 0.85,
) -> dict[str, list[int]]:
    action_to_idx = {action: idx for idx, action in enumerate(action_names)}
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for idx in train_idx:
        counts[cluster_labels[idx]][source_actions[idx]] += 1
    allowed = {}
    fallback = list(range(len(action_names)))
    for label, action_counts in counts.items():
        total = sum(action_counts.values())
        running = 0
        selected = []
        for action, count in sorted(action_counts.items(), key=lambda item: item[1], reverse=True):
            selected.append(action_to_idx[action])
            running += count
            if len(selected) >= min_actions and running / max(1, total) >= coverage:
                break
        allowed[label] = sorted(selected) if selected else fallback
    return allowed


OBSERVATION_TYPES = ["none", "success", "failure", "empty", "conflict", "commit"]


def last_tool_observation(row: dict[str, Any]) -> str:
    convo = conversation_segment(str(row.get("prompt") or ""))
    for line in reversed([line.strip() for line in convo.splitlines() if line.strip()]):
        if line.lower().startswith("tool "):
            return line
    return ""


def observation_type(row: dict[str, Any]) -> str:
    obs = last_tool_observation(row).lower()
    if not obs:
        return "none"
    if any(term in obs for term in ("error", "invalid", "permission", "cannot", "can't", "failed", "failure")):
        return "failure"
    if any(term in obs for term in ("[]", "empty", "no result", "not found", "none found")):
        return "empty"
    if any(term in obs for term in ("conflict", "inconsistent", "mismatch", "already", "different")):
        return "conflict"
    if any(name in obs for name in ("update_", "cancel_", "return_", "book_", "send_", "modify_")):
        return "commit"
    return "success"


def action_to_branch(action: str) -> str:
    if action in {"ANSWER", "STOP"}:
        return "finish"
    if action in {"ASK_USER", "THINK", "TRANSFER"}:
        return "recover"
    if action == "VERIFY":
        return "verify"
    if action.startswith(COLLECT_PREFIXES):
        return "acquire"
    if action.startswith(EXECUTE_PREFIXES):
        return "execute"
    return "recover"


def branch_action_indices(action_names: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, action in enumerate(action_names):
        groups[action_to_branch(action)].append(idx)
    return dict(groups)


def logsumexp_indices(logits: torch.Tensor, indices: list[int]) -> torch.Tensor:
    if not indices:
        return logits.new_tensor(-1e9)
    idx = torch.tensor(indices, dtype=torch.long, device=logits.device)
    return torch.logsumexp(logits.index_select(0, idx), dim=0)


def apply_write_pair_calibration(
    rows: list[dict[str, Any]],
    logits: torch.Tensor,
    action_names: list[str],
    margin: float,
    max_strength: float,
) -> torch.Tensor:
    """Observation-conditioned write branch prior for real tau states.

    It uses only current prompt history: the last concrete tool family and the
    parsed observation type. No target action labels are used.
    """
    groups = branch_action_indices(action_names)
    finish = groups.get("finish", [])
    recover = groups.get("recover", [])
    verify = groups.get("verify", [])
    acquire = groups.get("acquire", [])
    execute = groups.get("execute", [])
    non_finish = recover + verify + acquire + execute
    corrected = logits.clone()
    if not finish or not non_finish:
        return corrected
    for row_idx, row in enumerate(rows):
        if tool_family(last_tool_name(str(row.get("prompt") or ""))) != "write":
            continue
        obs = observation_type(row)
        if obs in {"success", "commit"}:
            finish_score = logsumexp_indices(corrected[row_idx], finish)
            bad_score = logsumexp_indices(corrected[row_idx], non_finish)
            need = min(float(max_strength), max(0.0, float(margin) - float((finish_score - bad_score).item())))
            if need <= 0.0:
                continue
            corrected[row_idx, finish] += 0.6 * need
            corrected[row_idx, non_finish] -= 0.15 * need
        elif obs == "failure":
            good = recover + verify
            bad = finish + execute
            if not good or not bad:
                continue
            good_score = logsumexp_indices(corrected[row_idx], good)
            bad_score = logsumexp_indices(corrected[row_idx], bad)
            need = min(float(max_strength), max(0.0, float(margin) - float((good_score - bad_score).item())))
            if need <= 0.0:
                continue
            corrected[row_idx, good] += 0.4 * need
            corrected[row_idx, bad] -= 0.4 * need
    return corrected


def observation_features(row: dict[str, Any], device: torch.device) -> torch.Tensor:
    obs_type = observation_type(row)
    values = [1.0 if obs_type == name else 0.0 for name in OBSERVATION_TYPES]
    obs = last_tool_observation(row).lower()
    values.extend(
        [
            1.0 if "{" in obs and "}" in obs else 0.0,
            1.0 if "user" in obs else 0.0,
            1.0 if "order" in obs or "reservation" in obs else 0.0,
            1.0 if "refund" in obs or "payment" in obs else 0.0,
        ]
    )
    return torch.tensor(values, dtype=torch.float32, device=device)


def response_delta_features(rows: list[dict[str, Any]], logits: torch.Tensor) -> torch.Tensor:
    deltas = transition_delta_features(rows, logits)
    centered = deltas - deltas.mean(dim=1, keepdim=True)
    return centered / centered.std(dim=1, keepdim=True).clamp_min(1e-6)


def response_operator_features(rows: list[dict[str, Any]], logits: torch.Tensor) -> torch.Tensor:
    centered = logits - logits.mean(dim=1, keepdim=True)
    scaled_logits = centered / centered.std(dim=1, keepdim=True).clamp_min(1e-6)
    deltas = response_delta_features(rows, logits)
    obs = torch.stack([observation_features(row, logits.device) for row in rows], dim=0)
    prefix = torch.stack([prefix_features(row, logits.device) for row in rows], dim=0)
    return torch.cat([scaled_logits, deltas, obs, prefix], dim=1).float()


def fit_feature_label_binder(
    features: torch.Tensor,
    train_idx: list[int],
    labels: list[str],
    label_names: list[str],
    ridge: float,
) -> torch.Tensor:
    label_to_pos = {label: idx for idx, label in enumerate(label_names)}
    x = features[torch.tensor(train_idx, dtype=torch.long)].float()
    target_pos = torch.tensor([label_to_pos[labels[idx]] for idx in train_idx], dtype=torch.long)
    y = torch.zeros((x.shape[0], len(label_names)), dtype=torch.float32)
    y[torch.arange(x.shape[0]), target_pos] = 1.0
    x_aug = torch.cat([x, torch.ones((x.shape[0], 1), dtype=x.dtype)], dim=1)
    reg = torch.eye(x_aug.shape[1], dtype=x.dtype) * float(ridge)
    reg[-1, -1] = 0.0
    return torch.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)


def decode_feature_label_binder(
    features: torch.Tensor,
    label_names: list[str],
    coef: torch.Tensor,
) -> tuple[list[str], torch.Tensor]:
    x = features.float()
    x_aug = torch.cat([x, torch.ones((x.shape[0], 1), dtype=x.dtype)], dim=1)
    scores = x_aug @ coef
    pred = scores.argmax(dim=1).cpu().tolist()
    return [label_names[int(idx)] for idx in pred], scores


def residuals_by_label(
    train_idx: list[int],
    labels: list[str],
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    label_names: list[str],
) -> dict[str, torch.Tensor]:
    residuals: dict[str, list[torch.Tensor]] = defaultdict(list)
    for idx in train_idx:
        residuals[labels[idx]].append(source_logits[idx] - target_logits[idx])
    zero = torch.zeros(source_logits.shape[1], dtype=torch.float32)
    return {
        label: torch.stack(residuals[label], dim=0).mean(dim=0).float() if residuals.get(label) else zero.clone()
        for label in label_names
    }


def apply_label_residuals(
    logits: torch.Tensor,
    labels: list[str],
    residuals: dict[str, torch.Tensor],
    alpha: float,
    *,
    gate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    corrected = logits.clone()
    for idx, label in enumerate(labels):
        if gate_mask is not None and not bool(gate_mask[idx].item()):
            continue
        corrected[idx] = corrected[idx] + float(alpha) * residuals.get(label, torch.zeros_like(corrected[idx]))
    return corrected


def response_branch_gate(rows: list[dict[str, Any]], action_margin: torch.Tensor) -> torch.Tensor:
    branch_types = {"failure", "empty", "conflict", "commit"}
    values = []
    for idx, row in enumerate(rows):
        values.append(observation_type(row) in branch_types or float(action_margin[idx].item()) <= 0.08)
    return torch.tensor(values, dtype=torch.bool)


def transition_pairs(rows: list[dict[str, Any]], selected: list[int]) -> list[tuple[int, int]]:
    selected_set = set(selected)
    grouped: dict[tuple[str, int, int], list[tuple[int, int]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        if idx not in selected_set:
            continue
        key = (str(row.get("domain")), int(row.get("task_id", -1)), int(row.get("trial", 0)))
        grouped[key].append((int(row.get("turn_index") or 0), idx))
    pairs: list[tuple[int, int]] = []
    for items in grouped.values():
        ordered = [idx for _, idx in sorted(items)]
        pairs.extend(zip(ordered, ordered[1:]))
    return pairs


def metrics(
    name: str,
    rows: list[dict[str, Any]],
    selected: list[int],
    pred_actions: list[str],
    source_actions: list[str],
) -> dict[str, Any]:
    expected = [str(row.get("correct_action")) for row in rows]
    pred_skills = [action_to_skill(action) for action in pred_actions]
    expected_skills = [action_to_skill(action) for action in expected]
    source_skills = [action_to_skill(action) for action in source_actions]
    action_correct = [pred_actions[idx] == expected[idx] for idx in selected]
    skill_correct = [pred_skills[idx] == expected_skills[idx] for idx in selected]
    source_action_agree = [pred_actions[idx] == source_actions[idx] for idx in selected]
    source_skill_agree = [pred_skills[idx] == source_skills[idx] for idx in selected]
    same_skill_wrong = [
        pred_skills[idx] == expected_skills[idx] and pred_actions[idx] != expected[idx]
        for idx in selected
    ]
    boundary_indices = []
    non_boundary_indices = []
    grouped: dict[tuple[str, int, int], list[tuple[int, int]]] = defaultdict(list)
    selected_set = set(selected)
    for idx, row in enumerate(rows):
        if idx not in selected_set:
            continue
        key = (str(row.get("domain")), int(row.get("task_id", -1)), int(row.get("trial", 0)))
        grouped[key].append((int(row.get("turn_index") or 0), idx))
    for items in grouped.values():
        previous = None
        for _, idx in sorted(items):
            current = expected_skills[idx]
            if previous is not None and current != previous:
                boundary_indices.append(idx)
            else:
                non_boundary_indices.append(idx)
            previous = current
    boundary_skill_correct = [pred_skills[idx] == expected_skills[idx] for idx in boundary_indices]
    non_boundary_skill_correct = [pred_skills[idx] == expected_skills[idx] for idx in non_boundary_indices]
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for idx in selected:
        confusion[expected_skills[idx]][pred_skills[idx]] += 1
    pairs = transition_pairs(rows, selected)
    action_transition = [
        pred_actions[left] == expected[left] and pred_actions[right] == expected[right]
        for left, right in pairs
    ]
    skill_transition = [
        pred_skills[left] == expected_skills[left] and pred_skills[right] == expected_skills[right]
        for left, right in pairs
    ]
    return {
        "method": name,
        "action_accuracy": sum(action_correct) / max(1, len(action_correct)),
        "skill_accuracy": sum(skill_correct) / max(1, len(skill_correct)),
        "skill_minus_action": (sum(skill_correct) - sum(action_correct)) / max(1, len(action_correct)),
        "source_action_agreement": sum(source_action_agree) / max(1, len(source_action_agree)),
        "source_skill_agreement": sum(source_skill_agree) / max(1, len(source_skill_agree)),
        "same_skill_wrong_action_rate": sum(same_skill_wrong) / max(1, len(same_skill_wrong)),
        "boundary_skill_accuracy": sum(boundary_skill_correct) / max(1, len(boundary_skill_correct)),
        "non_boundary_skill_accuracy": sum(non_boundary_skill_correct) / max(1, len(non_boundary_skill_correct)),
        "num_boundary_states": len(boundary_indices),
        "action_transition_accuracy": sum(action_transition) / max(1, len(action_transition)),
        "skill_transition_accuracy": sum(skill_transition) / max(1, len(skill_transition)),
        "skill_confusion": {gold: dict(preds) for gold, preds in sorted(confusion.items())},
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    rows = load_rows(args.decision_jsonl)
    train_idx, val_idx, split_metadata = eval_protocol.load_split(args.split_json, len(rows))
    train_selected = [int(idx) for idx in train_idx.tolist()]
    selected = [int(idx) for idx in val_idx.tolist()]
    source_obj = load_hidden_tensor(args.source_tensor)
    target_obj = load_hidden_tensor(args.target_tensor)
    if len(rows) != int(source_obj["h"].shape[0]) or len(rows) != int(target_obj["h"].shape[0]):
        raise ValueError("decision rows and hidden tensors must have the same length")

    source_parts = load_source_parts(args.source_policy, device)
    action_names = source_parts.action_names or list(rows[0]["action_names"])
    alignment = torch.load(args.alignment, map_location="cpu")
    target_projection = RidgeProjection(alignment["coef"]).to(device).eval()
    target_policy = torch.nn.Sequential(target_projection, source_parts.head).to(device).eval()

    with torch.no_grad():
        source_z = source_latents(source_parts.adapter, source_obj["h"], device, args.batch_size)
        source_logits = source_parts.head(source_z.to(device)).cpu()
        target_logits = []
        target_z = []
        h_target = target_obj["h"]
        for start in range(0, h_target.shape[0], args.batch_size):
            z_batch = target_projection(h_target[start : start + args.batch_size].to(device))
            target_z.append(z_batch.cpu())
            target_logits.append(source_parts.head(z_batch).cpu())
        target_logits = torch.cat(target_logits, dim=0)
        target_z = torch.cat(target_z, dim=0)

    expected_actions = [str(row.get("correct_action")) for row in rows]
    expected_skills = [action_to_skill(action) for action in expected_actions]
    prefix_skills = [prefix_defined_skill(row) for row in rows]
    prefix_skill_names = sorted(set(prefix_skills))
    prefix_allowed = prefix_skill_allowed_actions(action_names)
    prefix_random_allowed = random_allowed_actions(action_names, prefix_allowed)
    source_actions = logits_to_actions(source_logits, action_names)
    source_skills = [action_to_skill(action) for action in source_actions]
    semantic_skill_groups = skill_groups(action_names)
    random_group_map = random_action_groups(action_names, semantic_skill_groups)
    random_group_names = sorted(set(random_group_map.values()))
    source_random_groups = labels_from_action_map(source_actions, random_group_map)
    static_actions = logits_to_actions(target_logits, action_names)
    predicted_skills, predicted_skill_actions = logsumexp_skill_predictions(target_logits, action_names)
    skill_names, target_skill_logits = skill_logits_from_action_logits(target_logits, action_names)
    target_skill_log_probs = torch.log_softmax(target_skill_logits, dim=1)
    target_skill_probs = torch.softmax(target_skill_logits, dim=1)
    top_skill_probs, top_skill_ids = target_skill_probs.max(dim=1)
    sorted_skill_probs = torch.sort(target_skill_probs, dim=1, descending=True).values
    action_probs = torch.softmax(target_logits, dim=1)
    sorted_action_probs = torch.sort(action_probs, dim=1, descending=True).values
    action_margin = sorted_action_probs[:, 0] - sorted_action_probs[:, 1]
    skill_margin = sorted_skill_probs[:, 0] - sorted_skill_probs[:, 1]
    skill_conf_gate = (top_skill_probs >= args.skill_confidence_threshold) & (
        skill_margin >= args.selective_skill_margin_threshold
    )
    selective_hard_gate = skill_conf_gate & (action_margin <= args.selective_action_margin_threshold)
    soft_skill_actions = soft_skill_conditioned_actions(
        target_logits,
        action_names,
        target_skill_log_probs,
        skill_names,
        args.skill_prior_beta,
    )
    gated_soft_skill_actions = soft_skill_conditioned_actions(
        target_logits,
        action_names,
        target_skill_log_probs,
        skill_names,
        args.skill_prior_beta,
        skill_conf_gate,
    )
    selective_hard_actions = selective_hard_skill_actions(
        target_logits,
        action_names,
        predicted_skills,
        selective_hard_gate,
    )
    transition_log_probs = source_skill_transition_log_probs(
        rows,
        train_selected,
        source_skills,
        skill_names,
        args.temporal_skill_smoothing,
    )
    temporal_skill_actions = temporal_soft_skill_actions(
        target_logits,
        rows,
        action_names,
        target_skill_log_probs,
        skill_names,
        transition_log_probs,
        args.skill_prior_beta,
        args.temporal_skill_weight,
    )
    prefix_oracle_actions = constrained_actions_by_allowed_sets(
        target_logits,
        action_names,
        prefix_skills,
        prefix_allowed,
    )
    prefix_random_oracle_actions = constrained_actions_by_allowed_sets(
        target_logits,
        action_names,
        prefix_skills,
        prefix_random_allowed,
    )
    prefix_latent_binder_coef = fit_latent_prefix_binder(
        rows,
        train_selected,
        target_z,
        prefix_skills,
        prefix_skill_names,
        args.skill_binder_ridge,
    )
    prefix_latent_binder_skills, _ = decode_latent_prefix_binder(
        rows,
        target_z,
        prefix_skill_names,
        prefix_latent_binder_coef,
    )
    prefix_latent_binder_actions = constrained_actions_by_allowed_sets(
        target_logits,
        action_names,
        prefix_latent_binder_skills,
        prefix_allowed,
    )
    prefix_latent_no_prev_coef = fit_latent_prefix_binder(
        rows,
        train_selected,
        target_z,
        prefix_skills,
        prefix_skill_names,
        args.skill_binder_ridge,
        use_prev_label=False,
    )
    prefix_latent_no_prev_skills, _ = decode_latent_prefix_binder(
        rows,
        target_z,
        prefix_skill_names,
        prefix_latent_no_prev_coef,
        use_prev_label=False,
    )
    prefix_latent_no_prev_actions = constrained_actions_by_allowed_sets(
        target_logits,
        action_names,
        prefix_latent_no_prev_skills,
        prefix_allowed,
    )
    discovery_features = source_discovery_features(rows, source_logits)
    discovery_centroids = fit_kmeans(discovery_features, train_selected, k=min(6, len(train_selected)))
    discovered_labels = assign_clusters(discovery_features, discovery_centroids)
    discovered_names = sorted(set(discovered_labels))
    discovered_allowed = discovered_cluster_allowed_actions(
        train_selected,
        discovered_labels,
        source_actions,
        action_names,
    )
    source_discovered_oracle_actions = constrained_actions_by_allowed_sets(
        target_logits,
        action_names,
        discovered_labels,
        discovered_allowed,
    )
    discovered_latent_binder_coef = fit_latent_prefix_binder(
        rows,
        train_selected,
        target_z,
        discovered_labels,
        discovered_names,
        args.skill_binder_ridge,
    )
    discovered_binder_labels, _ = decode_latent_prefix_binder(
        rows,
        target_z,
        discovered_names,
        discovered_latent_binder_coef,
    )
    discovered_latent_binder_actions = constrained_actions_by_allowed_sets(
        target_logits,
        action_names,
        discovered_binder_labels,
        discovered_allowed,
    )
    source_response_features = response_operator_features(rows, source_logits)
    response_centroids = fit_kmeans(
        source_response_features,
        train_selected,
        k=min(args.response_operator_clusters, len(train_selected)),
    )
    response_operator_labels = assign_clusters(source_response_features, response_centroids)
    response_operator_names = sorted(set(response_operator_labels))
    response_operator_allowed = discovered_cluster_allowed_actions(
        train_selected,
        response_operator_labels,
        source_actions,
        action_names,
        min_actions=2,
        coverage=0.80,
    )
    response_operator_residuals = residuals_by_label(
        train_selected,
        response_operator_labels,
        source_logits,
        target_logits,
        response_operator_names,
    )
    target_response_features = response_operator_features(rows, target_logits)
    response_operator_binder_coef = fit_feature_label_binder(
        target_response_features,
        train_selected,
        response_operator_labels,
        response_operator_names,
        args.skill_binder_ridge,
    )
    response_operator_binder_labels, response_operator_binder_scores = decode_feature_label_binder(
        target_response_features,
        response_operator_names,
        response_operator_binder_coef,
    )
    response_branch_mask = response_branch_gate(rows, action_margin)
    response_operator_oracle_allowed_actions = constrained_actions_by_allowed_sets(
        target_logits,
        action_names,
        response_operator_labels,
        response_operator_allowed,
    )
    response_operator_binder_allowed_actions = constrained_actions_by_allowed_sets(
        target_logits,
        action_names,
        response_operator_binder_labels,
        response_operator_allowed,
    )
    observation_labels = [f"OBS_{observation_type(row)}" for row in rows]
    observation_names = sorted(set(observation_labels))
    observation_allowed = discovered_cluster_allowed_actions(
        train_selected,
        observation_labels,
        source_actions,
        action_names,
        min_actions=2,
        coverage=0.80,
    )
    observation_residuals = residuals_by_label(
        train_selected,
        observation_labels,
        source_logits,
        target_logits,
        observation_names,
    )
    observation_type_allowed_actions = constrained_actions_by_allowed_sets(
        target_logits,
        action_names,
        observation_labels,
        observation_allowed,
    )
    observation_type_residual_actions = logits_to_actions(
        apply_label_residuals(
            target_logits,
            observation_labels,
            observation_residuals,
            args.response_residual_alpha,
        ),
        action_names,
    )
    response_obs_labels = [
        f"{response_operator_labels[idx]}::{observation_labels[idx]}" for idx in range(len(rows))
    ]
    response_obs_names = sorted(set(response_obs_labels))
    response_obs_allowed = discovered_cluster_allowed_actions(
        train_selected,
        response_obs_labels,
        source_actions,
        action_names,
        min_actions=2,
        coverage=0.80,
    )
    response_obs_residuals = residuals_by_label(
        train_selected,
        response_obs_labels,
        source_logits,
        target_logits,
        response_obs_names,
    )
    response_obs_binder_labels = [
        f"{response_operator_binder_labels[idx]}::{observation_labels[idx]}" for idx in range(len(rows))
    ]
    response_operator_obs_oracle_allowed_actions = constrained_actions_by_allowed_sets(
        target_logits,
        action_names,
        response_obs_labels,
        response_obs_allowed,
    )
    response_operator_obs_binder_allowed_actions = constrained_actions_by_allowed_sets(
        target_logits,
        action_names,
        response_obs_binder_labels,
        response_obs_allowed,
    )
    response_operator_obs_oracle_residual_actions = logits_to_actions(
        apply_label_residuals(
            target_logits,
            response_obs_labels,
            response_obs_residuals,
            args.response_residual_alpha,
        ),
        action_names,
    )
    response_operator_obs_binder_residual_actions = logits_to_actions(
        apply_label_residuals(
            target_logits,
            response_obs_binder_labels,
            response_obs_residuals,
            args.response_residual_alpha,
        ),
        action_names,
    )
    write_pair_calibrated_logits = apply_write_pair_calibration(
        rows,
        target_logits,
        action_names,
        args.write_pair_calibration_margin,
        args.write_pair_max_calibration,
    )
    write_pair_calibrated_actions = logits_to_actions(write_pair_calibrated_logits, action_names)
    response_operator_obs_binder_residual_logits = apply_label_residuals(
        target_logits,
        response_obs_binder_labels,
        response_obs_residuals,
        args.response_residual_alpha,
    )
    response_operator_obs_binder_residual_write_pair_logits = apply_write_pair_calibration(
        rows,
        response_operator_obs_binder_residual_logits,
        action_names,
        args.write_pair_calibration_margin,
        args.write_pair_max_calibration,
    )
    response_operator_obs_binder_residual_write_pair_actions = logits_to_actions(
        response_operator_obs_binder_residual_write_pair_logits,
        action_names,
    )
    response_operator_oracle_residual_actions = logits_to_actions(
        apply_label_residuals(
            target_logits,
            response_operator_labels,
            response_operator_residuals,
            args.response_residual_alpha,
        ),
        action_names,
    )
    response_operator_binder_residual_actions = logits_to_actions(
        apply_label_residuals(
            target_logits,
            response_operator_binder_labels,
            response_operator_residuals,
            args.response_residual_alpha,
        ),
        action_names,
    )
    response_operator_gated_residual_actions = logits_to_actions(
        apply_label_residuals(
            target_logits,
            response_operator_binder_labels,
            response_operator_residuals,
            args.response_residual_alpha,
            gate_mask=response_branch_mask,
        ),
        action_names,
    )
    response_operator_soft_residual_actions = logits_to_actions(
        target_logits
        + args.response_residual_alpha
        * torch.stack(
            [
                sum(
                    torch.softmax(response_operator_binder_scores[idx], dim=0)[label_idx]
                    * response_operator_residuals[label]
                    for label_idx, label in enumerate(response_operator_names)
                )
                for idx in range(target_logits.shape[0])
            ],
            dim=0,
        ),
        action_names,
    )
    source_binder_coef = fit_skill_binder(
        rows,
        train_selected,
        target_logits,
        target_skill_log_probs,
        source_skills,
        skill_names,
        args.skill_binder_ridge,
    )
    source_binder_skills, source_binder_scores = decode_skill_binder(
        rows,
        target_logits,
        target_skill_log_probs,
        skill_names,
        source_binder_coef,
    )
    source_binder_actions = constrained_actions_by_skill(target_logits, action_names, source_binder_skills)
    source_binder_soft_actions = soft_actions_from_skill_scores(
        target_logits,
        action_names,
        source_binder_scores,
        skill_names,
        args.skill_binder_beta,
    )
    source_binder_no_turn_coef = fit_skill_binder(
        rows,
        train_selected,
        target_logits,
        target_skill_log_probs,
        source_skills,
        skill_names,
        args.skill_binder_ridge,
        use_turn_index=False,
    )
    source_binder_no_turn_skills, _ = decode_skill_binder(
        rows,
        target_logits,
        target_skill_log_probs,
        skill_names,
        source_binder_no_turn_coef,
        use_turn_index=False,
    )
    source_binder_no_turn_actions = constrained_actions_by_skill(
        target_logits,
        action_names,
        source_binder_no_turn_skills,
    )
    source_binder_no_prev_coef = fit_skill_binder(
        rows,
        train_selected,
        target_logits,
        target_skill_log_probs,
        source_skills,
        skill_names,
        args.skill_binder_ridge,
        use_prev_label=False,
    )
    source_binder_no_prev_skills, _ = decode_skill_binder(
        rows,
        target_logits,
        target_skill_log_probs,
        skill_names,
        source_binder_no_prev_coef,
        use_prev_label=False,
    )
    source_binder_no_prev_actions = constrained_actions_by_skill(
        target_logits,
        action_names,
        source_binder_no_prev_skills,
    )
    source_binder_no_action_logits_coef = fit_skill_binder(
        rows,
        train_selected,
        target_logits,
        target_skill_log_probs,
        source_skills,
        skill_names,
        args.skill_binder_ridge,
        use_action_logits=False,
    )
    source_binder_no_action_logits_skills, _ = decode_skill_binder(
        rows,
        target_logits,
        target_skill_log_probs,
        skill_names,
        source_binder_no_action_logits_coef,
        use_action_logits=False,
    )
    source_binder_no_action_logits_actions = constrained_actions_by_skill(
        target_logits,
        action_names,
        source_binder_no_action_logits_skills,
    )
    source_prefix_binder_coef = fit_skill_binder(
        rows,
        train_selected,
        target_logits,
        target_skill_log_probs,
        source_skills,
        skill_names,
        args.skill_binder_ridge,
        use_turn_index=False,
        use_prompt_features=True,
    )
    source_prefix_binder_skills, source_prefix_binder_scores = decode_skill_binder(
        rows,
        target_logits,
        target_skill_log_probs,
        skill_names,
        source_prefix_binder_coef,
        use_turn_index=False,
        use_prompt_features=True,
    )
    source_prefix_binder_actions = constrained_actions_by_skill(
        target_logits,
        action_names,
        source_prefix_binder_skills,
    )
    source_prefix_binder_soft_actions = soft_actions_from_skill_scores(
        target_logits,
        action_names,
        source_prefix_binder_scores,
        skill_names,
        args.skill_binder_beta,
    )
    source_prefix_binder_no_action_logits_coef = fit_skill_binder(
        rows,
        train_selected,
        target_logits,
        target_skill_log_probs,
        source_skills,
        skill_names,
        args.skill_binder_ridge,
        use_action_logits=False,
        use_turn_index=False,
        use_prompt_features=True,
    )
    source_prefix_binder_no_action_logits_skills, _ = decode_skill_binder(
        rows,
        target_logits,
        target_skill_log_probs,
        skill_names,
        source_prefix_binder_no_action_logits_coef,
        use_action_logits=False,
        use_turn_index=False,
        use_prompt_features=True,
    )
    source_prefix_binder_no_action_logits_actions = constrained_actions_by_skill(
        target_logits,
        action_names,
        source_prefix_binder_no_action_logits_skills,
    )
    outcome_weights = training_order_weights(
        rows,
        train_selected,
        source_skills,
        execute_weight=args.outcome_execute_skill_weight,
    )
    outcome_weighted_binder_coef = fit_skill_binder(
        rows,
        train_selected,
        target_logits,
        target_skill_log_probs,
        source_skills,
        skill_names,
        args.skill_binder_ridge,
        sample_weights=outcome_weights,
    )
    outcome_weighted_skills, outcome_weighted_scores = decode_skill_binder(
        rows,
        target_logits,
        target_skill_log_probs,
        skill_names,
        outcome_weighted_binder_coef,
    )
    outcome_weighted_actions = constrained_actions_by_skill(
        target_logits,
        action_names,
        outcome_weighted_skills,
    )
    outcome_weighted_execute_selective_actions = selective_critical_skill_actions(
        target_logits,
        action_names,
        static_actions,
        outcome_weighted_skills,
        outcome_weighted_scores,
        skill_names,
        confidence_threshold=args.outcome_critical_confidence_threshold,
    )
    outcome_weighted_prefix_coef = fit_skill_binder(
        rows,
        train_selected,
        target_logits,
        target_skill_log_probs,
        source_skills,
        skill_names,
        args.skill_binder_ridge,
        use_action_logits=False,
        use_turn_index=False,
        use_prompt_features=True,
        sample_weights=outcome_weights,
    )
    outcome_weighted_prefix_skills, outcome_weighted_prefix_scores = decode_skill_binder(
        rows,
        target_logits,
        target_skill_log_probs,
        skill_names,
        outcome_weighted_prefix_coef,
        use_action_logits=False,
        use_turn_index=False,
        use_prompt_features=True,
    )
    outcome_weighted_prefix_actions = constrained_actions_by_skill(
        target_logits,
        action_names,
        outcome_weighted_prefix_skills,
    )
    outcome_weighted_prefix_execute_selective_actions = selective_critical_skill_actions(
        target_logits,
        action_names,
        static_actions,
        outcome_weighted_prefix_skills,
        outcome_weighted_prefix_scores,
        skill_names,
        confidence_threshold=args.outcome_critical_confidence_threshold,
    )
    outcome_weighted_prefix_execute_override_actions = selective_skill_set_actions(
        target_logits,
        action_names,
        static_actions,
        outcome_weighted_prefix_skills,
        {"EXECUTE"},
    )
    outcome_weighted_prefix_execute_confirm_override_actions = selective_skill_set_actions(
        target_logits,
        action_names,
        static_actions,
        outcome_weighted_prefix_skills,
        {"EXECUTE", "CONFIRM"},
    )
    source_prefix_only_binder_coef = fit_skill_binder(
        rows,
        train_selected,
        target_logits,
        target_skill_log_probs,
        source_skills,
        skill_names,
        args.skill_binder_ridge,
        use_action_logits=False,
        use_turn_index=False,
        use_prev_label=False,
        use_prompt_features=True,
    )
    source_prefix_only_binder_skills, _ = decode_skill_binder(
        rows,
        target_logits,
        target_skill_log_probs,
        skill_names,
        source_prefix_only_binder_coef,
        use_action_logits=False,
        use_turn_index=False,
        use_prev_label=False,
        use_prompt_features=True,
    )
    source_prefix_only_binder_actions = constrained_actions_by_skill(
        target_logits,
        action_names,
        source_prefix_only_binder_skills,
    )
    source_action_binder_coef = fit_label_binder(
        rows,
        train_selected,
        target_logits,
        target_skill_log_probs,
        source_actions,
        action_names,
        args.skill_binder_ridge,
    )
    source_action_binder_actions, source_action_binder_scores = decode_label_binder(
        rows,
        target_logits,
        target_skill_log_probs,
        action_names,
        source_action_binder_coef,
    )
    source_action_binder_soft_actions = logits_to_actions(
        target_logits + args.skill_binder_beta * source_action_binder_scores,
        action_names,
    )
    source_random_group_binder_coef = fit_label_binder(
        rows,
        train_selected,
        target_logits,
        target_skill_log_probs,
        source_random_groups,
        random_group_names,
        args.skill_binder_ridge,
    )
    source_random_group_binder_labels, _ = decode_label_binder(
        rows,
        target_logits,
        target_skill_log_probs,
        random_group_names,
        source_random_group_binder_coef,
    )
    source_random_group_binder_actions = constrained_actions_by_label(
        target_logits,
        action_names,
        source_random_group_binder_labels,
        random_group_map,
    )
    oracle_train_binder_coef = fit_skill_binder(
        rows,
        train_selected,
        target_logits,
        target_skill_log_probs,
        expected_skills,
        skill_names,
        args.skill_binder_ridge,
    )
    oracle_train_binder_skills, oracle_train_binder_scores = decode_skill_binder(
        rows,
        target_logits,
        target_skill_log_probs,
        skill_names,
        oracle_train_binder_coef,
    )
    oracle_train_binder_actions = constrained_actions_by_skill(
        target_logits,
        action_names,
        oracle_train_binder_skills,
    )
    oracle_train_binder_soft_actions = soft_actions_from_skill_scores(
        target_logits,
        action_names,
        oracle_train_binder_scores,
        skill_names,
        args.skill_binder_beta,
    )
    source_skill_actions = constrained_actions_by_skill(target_logits, action_names, source_skills)
    oracle_skill_actions = constrained_actions_by_skill(target_logits, action_names, expected_skills)
    oracle_one_skill_actions: dict[str, list[str]] = {}
    for skill in skill_names:
        oracle_one_skill_actions[f"oracle_only_{skill.lower()}"] = [
            oracle_skill_actions[idx] if expected_skills[idx] == skill else static_actions[idx]
            for idx in range(len(rows))
        ]

    methods = {
        "static_action": static_actions,
        "predicted_skill_rerank": predicted_skill_actions,
        "soft_skill_prior": soft_skill_actions,
        "confidence_gated_soft_skill": gated_soft_skill_actions,
        "selective_hard_skill_rerank": selective_hard_actions,
        "temporal_soft_skill_prior": temporal_skill_actions,
        "prefix_defined_oracle_skill": prefix_oracle_actions,
        "prefix_defined_random_oracle": prefix_random_oracle_actions,
        "prefix_latent_skill_binder": prefix_latent_binder_actions,
        "prefix_latent_skill_binder_no_prev": prefix_latent_no_prev_actions,
        "source_discovered_cluster_oracle": source_discovered_oracle_actions,
        "source_discovered_cluster_binder": discovered_latent_binder_actions,
        "response_operator_oracle_allowed": response_operator_oracle_allowed_actions,
        "response_operator_binder_allowed": response_operator_binder_allowed_actions,
        "observation_type_allowed": observation_type_allowed_actions,
        "observation_type_residual": observation_type_residual_actions,
        "response_operator_obs_oracle_allowed": response_operator_obs_oracle_allowed_actions,
        "response_operator_obs_binder_allowed": response_operator_obs_binder_allowed_actions,
        "response_operator_obs_oracle_residual": response_operator_obs_oracle_residual_actions,
        "response_operator_obs_binder_residual": response_operator_obs_binder_residual_actions,
        "write_pair_calibrated_v2": write_pair_calibrated_actions,
        "response_operator_obs_binder_residual_plus_write_pair_v2": (
            response_operator_obs_binder_residual_write_pair_actions
        ),
        "response_operator_oracle_residual": response_operator_oracle_residual_actions,
        "response_operator_binder_residual": response_operator_binder_residual_actions,
        "response_operator_gated_residual": response_operator_gated_residual_actions,
        "response_operator_soft_residual": response_operator_soft_residual_actions,
        "source_distilled_skill_binder": source_binder_actions,
        "source_distilled_soft_skill_binder": source_binder_soft_actions,
        "source_distilled_skill_binder_no_turn": source_binder_no_turn_actions,
        "source_distilled_skill_binder_no_prev": source_binder_no_prev_actions,
        "source_distilled_skill_binder_no_action_logits": source_binder_no_action_logits_actions,
        "source_distilled_prefix_skill_binder": source_prefix_binder_actions,
        "source_distilled_prefix_soft_skill_binder": source_prefix_binder_soft_actions,
        "source_distilled_prefix_skill_binder_no_action_logits": source_prefix_binder_no_action_logits_actions,
        "outcome_weighted_source_skill_binder": outcome_weighted_actions,
        "outcome_weighted_execute_selective": outcome_weighted_execute_selective_actions,
        "outcome_weighted_prefix_skill_binder_no_action_logits": outcome_weighted_prefix_actions,
        "outcome_weighted_prefix_execute_selective": outcome_weighted_prefix_execute_selective_actions,
        "outcome_weighted_prefix_execute_override": outcome_weighted_prefix_execute_override_actions,
        "outcome_weighted_prefix_execute_confirm_override": outcome_weighted_prefix_execute_confirm_override_actions,
        "source_distilled_prefix_only_skill_binder": source_prefix_only_binder_actions,
        "source_distilled_action_binder": source_action_binder_actions,
        "source_distilled_action_soft_binder": source_action_binder_soft_actions,
        "source_distilled_random_group_binder": source_random_group_binder_actions,
        "oracle_train_skill_binder": oracle_train_binder_actions,
        "oracle_train_soft_skill_binder": oracle_train_binder_soft_actions,
        "source_skill_rerank": source_skill_actions,
        "oracle_skill_rerank": oracle_skill_actions,
        **oracle_one_skill_actions,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for method, actions in methods.items():
        artifact = prediction_records(rows, actions, method, {"split": split_metadata})
        save_json(artifact, out_dir / f"{method}_predictions.json")
        results.append(metrics(method, rows, selected, actions, source_actions))

    source_result = metrics("source_policy", rows, selected, source_actions, source_actions)
    skill_counts = defaultdict(int)
    for idx in selected:
        skill_counts[expected_skills[idx]] += 1
    payload = {
        "metadata": {
            "source_tensor": args.source_tensor,
            "target_tensor": args.target_tensor,
            "source_policy": args.source_policy,
            "alignment": args.alignment,
            "split": split_metadata,
            "num_heldout_samples": len(selected),
            "skills": skill_groups(action_names),
            "heldout_skill_counts": dict(sorted(skill_counts.items())),
            "skill_prior_beta": args.skill_prior_beta,
            "skill_confidence_threshold": args.skill_confidence_threshold,
            "skill_confidence_gate_rate": float(skill_conf_gate[selected].float().mean().item()) if selected else 0.0,
            "selective_action_margin_threshold": args.selective_action_margin_threshold,
            "selective_skill_margin_threshold": args.selective_skill_margin_threshold,
            "selective_hard_gate_rate": float(selective_hard_gate[selected].float().mean().item()) if selected else 0.0,
            "temporal_skill_weight": args.temporal_skill_weight,
            "temporal_skill_smoothing": args.temporal_skill_smoothing,
            "skill_binder_ridge": args.skill_binder_ridge,
            "skill_binder_beta": args.skill_binder_beta,
            "prefix_defined_skills": prefix_allowed,
            "prefix_defined_skill_counts": {
                name: sum(1 for idx in selected if prefix_skills[idx] == name)
                for name in prefix_skill_names
            },
            "prefix_defined_oracle_skill": (
                "Skill labels are generated only from prompt/history/observation prefix rules, "
                "not from current ground-truth action. Actions are selected by constraining "
                "target logits to a prefix-skill allowed action set."
            ),
            "prefix_latent_skill_binder": (
                "Action-logit-free ridge binder trained to predict prefix-defined skill labels "
                "from aligned target latent z_T plus parsed prefix/history features."
            ),
            "source_discovered_cluster_oracle": (
                "Source-only discovered skills: K-means over Source policy logits, policy deltas, "
                "and prefix features on train trajectories; cluster action support is estimated "
                "from Source policy actions in train clusters."
            ),
            "source_discovered_cluster_counts": {
                name: sum(1 for idx in selected if discovered_labels[idx] == name)
                for name in discovered_names
            },
            "source_discovered_allowed_actions": {
                name: [action_names[idx] for idx in idxs] for name, idxs in discovered_allowed.items()
            },
            "response_operator_skill": (
                "Observation-Response Skill Operator: source-only clusters over policy-output "
                "response fingerprints F(s)=[logits, logits_delta, observation_signature, prefix_features]. "
                "It asks whether states are equivalent by how the frozen policy responds to observations, "
                "not by hidden-state proximity or primitive action labels."
            ),
            "response_operator_clusters": args.response_operator_clusters,
            "response_residual_alpha": args.response_residual_alpha,
            "write_pair_calibration_margin": args.write_pair_calibration_margin,
            "write_pair_max_calibration": args.write_pair_max_calibration,
            "response_operator_branch_gate_rate": (
                float(response_branch_mask[selected].float().mean().item()) if selected else 0.0
            ),
            "response_operator_cluster_counts": {
                name: sum(1 for idx in selected if response_operator_labels[idx] == name)
                for name in response_operator_names
            },
            "response_operator_allowed_actions": {
                name: [action_names[idx] for idx in idxs] for name, idxs in response_operator_allowed.items()
            },
            "observation_type_counts": {
                name: sum(1 for idx in selected if observation_labels[idx] == name)
                for name in observation_names
            },
            "observation_type_allowed_actions": {
                name: [action_names[idx] for idx in idxs] for name, idxs in observation_allowed.items()
            },
            "observation_type_allowed": (
                "Deployable branch baseline: labels are parsed from the current prompt observation "
                "type only (none/success/failure/empty/conflict/commit), then the action is selected "
                "from Source-train action supports for that observation type."
            ),
            "response_operator_obs": (
                "Branch-conditioned operator: response-operator labels are refined with the current "
                "observation type. This tests whether policy response operators become more useful "
                "when success/failure/empty/conflict/commit are modeled explicitly."
            ),
            "response_operator_obs_allowed_actions": {
                name: [action_names[idx] for idx in idxs] for name, idxs in response_obs_allowed.items()
            },
            "response_operator_oracle_residual": (
                "Diagnostic upper bound using Source response-operator labels for every row; applies "
                "a train-split mean residual in shared action-logit space. No target action labels are used, "
                "but Source labels for held-out rows make it an oracle diagnostic."
            ),
            "response_operator_binder_residual": (
                "Deployable variant: a closed-form ridge binder predicts response-operator labels from "
                "Target policy-output fingerprints, then applies the corresponding residual correction."
            ),
            "write_pair_calibrated_v2": (
                "Deployable observation-conditioned write branch prior. If the current prompt shows "
                "a write tool followed by success/commit, it raises finish over every non-finish branch; "
                "if it shows write failure, it raises recover/verify over finish/execute. No target labels "
                "or future observations are used."
            ),
            "response_operator_obs_binder_residual_plus_write_pair_v2": (
                "The deployable response-operator observation binder residual plus the calibrated "
                "write success/failure branch prior."
            ),
            "source_distilled_skill_binder": (
                "Ridge skill binder trained on train-split target-side policy features using Source "
                "policy predicted skills as pseudo-labels; no target action labels are used."
            ),
            "source_distilled_action_binder": (
                "Ridge label binder trained with the same features and data as the skill binder, "
                "but using Source primitive action pseudo-labels instead of skill labels."
            ),
            "source_distilled_random_group_binder": (
                "Ridge label binder using deterministic random action groups with the same group "
                "sizes as the semantic skill taxonomy; diagnostic for action-space shrinkage."
            ),
            "source_distilled_prefix_skill_binder": (
                "Ridge skill binder trained on Source policy pseudo-skills with features from the "
                "current target policy distribution plus prefix/history/observation cues parsed "
                "from the prompt. Explicit turn index is disabled."
            ),
            "source_distilled_prefix_only_skill_binder": (
                "Same Source pseudo-skill objective, but excludes primitive action logits, turn "
                "index, and previous predicted skill. It tests whether prefix/history cues alone "
                "carry enough policy-skill signal."
            ),
            "outcome_weighted_source_skill_binder": (
                "Outcome-critical Source skill binder. It keeps the label-free Source pseudo-skill "
                "objective, but upweights Source EXECUTE/Commit skill states because single-class "
                "oracle decomposition showed this skill is the only class with large DB Hash impact."
            ),
            "outcome_weighted_execute_selective": (
                "Selective outcome-critical variant. It starts from the static low-rank policy and "
                "only constrains actions when the outcome-weighted binder predicts EXECUTE with "
                "sufficient confidence, reducing unnecessary perturbations on low-impact states."
            ),
            "outcome_weighted_prefix_skill_binder_no_action_logits": (
                "Outcome-weighted prefix/history binder without primitive action logits or turn index. "
                "This tests whether deployable prefix evidence can identify outcome-critical skills "
                "without using a primitive action self-reweighting shortcut."
            ),
            "outcome_weighted_prefix_execute_selective": (
                "Selective version of the outcome-weighted prefix/history binder: only high-confidence "
                "EXECUTE predictions are allowed to override the static low-rank action."
            ),
            "outcome_weighted_prefix_execute_override": (
                "Outcome-critical hybrid. It keeps the static low-rank action everywhere except when "
                "the prefix/history binder predicts EXECUTE; then it chooses the best EXECUTE action. "
                "This directly tests whether the DB gain is concentrated in Commit states."
            ),
            "outcome_weighted_prefix_execute_confirm_override": (
                "Slightly broader outcome-critical hybrid. It only allows EXECUTE or CONFIRM skill "
                "overrides, matching the task-level audit where repairs were concentrated in "
                "Commit/Execute with a smaller Verify/Confirm contribution."
            ),
            "outcome_execute_skill_weight": args.outcome_execute_skill_weight,
            "outcome_critical_confidence_threshold": args.outcome_critical_confidence_threshold,
            "random_action_groups": random_group_map,
            "oracle_train_skill_binder": (
                "Ridge skill binder trained on train-split ground-truth skill labels; this is a "
                "supervised diagnostic for skill binder learnability, not the deployable setting."
            ),
            "oracle_skill_leakage_check": (
                "Oracle skill is derived from the current ground-truth action label only. "
                "It is an upper-bound diagnostic and is not deployable; it does not use future observations."
            ),
        },
        "source": source_result,
        "results": results,
    }
    save_json(payload, out_dir / "skill_policy_results.json")

    def pct(value: float) -> str:
        return f"{100 * value:.2f}%"

    lines = [
        "# Skill Policy Validation",
        "",
        f"- held-out samples: {len(selected)}",
        f"- split: `{split_metadata.get('split_unit')}`",
        "",
        "## Skill Taxonomy",
        "",
        "| Skill | Actions |",
        "|---|---|",
    ]
    for skill, idxs in sorted(skill_groups(action_names).items()):
        lines.append(f"| {skill} | {', '.join(action_names[idx] for idx in idxs)} |")
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Method | Action Acc | Skill Acc | Boundary Skill Acc | Skill-Action Gap | Source Action Agree | Source Skill Agree | Same-Skill Wrong Action | Action Transition | Skill Transition |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| source_policy | {pct(source_result['action_accuracy'])} | "
                f"{pct(source_result['skill_accuracy'])} | {pct(source_result['boundary_skill_accuracy'])} | "
                f"{pct(source_result['skill_minus_action'])} | "
                f"{pct(source_result['source_action_agreement'])} | {pct(source_result['source_skill_agreement'])} | "
                f"{pct(source_result['same_skill_wrong_action_rate'])} | {pct(source_result['action_transition_accuracy'])} | "
                f"{pct(source_result['skill_transition_accuracy'])} |"
            ),
        ]
    )
    for row in results:
        lines.append(
            f"| {row['method']} | {pct(row['action_accuracy'])} | {pct(row['skill_accuracy'])} | "
            f"{pct(row['boundary_skill_accuracy'])} | "
            f"{pct(row['skill_minus_action'])} | {pct(row['source_action_agreement'])} | "
            f"{pct(row['source_skill_agreement'])} | {pct(row['same_skill_wrong_action_rate'])} | "
            f"{pct(row['action_transition_accuracy'])} | {pct(row['skill_transition_accuracy'])} |"
        )
    (out_dir / "skill_policy_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
