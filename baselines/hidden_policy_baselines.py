from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from baselines import eval_protocol
from baselines.tool_policy_utils import prediction_records, save_json
from universal_agent_policy.adapters import AdapterConfig, AdapterPolicy, PolicyHead, build_adapter
from universal_agent_policy.data import load_hidden_tensor
from universal_agent_policy.runtime.tau_grounding import row_context_text


TOOL_ACTIONS = {"THINK", "TRANSFER"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-tensor", required=True)
    parser.add_argument("--target-tensor", required=True)
    parser.add_argument("--source-policy", required=True)
    parser.add_argument("--decision-jsonl")
    parser.add_argument("--ours-target-adapter")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--methods", nargs="+", default=["all"])
    parser.add_argument("--z-dim", type=int, default=256)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--unlock-vector-min-confidence", type=float, default=0.0)
    parser.add_argument("--unlock-vector-bias-scale", type=float, default=0.1)
    parser.add_argument("--unlock-vector-logit-scale", type=float, default=20.0)
    parser.add_argument("--unlock-intervention-alpha", type=float, default=0.25)
    parser.add_argument("--dynamic-transition-weight", type=float, default=0.35)
    parser.add_argument("--dynamic-transition-smoothing", type=float, default=0.1)
    parser.add_argument("--dynamic-margin-threshold", type=float, default=0.12)
    parser.add_argument("--dynamic-prior-confidence-threshold", type=float, default=0.35)
    parser.add_argument("--state-dynamic-min-count", type=int, default=4)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument(
        "--split-unit",
        choices=["decision", "trajectory", "task"],
        default="trajectory",
        help="Held-out protocol. Use trajectory/task to avoid train/val turns from the same tau-bench trajectory/task.",
    )
    parser.add_argument("--split-json", help="Load a precomputed split JSON produced by --write-split-json.")
    parser.add_argument("--write-split-json", help="Write the resolved train/val split for exact reuse.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_rows(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def action_names_from_obj(obj: dict[str, Any], rows: list[dict[str, Any]]) -> list[str] | None:
    names = obj.get("action_names")
    if names:
        return list(names)
    for row in rows:
        row_names = row.get("action_names")
        if row_names:
            return list(row_names)
    return None


def is_tool_action(action_name: str) -> bool:
    return action_name.startswith("ACT_") or action_name in TOOL_ACTIONS


def count_parameters(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


@dataclass
class SourceParts:
    adapter: nn.Module
    head: PolicyHead
    model: AdapterPolicy
    z_dim: int
    num_actions: int
    action_names: list[str] | None


def load_source_parts(path: str, device: torch.device) -> SourceParts:
    checkpoint = torch.load(path, map_location="cpu")
    adapter = build_adapter(AdapterConfig(**checkpoint["adapter_config"]))
    head = PolicyHead(int(checkpoint["z_dim"]), int(checkpoint["num_actions"]))
    adapter.load_state_dict(
        {
            key.removeprefix("adapter."): value
            for key, value in checkpoint["model_state"].items()
            if key.startswith("adapter.")
        }
    )
    head.load_state_dict(
        {
            key.removeprefix("policy_head."): value
            for key, value in checkpoint["model_state"].items()
            if key.startswith("policy_head.")
        }
    )
    model = AdapterPolicy(adapter, head).to(device).eval()
    return SourceParts(
        adapter=model.adapter,
        head=model.policy_head,
        model=model,
        z_dim=int(checkpoint["z_dim"]),
        num_actions=int(checkpoint["num_actions"]),
        action_names=checkpoint.get("action_names"),
    )


def batched_logits(module: nn.Module, h: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    module.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, h.shape[0], batch_size):
            outputs.append(module(h[start : start + batch_size].to(device)).cpu())
    return torch.cat(outputs, dim=0)


def source_latents(adapter: nn.Module, h: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    adapter.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, h.shape[0], batch_size):
            outputs.append(adapter(h[start : start + batch_size].to(device)).cpu())
    return torch.cat(outputs, dim=0)


class SliceProjection(nn.Module):
    def __init__(self, input_dim: int, z_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.input_dim >= self.z_dim:
            z = h[:, : self.z_dim]
        else:
            z = torch.nn.functional.pad(h, (0, self.z_dim - self.input_dim))
        return torch.nn.functional.normalize(z, dim=-1)


class RidgeProjection(nn.Module):
    def __init__(self, coef: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("coef", coef.float())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(h.shape[0], 1, device=h.device, dtype=h.dtype)
        z = torch.cat([h, ones], dim=1) @ self.coef.to(h.device)
        return torch.nn.functional.normalize(z, dim=-1)


class FrozenHeadModel(nn.Module):
    def __init__(self, adapter: nn.Module, head: PolicyHead) -> None:
        super().__init__()
        self.adapter = adapter
        self.head = head
        for param in self.head.parameters():
            param.requires_grad = False

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(self.adapter(h))


class UnlockVectorPolicy(nn.Module):
    """Pure vector capability-direction policy with no trainable target adapter."""

    def __init__(self, center: torch.Tensor, directions: torch.Tensor, bias: torch.Tensor, logit_scale: float) -> None:
        super().__init__()
        self.register_buffer("center", center.float())
        self.register_buffer("directions", directions.float())
        self.register_buffer("bias", bias.float())
        self.logit_scale = float(logit_scale)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        centered = torch.nn.functional.normalize(h - self.center.to(h.device), dim=-1)
        directions = self.directions.to(h.device)
        bias = self.bias.to(h.device)
        return self.logit_scale * (centered @ directions.T) + bias


class UnlockLowRankInterventionPolicy(nn.Module):
    """Low-rank latent alignment followed by optional action-direction intervention."""

    def __init__(
        self,
        coef: torch.Tensor,
        head: PolicyHead,
        center: torch.Tensor,
        directions: torch.Tensor,
        bias: torch.Tensor,
        logit_scale: float,
        intervention_alpha: float,
    ) -> None:
        super().__init__()
        self.projection = RidgeProjection(coef)
        self.head = head
        for param in self.head.parameters():
            param.requires_grad = False
        self.register_buffer("center", center.float())
        self.register_buffer("directions", directions.float())
        self.register_buffer("bias", bias.float())
        self.logit_scale = float(logit_scale)
        self.intervention_alpha = float(intervention_alpha)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.projection(h)
        logits = self.head(z)
        if self.intervention_alpha == 0.0:
            return logits
        centered = torch.nn.functional.normalize(z - self.center.to(z.device), dim=-1)
        direction_logits = self.logit_scale * (centered @ self.directions.to(z.device).T) + self.bias.to(z.device)
        return logits + self.intervention_alpha * direction_logits


class UnlockMappedDirectionPolicy(nn.Module):
    """Paper-faithful direction-transfer baseline: align latent space, then score action directions only."""

    def __init__(
        self,
        coef: torch.Tensor,
        center: torch.Tensor,
        directions: torch.Tensor,
        bias: torch.Tensor,
        logit_scale: float,
    ) -> None:
        super().__init__()
        self.projection = RidgeProjection(coef)
        self.register_buffer("center", center.float())
        self.register_buffer("directions", directions.float())
        self.register_buffer("bias", bias.float())
        self.logit_scale = float(logit_scale)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.projection(h)
        centered = torch.nn.functional.normalize(z - self.center.to(z.device), dim=-1)
        return self.logit_scale * (centered @ self.directions.to(z.device).T) + self.bias.to(z.device)


def metrics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    source_pred: torch.Tensor,
    val_idx: torch.Tensor,
    action_names: list[str] | None,
    rows: list[dict[str, Any]] | None = None,
    num_actions: int | None = None,
) -> dict[str, float]:
    pred = logits.argmax(dim=-1)
    all_idx = torch.arange(labels.shape[0])
    resolved_num_actions = int(num_actions or int(labels.max().item()) + 1)
    metrics = {
        "accuracy_full": eval_protocol.accuracy_on_indices(pred, labels, all_idx),
        "accuracy_val": eval_protocol.accuracy_on_indices(pred, labels, val_idx),
        "heldout_action_accuracy": eval_protocol.accuracy_on_indices(pred, labels, val_idx),
        "macro_action_accuracy_val": eval_protocol.macro_action_accuracy(pred, labels, val_idx, resolved_num_actions),
        "policy_consistency_full": eval_protocol.accuracy_on_indices(pred, source_pred, all_idx),
        "policy_consistency_val": eval_protocol.accuracy_on_indices(pred, source_pred, val_idx),
        "tool_action_accuracy_val": tool_accuracy(pred, labels, val_idx, action_names),
    }
    if rows:
        metrics["transition_state_accuracy_val"] = eval_protocol.transition_state_accuracy(pred, labels, rows, val_idx)
    return metrics


def predictions_from_logits(logits: torch.Tensor, action_names: list[str] | None) -> list[str]:
    pred = logits.argmax(dim=-1).cpu().tolist()
    if not action_names:
        return [str(item) for item in pred]
    return [action_names[int(item)] if int(item) < len(action_names) else "" for item in pred]


def save_predictions_if_requested(
    enabled: bool,
    rows: list[dict[str, Any]],
    logits: torch.Tensor,
    action_names: list[str] | None,
    out_dir: Path,
    method: str,
    metadata: dict[str, Any],
) -> None:
    if not enabled or not rows or not action_names:
        return
    artifact = prediction_records(rows, predictions_from_logits(logits, action_names), method, metadata)
    save_json(artifact, out_dir / f"{method}_predictions.json")


def trajectory_groups(rows: list[dict[str, Any]]) -> list[list[int]]:
    groups: dict[tuple[str, int, int], list[tuple[int, int]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        key = (
            str(row.get("domain", "unknown")),
            int(row.get("task_id", -1)),
            int(row.get("trial", 0)),
        )
        groups[key].append((int(row.get("turn_index", idx)), idx))
    return [[idx for _, idx in sorted(items)] for items in groups.values()]


def fit_source_transition_log_probs(
    rows: list[dict[str, Any]],
    source_pred: torch.Tensor,
    train_idx: torch.Tensor,
    num_actions: int,
    smoothing: float,
) -> torch.Tensor:
    train_set = {int(idx) for idx in train_idx.cpu().tolist()}
    counts = torch.full((num_actions, num_actions), float(smoothing))
    for group in trajectory_groups(rows):
        for left, right in zip(group, group[1:]):
            if left in train_set and right in train_set:
                counts[int(source_pred[left]), int(source_pred[right])] += 1.0
    return torch.log(counts / counts.sum(dim=1, keepdim=True))


def _last_called_tool_kind(text: str) -> str:
    calls = re.findall(r"assistant tool_call:\s*([a-zA-Z0-9_]+)\(", text)
    if not calls:
        return "none"
    name = calls[-1].lower()
    if name.startswith(("find_", "get_", "list_")):
        return "retrieve"
    if name.startswith("search_"):
        return "search"
    if name.startswith(("book_", "cancel_", "exchange_", "modify_", "return_", "send_", "update_")):
        return "update"
    if name == "calculate":
        return "compute"
    if name == "think":
        return "think"
    if "transfer" in name:
        return "transfer"
    return "other"


def _entity_bucket(text: str) -> str:
    if re.search(r"#W\d{7}|order_id|orders", text):
        return "order"
    if re.search(r"\b[A-Z0-9]{6}\b|reservation_id|reservation", text):
        return "reservation"
    if "product_id" in text or "product" in text.lower():
        return "product"
    if re.search(r"\b[A-Z]{3}\b", text) or "airport" in text.lower() or "flight" in text.lower():
        return "flight"
    if re.search(r"\b[a-z]+_[a-z]+_\d{3,}\b", text) or "user_id" in text or "email" in text.lower():
        return "user"
    return "none"


def _status_bucket(text: str) -> str:
    lowered = text.lower()
    tail = lowered[-2500:]
    if any(marker in tail for marker in ["error:", "traceback", "invalid", "exception"]):
        return "tool_error"
    if any(marker in tail for marker in ["not found", "no results", "empty", "does not exist"]):
        return "empty_or_missing"
    if any(marker in tail for marker in ["success", "updated", "cancelled", "canceled", "booked", "modified", "sent"]):
        return "write_success"
    if "assistant tool_call:" in lowered:
        return "tool_observed"
    return "no_tool_observation"


def _intent_bucket(text: str) -> str:
    matches = re.findall(r"(?:^|\n)user:\s*(.*)", text, flags=re.IGNORECASE)
    intent = (matches[-1] if matches else text[-1000:]).lower()
    if any(word in intent for word in ["cancel", "refund"]):
        return "cancel"
    if any(word in intent for word in ["change", "modify", "update", "exchange", "return", "switch"]):
        return "modify"
    if any(word in intent for word in ["book", "buy", "reserve"]):
        return "book"
    if any(word in intent for word in ["find", "search", "look up", "what", "which", "show"]):
        return "retrieve"
    if any(word in intent for word in ["confirm", "yes", "ok", "sure"]):
        return "confirm"
    return "other"


def state_dynamic_bucket(row: dict[str, Any]) -> str:
    text = row_context_text(row)
    domain = str(row.get("domain", "unknown"))
    turn = int(row.get("turn_index") or 0)
    if turn <= 1:
        phase = "early"
    elif turn <= 4:
        phase = "mid"
    else:
        phase = "late"
    return "|".join(
        [
            domain,
            phase,
            _last_called_tool_kind(text),
            _status_bucket(text),
            _entity_bucket(text),
            _intent_bucket(text),
        ]
    )


def fit_state_conditioned_transition_log_probs(
    rows: list[dict[str, Any]],
    source_pred: torch.Tensor,
    train_idx: torch.Tensor,
    num_actions: int,
    smoothing: float,
) -> tuple[dict[tuple[int, str], torch.Tensor], dict[tuple[int, str], int], torch.Tensor]:
    train_set = {int(idx) for idx in train_idx.cpu().tolist()}
    global_log_probs = fit_source_transition_log_probs(rows, source_pred, train_idx, num_actions, smoothing)
    counts: dict[tuple[int, str], torch.Tensor] = {}
    totals: dict[tuple[int, str], int] = {}
    for group in trajectory_groups(rows):
        for left, right in zip(group, group[1:]):
            if left not in train_set or right not in train_set:
                continue
            key = (int(source_pred[left]), state_dynamic_bucket(rows[right]))
            if key not in counts:
                counts[key] = torch.full((num_actions,), float(smoothing))
                totals[key] = 0
            counts[key][int(source_pred[right])] += 1.0
            totals[key] += 1
    log_probs = {key: torch.log(value / value.sum()) for key, value in counts.items()}
    return log_probs, totals, global_log_probs


def apply_dynamic_transition_prior(
    logits: torch.Tensor,
    rows: list[dict[str, Any]],
    transition_log_probs: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    if not rows or weight == 0.0:
        return logits
    adjusted = logits.clone()
    for group in trajectory_groups(rows):
        prev_action: int | None = None
        for idx in group:
            step_logits = logits[idx].clone()
            if prev_action is not None:
                step_logits = step_logits + weight * transition_log_probs[prev_action]
            adjusted[idx] = step_logits
            prev_action = int(step_logits.argmax().item())
    return adjusted


def action_margin(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits.float(), dim=-1)
    if probs.numel() < 2:
        return 1.0
    top2 = torch.topk(probs, k=2).values
    return float((top2[0] - top2[1]).item())


def prior_confidence(log_probs: torch.Tensor) -> float:
    return float(torch.exp(log_probs.float()).max().item())


def apply_selective_dynamic_transition_prior(
    logits: torch.Tensor,
    rows: list[dict[str, Any]],
    transition_log_probs: torch.Tensor,
    weight: float,
    margin_threshold: float,
    prior_confidence_threshold: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not rows or weight == 0.0:
        return logits, {
            "dynamic_gate_open_rate": 0.0,
            "dynamic_gate_candidate_rate": 0.0,
            "dynamic_num_transition_steps": 0.0,
        }
    adjusted = logits.clone()
    num_transition_steps = 0
    num_uncertain = 0
    num_open = 0
    for group in trajectory_groups(rows):
        prev_action: int | None = None
        for idx in group:
            step_logits = adjusted[idx].clone()
            if prev_action is not None:
                num_transition_steps += 1
                margin = action_margin(step_logits)
                conf = prior_confidence(transition_log_probs[prev_action])
                if margin <= margin_threshold:
                    num_uncertain += 1
                if margin <= margin_threshold and conf >= prior_confidence_threshold:
                    step_logits = step_logits + weight * transition_log_probs[prev_action]
                    num_open += 1
            adjusted[idx] = step_logits
            prev_action = int(step_logits.argmax().item())
    denom = max(1, num_transition_steps)
    return adjusted, {
        "dynamic_gate_open_rate": num_open / denom,
        "dynamic_gate_candidate_rate": num_uncertain / denom,
        "dynamic_num_transition_steps": float(num_transition_steps),
    }


def apply_state_conditioned_dynamic_transition_prior(
    logits: torch.Tensor,
    rows: list[dict[str, Any]],
    state_log_probs: dict[tuple[int, str], torch.Tensor],
    state_counts: dict[tuple[int, str], int],
    global_log_probs: torch.Tensor,
    weight: float,
    margin_threshold: float,
    prior_confidence_threshold: float,
    min_count: int,
    use_global_fallback: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not rows or weight == 0.0:
        return logits, {
            "dynamic_gate_open_rate": 0.0,
            "dynamic_gate_candidate_rate": 0.0,
            "state_dynamic_coverage": 0.0,
            "dynamic_num_transition_steps": 0.0,
        }
    adjusted = logits.clone()
    num_transition_steps = 0
    num_candidate = 0
    num_open = 0
    num_state_hit = 0
    num_global = 0
    for group in trajectory_groups(rows):
        prev_action: int | None = None
        for idx in group:
            step_logits = adjusted[idx].clone()
            if prev_action is not None:
                num_transition_steps += 1
                key = (prev_action, state_dynamic_bucket(rows[idx]))
                prior = None
                if state_counts.get(key, 0) >= min_count:
                    prior = state_log_probs[key]
                    num_state_hit += 1
                elif use_global_fallback:
                    prior = global_log_probs[prev_action]
                    num_global += 1
                if prior is not None:
                    margin = action_margin(step_logits)
                    conf = prior_confidence(prior)
                    if margin <= margin_threshold:
                        num_candidate += 1
                    if margin <= margin_threshold and conf >= prior_confidence_threshold:
                        step_logits = step_logits + weight * prior
                        num_open += 1
            adjusted[idx] = step_logits
            prev_action = int(step_logits.argmax().item())
    denom = max(1, num_transition_steps)
    return adjusted, {
        "dynamic_gate_open_rate": num_open / denom,
        "dynamic_gate_candidate_rate": num_candidate / denom,
        "state_dynamic_coverage": num_state_hit / denom,
        "state_dynamic_global_fallback_rate": num_global / denom,
        "dynamic_num_transition_steps": float(num_transition_steps),
    }


def tool_accuracy(pred: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor, action_names: list[str] | None) -> float:
    if not action_names:
        return float("nan")
    idx = idx.cpu()
    mask = torch.tensor([is_tool_action(action_names[int(labels[i])]) for i in idx], dtype=torch.bool)
    if not bool(mask.any()):
        return float("nan")
    tool_idx = idx[mask]
    return float((pred[tool_idx].cpu() == labels[tool_idx].cpu()).float().mean().item())


def fit_ridge_projection(x: torch.Tensor, y: torch.Tensor, ridge: float) -> torch.Tensor:
    ones = torch.ones(x.shape[0], 1)
    x_aug = torch.cat([x, ones], dim=1)
    eye = torch.eye(x_aug.shape[1])
    eye[-1, -1] = 0.0
    return torch.linalg.solve(x_aug.T @ x_aug + ridge * eye, x_aug.T @ y)


def truncate_projection_rank(coef: torch.Tensor, rank: int) -> torch.Tensor:
    linear = coef[:-1]
    bias = coef[-1:]
    max_rank = min(linear.shape)
    kept_rank = max(1, min(int(rank), max_rank))
    u, s, vh = torch.linalg.svd(linear.float(), full_matrices=False)
    low_rank = (u[:, :kept_rank] * s[:kept_rank]) @ vh[:kept_rank]
    return torch.cat([low_rank, bias.float()], dim=0)


def action_directions_from_latents(
    z: torch.Tensor,
    source_logits: torch.Tensor,
    train_idx: torch.Tensor,
    num_actions: int,
    min_confidence: float,
    bias_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    source_probs = torch.softmax(source_logits.float(), dim=-1)
    source_conf, source_pseudo = source_probs.max(dim=-1)
    fit_idx = train_idx.cpu()
    if min_confidence > 0.0:
        confident = fit_idx[source_conf[fit_idx] >= min_confidence]
        if confident.numel() >= max(num_actions, 2):
            fit_idx = confident

    center = z[fit_idx].float().mean(dim=0)
    x_train = torch.nn.functional.normalize(z[fit_idx].float() - center, dim=-1)
    y_train = source_pseudo[fit_idx].long()
    counts = torch.bincount(y_train, minlength=num_actions).float()
    directions = []
    for action_id in range(num_actions):
        pos_mask = y_train == action_id
        neg_mask = ~pos_mask
        if bool(pos_mask.any()) and bool(neg_mask.any()):
            direction = x_train[pos_mask].mean(dim=0) - x_train[neg_mask].mean(dim=0)
        elif bool(pos_mask.any()):
            direction = x_train[pos_mask].mean(dim=0)
        else:
            direction = torch.zeros(z.shape[1])
        directions.append(direction)

    directions_tensor = torch.stack(directions, dim=0)
    directions_tensor = torch.nn.functional.normalize(directions_tensor, dim=-1)
    directions_tensor = torch.nan_to_num(directions_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    priors = (counts + 1.0) / (counts.sum() + float(num_actions))
    bias = bias_scale * torch.log(priors)
    metadata = {
        "fit_samples": int(fit_idx.numel()),
        "source_pseudo_label_counts": [int(value) for value in counts.tolist()],
        "min_confidence": min_confidence,
        "bias_scale": bias_scale,
    }
    return center, directions_tensor, bias, metadata


def fit_unlock_vector_policy(
    target_h: torch.Tensor,
    source_logits: torch.Tensor,
    train_idx: torch.Tensor,
    num_actions: int,
    min_confidence: float,
    bias_scale: float,
    logit_scale: float,
) -> tuple[UnlockVectorPolicy, dict[str, Any]]:
    """Build source-induced one-vs-rest action directions in target activation space.

    This is the strict pure-vector Unlock-style baseline used here: target labels
    and gradient updates are not used. Source policy predictions only choose
    which paired target activations define each capability/action direction.
    """
    center, directions_tensor, bias, metadata = action_directions_from_latents(
        target_h, source_logits, train_idx, num_actions, min_confidence, bias_scale
    )
    policy = UnlockVectorPolicy(center, directions_tensor, bias, logit_scale)
    metadata.update({"logit_scale": logit_scale, "stored_vector_params": int(center.numel() + directions_tensor.numel() + bias.numel())})
    return policy, metadata


def train_mse_adapter(
    target_h: torch.Tensor,
    source_z: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    input_dim: int,
    z_dim: int,
    rank: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[nn.Module, float]:
    adapter = build_adapter(AdapterConfig("low_rank_mlp", input_dim, z_dim, rank)).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(target_h[train_idx], source_z[train_idx]), batch_size=batch_size, shuffle=True)
    best_state = None
    best_val = float("inf")
    for _ in range(epochs):
        adapter.train()
        for h, z in loader:
            pred = adapter(h.to(device))
            loss = torch.nn.functional.mse_loss(pred, z.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        adapter.eval()
        with torch.no_grad():
            val_pred = adapter(target_h[val_idx].to(device)).cpu()
            val_loss = float(torch.nn.functional.mse_loss(val_pred, source_z[val_idx]).item())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu() for key, value in adapter.state_dict().items()}
    adapter.load_state_dict(best_state or adapter.state_dict())
    return adapter.cpu(), best_val


def train_target_local_probe(
    target_h: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    num_actions: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[nn.Module, float]:
    probe = nn.Linear(target_h.shape[1], num_actions).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(target_h[train_idx], labels[train_idx]), batch_size=batch_size, shuffle=True)
    best_state = None
    best_val = -1.0
    for _ in range(epochs):
        probe.train()
        for h, y in loader:
            loss = torch.nn.functional.cross_entropy(probe(h.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        logits = batched_logits(probe, target_h[val_idx], device, batch_size)
        val_acc = float((logits.argmax(dim=-1) == labels[val_idx]).float().mean().item())
        if val_acc > best_val:
            best_val = val_acc
            best_state = {key: value.detach().cpu() for key, value in probe.state_dict().items()}
    probe.load_state_dict(best_state or probe.state_dict())
    return probe.cpu(), best_val


def evaluate_frozen_adapter(
    name: str,
    adapter: nn.Module,
    source: SourceParts,
    target_h: torch.Tensor,
    labels: torch.Tensor,
    source_pred: torch.Tensor,
    val_idx: torch.Tensor,
    action_names: list[str] | None,
    device: torch.device,
    batch_size: int,
    rows: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    model = FrozenHeadModel(adapter, source.head.cpu()).to(device)
    logits = batched_logits(model, target_h, device, batch_size)
    metrics = metrics_from_logits(logits, labels, source_pred, val_idx, action_names, rows, source.num_actions)
    result = {
        "method": name,
        "family": "frozen_source_head",
        "adapter_params": count_parameters(adapter),
        "trained_params": extra.pop("trained_params", count_parameters(adapter)) if extra else count_parameters(adapter),
        "eval_seconds": time.perf_counter() - started,
        **metrics,
    }
    if extra:
        result.update(extra)
    return result


def evaluate_probe(
    probe: nn.Module,
    target_h: torch.Tensor,
    labels: torch.Tensor,
    source_pred: torch.Tensor,
    val_idx: torch.Tensor,
    action_names: list[str] | None,
    device: torch.device,
    batch_size: int,
    rows: list[dict[str, Any]] | None,
    train_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    probe = probe.to(device)
    logits = batched_logits(probe, target_h, device, batch_size)
    return {
        "method": "target_local_linear_probe",
        "family": "target_supervised_probe",
        "adapter_params": count_parameters(probe),
        "trained_params": count_parameters(probe),
        "train_seconds": train_seconds,
        "eval_seconds": time.perf_counter() - started,
        **metrics_from_logits(logits, labels, source_pred, val_idx, action_names, rows),
    }


def evaluate_direct_policy(
    name: str,
    model: nn.Module,
    target_h: torch.Tensor,
    labels: torch.Tensor,
    source_pred: torch.Tensor,
    val_idx: torch.Tensor,
    action_names: list[str] | None,
    device: torch.device,
    batch_size: int,
    rows: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    model = model.to(device).eval()
    logits = batched_logits(model, target_h, device, batch_size)
    result = {
        "method": name,
        "family": "pure_vector_policy",
        "adapter_params": 0,
        "trained_params": 0,
        "eval_seconds": time.perf_counter() - started,
        **metrics_from_logits(logits, labels, source_pred, val_idx, action_names, rows),
    }
    if extra:
        result.update(extra)
    return result


def load_ours_adapter(path: str) -> nn.Module:
    checkpoint = torch.load(path, map_location="cpu")
    adapter = build_adapter(AdapterConfig(**checkpoint["adapter_config"]))
    adapter.load_state_dict(checkpoint["adapter_state"])
    return adapter


def expand_methods(methods: list[str]) -> set[str]:
    selected = set(methods)
    if "all" in selected:
        return {
            "fixed_projection",
            "random_projection",
            "pairwise_linear_ridge",
            "faithful_unlock_direction",
            "unlock_pure_vector",
            "unlock_lowrank_subspace",
            "unlock_lowrank_intervention",
            "unlock_lowrank_dynamic",
            "unlock_lowrank_selective_dynamic",
            "unlock_lowrank_state_dynamic",
            "unlock_lowrank_state_dynamic_fallback",
            "unlock_style_latent_mse",
            "target_local_linear_probe",
            "ours_policy_aware_adapter",
        }
    return selected


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.decision_jsonl)
    source_obj = load_hidden_tensor(args.source_tensor)
    target_obj = load_hidden_tensor(args.target_tensor)
    if source_obj["h"].shape[0] != target_obj["h"].shape[0]:
        raise ValueError("source and target tensors must be paired and ordered identically")
    if rows and len(rows) != target_obj["h"].shape[0]:
        raise ValueError("decision rows and hidden tensor must have the same length")

    source = load_source_parts(args.source_policy, device)
    labels = target_obj["y"].long()
    if args.split_json:
        train_idx, val_idx, split_metadata = eval_protocol.load_split(args.split_json, len(labels))
    else:
        train_idx, val_idx, split_metadata = eval_protocol.split_indices_for_protocol(
            len(labels), args.val_frac, args.seed, rows, args.split_unit
        )
    if args.write_split_json:
        eval_protocol.save_split(args.write_split_json, train_idx, val_idx, split_metadata)
    action_names = action_names_from_obj(target_obj, rows) or source.action_names

    source_logits = batched_logits(source.model, source_obj["h"], device, args.batch_size)
    source_pred = source_logits.argmax(dim=-1)
    source_z = source_latents(source.adapter, source_obj["h"], device, args.batch_size)
    selected = expand_methods(args.methods)
    results: list[dict[str, Any]] = []

    if "fixed_projection" in selected:
        adapter = SliceProjection(target_obj["h"].shape[1], source.z_dim)
        results.append(
            evaluate_frozen_adapter(
                "fixed_slice_projection",
                adapter,
                source,
                target_obj["h"],
                labels,
                source_pred,
                val_idx,
                action_names,
                device,
                args.batch_size,
                rows,
                {"training_signal": "none", "trained_params": 0},
            )
        )

    if "random_projection" in selected:
        torch.manual_seed(args.seed)
        adapter = build_adapter(AdapterConfig("linear", target_obj["h"].shape[1], source.z_dim, args.rank))
        results.append(
            evaluate_frozen_adapter(
                "random_linear_projection",
                adapter,
                source,
                target_obj["h"],
                labels,
                source_pred,
                val_idx,
                action_names,
                device,
                args.batch_size,
                rows,
                {"training_signal": "none", "trained_params": 0},
            )
        )

    if "pairwise_linear_ridge" in selected:
        started = time.perf_counter()
        coef = fit_ridge_projection(target_obj["h"][train_idx], source_z[train_idx], args.ridge)
        adapter = RidgeProjection(coef)
        results.append(
            evaluate_frozen_adapter(
                "pairwise_linear_ridge",
                adapter,
                source,
                target_obj["h"],
                labels,
                source_pred,
                val_idx,
                action_names,
                device,
                args.batch_size,
                rows,
                {
                    "training_signal": "paired_source_latents",
                    "trained_params": int(coef.numel()),
                    "train_seconds": time.perf_counter() - started,
                    "ridge": args.ridge,
                },
            )
        )
        torch.save({"coef": coef, "ridge": args.ridge}, out_dir / "pairwise_linear_ridge.pt")

    if (
        "unlock_lowrank_subspace" in selected
        or "faithful_unlock_direction" in selected
        or "unlock_lowrank_intervention" in selected
        or "unlock_lowrank_dynamic" in selected
        or "unlock_lowrank_selective_dynamic" in selected
        or "unlock_lowrank_state_dynamic" in selected
        or "unlock_lowrank_state_dynamic_fallback" in selected
    ):
        started = time.perf_counter()
        full_coef = fit_ridge_projection(target_obj["h"][train_idx], source_z[train_idx], args.ridge)
        lowrank_coef = truncate_projection_rank(full_coef, args.rank)
        center, directions, bias, direction_meta = action_directions_from_latents(
            source_z,
            source_logits,
            train_idx,
            source.num_actions,
            args.unlock_vector_min_confidence,
            args.unlock_vector_bias_scale,
        )
        fit_seconds = time.perf_counter() - started
        base_extra = {
            "training_signal": "paired_source_latents_lowrank_closed_form",
            "train_seconds": fit_seconds,
            "ridge": args.ridge,
            "rank": args.rank,
            "target_labels_used": False,
            "target_gradient_steps": 0,
            "source_online_query": 0,
            "stored_alignment_params": int(lowrank_coef.numel()),
            "stored_direction_params": int(center.numel() + directions.numel() + bias.numel()),
            **direction_meta,
        }
        if "faithful_unlock_direction" in selected:
            model = UnlockMappedDirectionPolicy(
                lowrank_coef,
                center,
                directions,
                bias,
                args.unlock_vector_logit_scale,
            )
            logits = batched_logits(model.to(device), target_obj["h"], device, args.batch_size)
            direction_extra = {
                **base_extra,
                "training_signal": "source_capability_directions_plus_lowrank_subspace_transfer",
                "intervention_alpha": None,
                "unlock_interpretation": "single_or_few_capability_direction_transfer",
            }
            results.append(
                {
                    "method": "faithful_unlock_direction",
                    "family": "faithful_unlock_direction_transfer",
                    "adapter_params": 0,
                    "trained_params": int(lowrank_coef.numel()),
                    "eval_seconds": 0.0,
                    **metrics_from_logits(logits, labels, source_pred, val_idx, action_names, rows, source.num_actions),
                    **direction_extra,
                }
            )
            save_predictions_if_requested(
                args.save_predictions,
                rows,
                logits,
                action_names,
                out_dir,
                "faithful_unlock_direction",
                direction_extra,
            )
        if "unlock_lowrank_subspace" in selected:
            model = UnlockLowRankInterventionPolicy(
                lowrank_coef,
                source.head.cpu(),
                center,
                directions,
                bias,
                args.unlock_vector_logit_scale,
                intervention_alpha=0.0,
            )
            logits = batched_logits(model.to(device), target_obj["h"], device, args.batch_size)
            results.append(
                {
                    "method": "unlock_lowrank_subspace",
                    "family": "lowrank_subspace_alignment",
                    "adapter_params": 0,
                    "trained_params": int(lowrank_coef.numel()),
                    "eval_seconds": 0.0,
                    **metrics_from_logits(logits, labels, source_pred, val_idx, action_names, rows, source.num_actions),
                    **base_extra,
                    "intervention_alpha": 0.0,
                }
            )
            save_predictions_if_requested(
                args.save_predictions,
                rows,
                logits,
                action_names,
                out_dir,
                "unlock_lowrank_subspace",
                {**base_extra, "intervention_alpha": 0.0},
            )
        if "unlock_lowrank_intervention" in selected:
            model = UnlockLowRankInterventionPolicy(
                lowrank_coef,
                source.head.cpu(),
                center,
                directions,
                bias,
                args.unlock_vector_logit_scale,
                intervention_alpha=args.unlock_intervention_alpha,
            )
            logits = batched_logits(model.to(device), target_obj["h"], device, args.batch_size)
            results.append(
                {
                    "method": "unlock_lowrank_intervention",
                    "family": "lowrank_subspace_alignment",
                    "adapter_params": 0,
                    "trained_params": int(lowrank_coef.numel()),
                    "eval_seconds": 0.0,
                    **metrics_from_logits(logits, labels, source_pred, val_idx, action_names, rows, source.num_actions),
                    **base_extra,
                    "training_signal": "paired_source_latents_lowrank_closed_form_plus_action_direction_intervention",
                    "intervention_alpha": args.unlock_intervention_alpha,
                }
            )
            save_predictions_if_requested(
                args.save_predictions,
                rows,
                logits,
                action_names,
                out_dir,
                "unlock_lowrank_intervention",
                {
                    **base_extra,
                    "training_signal": "paired_source_latents_lowrank_closed_form_plus_action_direction_intervention",
                    "intervention_alpha": args.unlock_intervention_alpha,
                },
            )
        if "unlock_lowrank_dynamic" in selected:
            if not rows:
                raise ValueError("unlock_lowrank_dynamic requires --decision-jsonl for trajectory order")
            model = UnlockLowRankInterventionPolicy(
                lowrank_coef,
                source.head.cpu(),
                center,
                directions,
                bias,
                args.unlock_vector_logit_scale,
                intervention_alpha=0.0,
            )
            static_logits = batched_logits(model.to(device), target_obj["h"], device, args.batch_size)
            transition_log_probs = fit_source_transition_log_probs(
                rows,
                source_pred,
                train_idx,
                source.num_actions,
                args.dynamic_transition_smoothing,
            )
            logits = apply_dynamic_transition_prior(static_logits, rows, transition_log_probs, args.dynamic_transition_weight)
            dynamic_extra = {
                **base_extra,
                "training_signal": "paired_source_latents_lowrank_closed_form_plus_source_transition_prior",
                "dynamic_transition_weight": args.dynamic_transition_weight,
                "dynamic_transition_smoothing": args.dynamic_transition_smoothing,
                "intervention_alpha": 0.0,
            }
            results.append(
                {
                    "method": "unlock_lowrank_dynamic",
                    "family": "lowrank_subspace_alignment_dynamic_policy",
                    "adapter_params": 0,
                    "trained_params": int(lowrank_coef.numel()),
                    "eval_seconds": 0.0,
                    **metrics_from_logits(logits, labels, source_pred, val_idx, action_names, rows, source.num_actions),
                    **dynamic_extra,
                }
            )
            save_predictions_if_requested(
                args.save_predictions,
                rows,
                logits,
                action_names,
                out_dir,
                "unlock_lowrank_dynamic",
                dynamic_extra,
            )
        if "unlock_lowrank_selective_dynamic" in selected:
            if not rows:
                raise ValueError("unlock_lowrank_selective_dynamic requires --decision-jsonl for trajectory order")
            model = UnlockLowRankInterventionPolicy(
                lowrank_coef,
                source.head.cpu(),
                center,
                directions,
                bias,
                args.unlock_vector_logit_scale,
                intervention_alpha=0.0,
            )
            static_logits = batched_logits(model.to(device), target_obj["h"], device, args.batch_size)
            transition_log_probs = fit_source_transition_log_probs(
                rows,
                source_pred,
                train_idx,
                source.num_actions,
                args.dynamic_transition_smoothing,
            )
            logits, gate_stats = apply_selective_dynamic_transition_prior(
                static_logits,
                rows,
                transition_log_probs,
                args.dynamic_transition_weight,
                args.dynamic_margin_threshold,
                args.dynamic_prior_confidence_threshold,
            )
            dynamic_extra = {
                **base_extra,
                "training_signal": "paired_source_latents_lowrank_closed_form_plus_selective_source_transition_prior",
                "dynamic_transition_weight": args.dynamic_transition_weight,
                "dynamic_transition_smoothing": args.dynamic_transition_smoothing,
                "dynamic_margin_threshold": args.dynamic_margin_threshold,
                "dynamic_prior_confidence_threshold": args.dynamic_prior_confidence_threshold,
                "intervention_alpha": 0.0,
                **gate_stats,
            }
            results.append(
                {
                    "method": "unlock_lowrank_selective_dynamic",
                    "family": "lowrank_subspace_alignment_selective_dynamic_policy",
                    "adapter_params": 0,
                    "trained_params": int(lowrank_coef.numel()),
                    "eval_seconds": 0.0,
                    **metrics_from_logits(logits, labels, source_pred, val_idx, action_names, rows, source.num_actions),
                    **dynamic_extra,
                }
            )
            save_predictions_if_requested(
                args.save_predictions,
                rows,
                logits,
                action_names,
                out_dir,
                "unlock_lowrank_selective_dynamic",
                dynamic_extra,
            )
        for method_name, use_global_fallback in (
            ("unlock_lowrank_state_dynamic", False),
            ("unlock_lowrank_state_dynamic_fallback", True),
        ):
            if method_name not in selected:
                continue
            if not rows:
                raise ValueError(f"{method_name} requires --decision-jsonl for trajectory order")
            model = UnlockLowRankInterventionPolicy(
                lowrank_coef,
                source.head.cpu(),
                center,
                directions,
                bias,
                args.unlock_vector_logit_scale,
                intervention_alpha=0.0,
            )
            static_logits = batched_logits(model.to(device), target_obj["h"], device, args.batch_size)
            state_log_probs, state_counts, global_log_probs = fit_state_conditioned_transition_log_probs(
                rows,
                source_pred,
                train_idx,
                source.num_actions,
                args.dynamic_transition_smoothing,
            )
            logits, gate_stats = apply_state_conditioned_dynamic_transition_prior(
                static_logits,
                rows,
                state_log_probs,
                state_counts,
                global_log_probs,
                args.dynamic_transition_weight,
                args.dynamic_margin_threshold,
                args.dynamic_prior_confidence_threshold,
                args.state_dynamic_min_count,
                use_global_fallback,
            )
            dynamic_extra = {
                **base_extra,
                "training_signal": "paired_source_latents_lowrank_closed_form_plus_state_conditioned_transition_prior",
                "dynamic_transition_weight": args.dynamic_transition_weight,
                "dynamic_transition_smoothing": args.dynamic_transition_smoothing,
                "dynamic_margin_threshold": args.dynamic_margin_threshold,
                "dynamic_prior_confidence_threshold": args.dynamic_prior_confidence_threshold,
                "state_dynamic_min_count": args.state_dynamic_min_count,
                "state_dynamic_use_global_fallback": use_global_fallback,
                "state_dynamic_num_buckets": len(state_log_probs),
                "intervention_alpha": 0.0,
                **gate_stats,
            }
            results.append(
                {
                    "method": method_name,
                    "family": "lowrank_subspace_alignment_state_conditioned_dynamic_policy",
                    "adapter_params": 0,
                    "trained_params": int(lowrank_coef.numel()),
                    "eval_seconds": 0.0,
                    **metrics_from_logits(logits, labels, source_pred, val_idx, action_names, rows, source.num_actions),
                    **dynamic_extra,
                }
            )
            save_predictions_if_requested(
                args.save_predictions,
                rows,
                logits,
                action_names,
                out_dir,
                method_name,
                dynamic_extra,
            )
        torch.save(
            {
                "coef": lowrank_coef,
                "ridge": args.ridge,
                "rank": args.rank,
                "center": center,
                "directions": directions,
                "bias": bias,
                "intervention_alpha": args.unlock_intervention_alpha,
                **direction_meta,
            },
            out_dir / "unlock_lowrank_subspace_alignment.pt",
        )

    if "unlock_pure_vector" in selected:
        started = time.perf_counter()
        vector_policy, vector_meta = fit_unlock_vector_policy(
            target_obj["h"],
            source_logits,
            train_idx,
            source.num_actions,
            args.unlock_vector_min_confidence,
            args.unlock_vector_bias_scale,
            args.unlock_vector_logit_scale,
        )
        result = evaluate_direct_policy(
            "unlock_pure_vector",
            vector_policy,
            target_obj["h"],
            labels,
            source_pred,
            val_idx,
            action_names,
            device,
            args.batch_size,
            rows,
            {
                "training_signal": "source_policy_pseudo_labels_contrastive_directions",
                "train_seconds": time.perf_counter() - started,
                "target_labels_used": False,
                "target_gradient_steps": 0,
                "source_online_query": 0,
                **vector_meta,
            },
        )
        results.append(result)
        if args.save_predictions and rows and action_names:
            logits = batched_logits(vector_policy.to(device), target_obj["h"], device, args.batch_size)
            save_predictions_if_requested(
                True,
                rows,
                logits,
                action_names,
                out_dir,
                "unlock_pure_vector",
                {
                    "training_signal": "source_policy_pseudo_labels_contrastive_directions",
                    "target_labels_used": False,
                    "target_gradient_steps": 0,
                    "source_online_query": 0,
                    **vector_meta,
                },
            )
        torch.save(
            {
                "center": vector_policy.center.cpu(),
                "directions": vector_policy.directions.cpu(),
                "bias": vector_policy.bias.cpu(),
                "training_signal": "source_policy_pseudo_labels_contrastive_directions",
                **vector_meta,
            },
            out_dir / "unlock_pure_vector.pt",
        )

    if "unlock_style_latent_mse" in selected:
        started = time.perf_counter()
        adapter, val_mse = train_mse_adapter(
            target_obj["h"],
            source_z,
            train_idx,
            val_idx,
            target_obj["h"].shape[1],
            source.z_dim,
            args.rank,
            args.epochs,
            args.batch_size,
            args.lr,
            args.weight_decay,
            device,
        )
        train_seconds = time.perf_counter() - started
        results.append(
            evaluate_frozen_adapter(
                "unlock_style_latent_mse",
                adapter,
                source,
                target_obj["h"],
                labels,
                source_pred,
                val_idx,
                action_names,
                device,
                args.batch_size,
                rows,
                {
                    "training_signal": "paired_source_latents_mse",
                    "train_seconds": train_seconds,
                    "val_mse": val_mse,
                },
            )
        )
        torch.save(
            {
                "adapter_state": adapter.state_dict(),
                "adapter_config": AdapterConfig("low_rank_mlp", target_obj["h"].shape[1], source.z_dim, args.rank).__dict__,
                "source_policy": str(Path(args.source_policy)),
                "training_signal": "paired_source_latents_mse",
                "val_mse": val_mse,
            },
            out_dir / "unlock_style_latent_mse_adapter.pt",
        )

    if "target_local_linear_probe" in selected:
        started = time.perf_counter()
        probe, best_val = train_target_local_probe(
            target_obj["h"],
            labels,
            train_idx,
            val_idx,
            source.num_actions,
            args.epochs,
            args.batch_size,
            args.lr,
            args.weight_decay,
            device,
        )
        result = evaluate_probe(
            probe,
            target_obj["h"],
            labels,
            source_pred,
            val_idx,
            action_names,
            device,
            args.batch_size,
            rows,
            time.perf_counter() - started,
        )
        result["best_val_accuracy"] = best_val
        result["training_signal"] = "target_labels"
        results.append(result)
        torch.save({"model_state": probe.state_dict(), "num_actions": source.num_actions}, out_dir / "target_local_linear_probe.pt")

    if "ours_policy_aware_adapter" in selected and args.ours_target_adapter:
        adapter = load_ours_adapter(args.ours_target_adapter)
        results.append(
            evaluate_frozen_adapter(
                "ours_policy_aware_adapter",
                adapter,
                source,
                target_obj["h"],
                labels,
                source_pred,
                val_idx,
                action_names,
                device,
                args.batch_size,
                rows,
                {"training_signal": "target_labels_ce_frozen_source_head"},
            )
        )

    source_metrics = metrics_from_logits(source_logits, labels, source_pred, val_idx, action_names, rows, source.num_actions)
    metadata = {
        "num_samples": int(labels.shape[0]),
        "num_actions": source.num_actions,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "split": split_metadata,
        "source_policy_on_source": {
            "accuracy_full": source_metrics["accuracy_full"],
            "accuracy_val": source_metrics["accuracy_val"],
            "heldout_action_accuracy": source_metrics["heldout_action_accuracy"],
            "macro_action_accuracy_val": source_metrics["macro_action_accuracy_val"],
            "transition_state_accuracy_val": source_metrics.get("transition_state_accuracy_val"),
            "tool_action_accuracy_val": source_metrics["tool_action_accuracy_val"],
        },
    }
    payload = {"metadata": metadata, "results": results}
    (out_dir / "policy_baseline_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "policy_baseline_results.md").write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def render_markdown(payload: dict[str, Any]) -> str:
    metadata = payload["metadata"]
    split = metadata.get("split", {})
    lines = [
        "# Hidden Policy Baseline Results",
        "",
        f"- samples: {metadata['num_samples']}",
        f"- actions: {metadata['num_actions']}",
        f"- split: `{split.get('split_unit', 'unknown')}`",
        f"- validation split seed: {metadata['seed']}",
        f"- train samples: {split.get('num_train_samples', 'unknown')}",
        f"- held-out samples: {split.get('num_val_samples', 'unknown')}",
        "",
        "| Method | Held-out Action Acc | Macro Action Acc | Transition Acc | Policy Consistency | Tool Action Acc | Trainable/Fit Params | Training Signal |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(payload["results"], key=lambda item: item.get("heldout_action_accuracy", item.get("accuracy_val", 0.0)), reverse=True):
        lines.append(
            "| {method} | {acc:.2%} | {macro:.2%} | {trans:.2%} | {cons:.2%} | {tool:.2%} | {params} | {signal} |".format(
                method=row["method"],
                acc=row.get("heldout_action_accuracy", row.get("accuracy_val", float("nan"))),
                macro=row.get("macro_action_accuracy_val", float("nan")),
                trans=row.get("transition_state_accuracy_val", float("nan")),
                cons=row.get("policy_consistency_val", float("nan")),
                tool=row.get("tool_action_accuracy_val", float("nan")),
                params=row.get("trained_params", 0),
                signal=row.get("training_signal", ""),
            )
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
