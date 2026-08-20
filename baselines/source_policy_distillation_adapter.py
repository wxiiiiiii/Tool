from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from universal_agent_policy.adapters import AdapterConfig, AdapterPolicy, PolicyHead, build_adapter
from universal_agent_policy.data import load_hidden_tensor, split_indices
from universal_agent_policy.metrics import accuracy


class FrozenHeadAdapter(nn.Module):
    def __init__(self, adapter: nn.Module, head: PolicyHead) -> None:
        super().__init__()
        self.adapter = adapter
        self.head = head
        for param in self.head.parameters():
            param.requires_grad = False

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.adapter(h)
        return z, self.head(z)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-tensor", required=True)
    parser.add_argument("--target-tensor", required=True)
    parser.add_argument("--source-policy", required=True)
    parser.add_argument("--init-adapter")
    parser.add_argument("--decision-jsonl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--adapter", choices=["linear", "low_rank_mlp"], default="low_rank_mlp")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--state-weight", type=float, default=0.0)
    parser.add_argument("--subspace-weight", type=float, default=0.0)
    parser.add_argument("--margin-weight", type=float, default=0.0)
    parser.add_argument("--transition-weight", type=float, default=0.0)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--confidence-gamma", type=float, default=0.0)
    parser.add_argument("--confidence-floor", type=float, default=0.0)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument(
        "--confidence-mode",
        choices=["max_prob", "entropy", "margin"],
        default="max_prob",
        help="Teacher-confidence score used for KD weighting. entropy matches 1 - H(p)/log|A|.",
    )
    parser.add_argument("--transition-confidence-gamma", type=float, default=-1.0)
    parser.add_argument("--transition-confidence-floor", type=float, default=-1.0)
    parser.add_argument("--min-transition-confidence", type=float, default=-1.0)
    parser.add_argument(
        "--transition-weight-mode",
        choices=["uniform", "adaptive"],
        default="uniform",
    )
    parser.add_argument(
        "--transition-impact-preset",
        choices=["tau_hier_v1"],
        default="tau_hier_v1",
    )
    parser.add_argument("--transition-impact-strength", type=float, default=1.0)
    parser.add_argument(
        "--transition-delta-strength",
        type=float,
        default=0.0,
        help="Extra adaptive factor from source centered-logit transition magnitude; 0 disables it.",
    )
    parser.add_argument("--no-normalize-transition-weights", action="store_true")
    parser.add_argument("--sequence-window", type=int, default=2)
    parser.add_argument("--sequence-weight", type=float, default=0.0)
    parser.add_argument(
        "--confidence-normalize-by",
        choices=["none", "predicted_action"],
        default="none",
    )
    parser.add_argument(
        "--teacher-filter",
        choices=["none", "oracle_correct"],
        default="none",
        help="Diagnostic only: oracle_correct uses labels to drop source-teacher mistakes.",
    )
    parser.add_argument("--policy-subspace-rank", type=int, default=-1)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument(
        "--selection-metric",
        choices=["policy_consistency", "transition_agreement", "fidelity_composite"],
        default="policy_consistency",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_source_model(path: str, device: torch.device) -> tuple[AdapterPolicy, dict[str, Any]]:
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
    return AdapterPolicy(adapter, head).to(device).eval(), checkpoint


def source_targets(
    model: AdapterPolicy,
    h: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    zs, logits = [], []
    with torch.no_grad():
        for start in range(0, h.shape[0], batch_size):
            batch = h[start : start + batch_size].to(device)
            z = model.adapter(batch)
            zs.append(z.cpu())
            logits.append(model.policy_head(z).cpu())
    return torch.cat(zs, dim=0), torch.cat(logits, dim=0)


def load_rows(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def subset_rows(rows: list[dict[str, Any]], indices: torch.Tensor) -> list[dict[str, Any]]:
    if not rows:
        return []
    return [rows[int(idx)] for idx in indices]


def build_transition_pairs(rows: list[dict[str, Any]], n: int) -> torch.Tensor:
    if not rows:
        return torch.empty(0, 2, dtype=torch.long)
    if len(rows) != n:
        raise ValueError("decision rows and hidden tensors must have the same number of samples")
    groups: dict[tuple[str, int, int], list[tuple[int, int]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        key = (
            str(row.get("domain", "unknown")),
            int(row.get("task_id", -1)),
            int(row.get("trial", 0)),
        )
        groups[key].append((int(row.get("turn_index", idx)), idx))
    pairs: list[tuple[int, int]] = []
    for items in groups.values():
        ordered = [idx for _, idx in sorted(items)]
        pairs.extend((left, right) for left, right in zip(ordered, ordered[1:]))
    if not pairs:
        return torch.empty(0, 2, dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long)


def build_transition_windows(rows: list[dict[str, Any]], n: int, window: int) -> torch.Tensor:
    if window <= 2:
        return torch.empty(0, max(0, window), dtype=torch.long)
    if not rows:
        return torch.empty(0, window, dtype=torch.long)
    if len(rows) != n:
        raise ValueError("decision rows and hidden tensors must have the same number of samples")
    groups: dict[tuple[str, int, int], list[tuple[int, int]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        key = (
            str(row.get("domain", "unknown")),
            int(row.get("task_id", -1)),
            int(row.get("trial", 0)),
        )
        groups[key].append((int(row.get("turn_index", idx)), idx))
    windows: list[list[int]] = []
    for items in groups.values():
        ordered = [idx for _, idx in sorted(items)]
        if len(ordered) < window:
            continue
        windows.extend(ordered[start : start + window] for start in range(len(ordered) - window + 1))
    if not windows:
        return torch.empty(0, window, dtype=torch.long)
    return torch.tensor(windows, dtype=torch.long)


def pairs_within_indices(pairs: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if pairs.numel() == 0:
        return pairs
    allowed = torch.zeros(int(pairs.max().item()) + 1, dtype=torch.bool)
    allowed[indices.cpu()] = True
    mask = allowed[pairs[:, 0]] & allowed[pairs[:, 1]]
    return pairs[mask]


def windows_within_indices(windows: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if windows.numel() == 0:
        return windows
    allowed = torch.zeros(int(windows.max().item()) + 1, dtype=torch.bool)
    allowed[indices.cpu()] = True
    mask = allowed[windows].all(dim=1)
    return windows[mask]


def teacher_confidence(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits, dim=-1)
    top2 = probs.topk(k=min(2, probs.shape[-1]), dim=-1).values
    confidence = top2[:, 0]
    if top2.shape[-1] == 1:
        margin = top2[:, 0]
    else:
        margin = top2[:, 0] - top2[:, 1]
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    entropy_confidence = 1.0 - entropy / max(1e-6, torch.log(torch.tensor(float(logits.shape[-1]))))
    return confidence, margin, entropy_confidence.clamp(min=0.0, max=1.0)


def confidence_weights(
    logits: torch.Tensor,
    gamma: float,
    floor: float,
    min_confidence: float,
    normalize_by: str = "none",
    confidence_mode: str = "max_prob",
) -> torch.Tensor:
    if gamma <= 0 and floor <= 0 and min_confidence <= 0:
        return torch.ones(logits.shape[0])
    max_prob, margin, entropy_confidence = teacher_confidence(logits)
    if confidence_mode == "entropy":
        confidence = entropy_confidence
    elif confidence_mode == "margin":
        confidence = margin
    else:
        confidence = max_prob
    if normalize_by == "predicted_action":
        pred = logits.argmax(dim=-1)
        normalized = torch.ones_like(confidence)
        for action in pred.unique():
            mask = pred == action
            values = confidence[mask]
            if values.numel() <= 1:
                normalized[mask] = 1.0
                continue
            low = values.min()
            high = values.max()
            normalized[mask] = (values - low) / (high - low).clamp_min(1e-6)
    else:
        num_actions = logits.shape[-1]
        random_confidence = 1.0 / max(1, num_actions)
        normalized = (confidence - random_confidence) / max(1e-6, 1.0 - random_confidence)
    weights = normalized.clamp(min=0.0, max=1.0)
    if gamma > 0:
        weights = weights.pow(gamma)
    if floor > 0:
        weights = torch.maximum(weights, torch.full_like(weights, floor))
    if min_confidence > 0:
        weights = torch.where(confidence >= min_confidence, weights, torch.zeros_like(weights))
    return weights


def transition_confidence_weights(
    logits: torch.Tensor,
    pairs: torch.Tensor,
    gamma: float,
    floor: float,
    min_confidence: float,
    normalize_by: str = "none",
    confidence_mode: str = "max_prob",
) -> torch.Tensor:
    if pairs.numel() == 0:
        return torch.empty(0)
    endpoint_weights = confidence_weights(logits, gamma, floor, min_confidence, normalize_by, confidence_mode)
    left = endpoint_weights[pairs[:, 0]]
    right = endpoint_weights[pairs[:, 1]]
    return torch.sqrt((left * right).clamp_min(0.0))


def transition_delta_weights(
    logits: torch.Tensor,
    pairs: torch.Tensor,
    strength: float,
    normalize: bool,
) -> torch.Tensor:
    if pairs.numel() == 0:
        return torch.empty(0)
    if strength <= 0:
        return torch.ones(pairs.shape[0])
    left = logits[pairs[:, 0]]
    right = logits[pairs[:, 1]]
    source_delta = right - left
    source_delta = source_delta - source_delta.mean(dim=-1, keepdim=True)
    magnitude = source_delta.norm(dim=-1)
    if magnitude.numel() <= 1:
        scaled = torch.ones_like(magnitude)
    else:
        low = magnitude.quantile(0.1)
        high = magnitude.quantile(0.9)
        scaled = (magnitude - low) / (high - low).clamp_min(1e-6)
        scaled = scaled.clamp(min=0.0, max=1.0)
    out = 1.0 + strength * scaled
    if normalize and out.numel():
        out = out / out.mean().clamp_min(1e-6)
    return out.float()


def action_names_from_rows(rows: list[dict[str, Any]]) -> list[str] | None:
    for row in rows:
        names = row.get("action_names")
        if names:
            return list(names)
    return None


def tau_hier_action_impact(name: str) -> float:
    if name in {"ANSWER", "TRANSFER"}:
        return 1.35
    if name == "ASK_USER":
        return 1.25
    if name == "VERIFY":
        return 1.15
    if name == "THINK":
        return 0.7
    if name in {"ACT_RETRIEVE_USER", "ACT_RETRIEVE_ORDER", "ACT_RETRIEVE_RESERVATION"}:
        return 0.95
    if name in {"ACT_RETRIEVE_PRODUCT", "ACT_RETRIEVE_CATALOG", "ACT_SEARCH_FLIGHT"}:
        return 0.85
    if name.startswith("ACT_UPDATE_") or name in {
        "ACT_UPDATE_ORDER_CANCEL",
        "ACT_UPDATE_RESERVATION_CANCEL",
        "ACT_UPDATE_RESERVATION_BOOK",
        "ACT_SEND_CERTIFICATE",
    }:
        return 1.6
    if name.startswith("ACT_"):
        return 1.1
    return 1.0


def transition_type_weight(left_name: str, right_name: str, strength: float) -> float:
    base = max(tau_hier_action_impact(left_name), tau_hier_action_impact(right_name))
    if left_name == "ASK_USER" or right_name.startswith("ACT_"):
        base = max(base, 1.25)
    if left_name.startswith("ACT_") and right_name in {"ANSWER", "TRANSFER"}:
        base = max(base, 1.45)
    if left_name.startswith("ACT_UPDATE_") or right_name.startswith("ACT_UPDATE_"):
        base = max(base, 1.6)
    if left_name == "THINK" and right_name == "THINK":
        base = min(base, 0.7)
    return 1.0 + strength * (base - 1.0)


def adaptive_transition_weights(
    logits: torch.Tensor,
    pairs: torch.Tensor,
    action_names: list[str] | None,
    strength: float,
    normalize: bool,
) -> torch.Tensor:
    if pairs.numel() == 0:
        return torch.empty(0)
    if not action_names:
        return torch.ones(pairs.shape[0])
    pred = logits.argmax(dim=-1)
    weights = []
    for left, right in pairs.tolist():
        left_id = int(pred[left].item())
        right_id = int(pred[right].item())
        left_name = action_names[left_id] if 0 <= left_id < len(action_names) else str(left_id)
        right_name = action_names[right_id] if 0 <= right_id < len(action_names) else str(right_id)
        weights.append(transition_type_weight(left_name, right_name, strength))
    out = torch.tensor(weights, dtype=torch.float32)
    if normalize and out.numel():
        out = out / out.mean().clamp_min(1e-6)
    return out


def policy_subspace_basis(head: PolicyHead, rank: int) -> torch.Tensor:
    weight = head.linear.weight.detach().float().cpu()
    _, singular_values, vh = torch.linalg.svd(weight, full_matrices=False)
    nonzero = int((singular_values > 1e-6).sum().item())
    if rank <= 0:
        rank = nonzero
    rank = max(1, min(rank, vh.shape[0]))
    return vh[:rank].T.contiguous()


def project_policy_subspace(z: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return z @ basis.to(z.device, dtype=z.dtype)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def evaluate(
    model: FrozenHeadAdapter,
    target_h: torch.Tensor,
    labels: torch.Tensor,
    source_logits: torch.Tensor,
    indices: torch.Tensor,
    transition_pairs: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    logits_all = []
    model.eval()
    with torch.no_grad():
        for start in range(0, target_h.shape[0], batch_size):
            _, logits = model(target_h[start : start + batch_size].to(device))
            logits_all.append(logits.cpu())
    logits = torch.cat(logits_all, dim=0)
    pred = logits.argmax(dim=-1)
    source_pred = source_logits.argmax(dim=-1)
    idx = indices.cpu()
    teacher_correct = source_pred[idx] == labels[idx]
    retained = pred[idx] == source_pred[idx]
    correct_teacher_mask = teacher_correct
    incorrect_teacher_mask = ~teacher_correct
    metrics = {
        "action_accuracy": accuracy(logits[idx], labels[idx]),
        "teacher_accuracy": float(teacher_correct.float().mean().item()),
        "policy_consistency": float((pred[idx] == source_pred[idx]).float().mean().item()),
        "correct_teacher_retention": float(retained[correct_teacher_mask].float().mean().item())
        if bool(correct_teacher_mask.any())
        else 0.0,
        "incorrect_teacher_agreement": float(retained[incorrect_teacher_mask].float().mean().item())
        if bool(incorrect_teacher_mask.any())
        else 0.0,
        "kl_to_source": float(
            nn.functional.kl_div(
                nn.functional.log_softmax(logits[idx], dim=-1),
                nn.functional.softmax(source_logits[idx], dim=-1),
                reduction="batchmean",
            ).item()
        ),
    }
    eval_pairs = pairs_within_indices(transition_pairs, idx) if transition_pairs.numel() else transition_pairs
    if eval_pairs.numel():
        left = eval_pairs[:, 0]
        right = eval_pairs[:, 1]
        source_edge = (source_pred[left], source_pred[right])
        target_edge = (pred[left], pred[right])
        edge_agreement = (target_edge[0] == source_edge[0]) & (target_edge[1] == source_edge[1])
        source_delta = source_logits[right] - source_logits[left]
        target_delta = logits[right] - logits[left]
        source_delta = source_delta - source_delta.mean(dim=-1, keepdim=True)
        target_delta = target_delta - target_delta.mean(dim=-1, keepdim=True)
        source_both_correct = source_pred[left].eq(labels[left]) & source_pred[right].eq(labels[right])
        metrics.update(
            {
                "transition_pairs": int(eval_pairs.shape[0]),
                "transition_agreement": float(edge_agreement.float().mean().item()),
                "correct_teacher_transition_retention": float(edge_agreement[source_both_correct].float().mean().item())
                if bool(source_both_correct.any())
                else 0.0,
                "incorrect_teacher_transition_agreement": float(edge_agreement[~source_both_correct].float().mean().item())
                if bool((~source_both_correct).any())
                else 0.0,
                "transition_delta_mse": float((target_delta - source_delta).pow(2).mean().item()),
            }
        )
    else:
        metrics.update(
            {
                "transition_pairs": 0,
                "transition_agreement": 0.0,
                "correct_teacher_transition_retention": 0.0,
                "incorrect_teacher_transition_agreement": 0.0,
                "transition_delta_mse": 0.0,
            }
        )
    return metrics


def distillation_loss(
    z_target: torch.Tensor,
    logits_target: torch.Tensor,
    z_source: torch.Tensor,
    logits_source: torch.Tensor,
    weights: torch.Tensor,
    subspace_basis: torch.Tensor | None,
    temperature: float,
    state_weight: float,
    subspace_weight: float,
    margin_weight: float,
    kl_weight: float,
) -> torch.Tensor:
    t = max(1e-6, temperature)
    per_sample_kl = nn.functional.kl_div(
        nn.functional.log_softmax(logits_target / t, dim=-1),
        nn.functional.softmax(logits_source / t, dim=-1),
        reduction="none",
    ).sum(dim=-1) * (t * t)
    loss = kl_weight * weighted_mean(per_sample_kl, weights)
    if state_weight > 0:
        state = (z_target - z_source).pow(2).mean(dim=-1)
        loss = loss + state_weight * weighted_mean(state, weights)
    if subspace_weight > 0:
        if subspace_basis is None:
            raise ValueError("subspace_weight requires a policy subspace basis")
        zt = project_policy_subspace(z_target, subspace_basis)
        zs = project_policy_subspace(z_source, subspace_basis)
        subspace = (zt - zs).pow(2).mean(dim=-1)
        loss = loss + subspace_weight * weighted_mean(subspace, weights)
    if margin_weight > 0:
        centered_target = logits_target - logits_target.mean(dim=-1, keepdim=True)
        centered_source = logits_source - logits_source.mean(dim=-1, keepdim=True)
        margin = (centered_target - centered_source).pow(2).mean(dim=-1)
        loss = loss + margin_weight * weighted_mean(margin, weights)
    return loss


def transition_loss(
    logits_left: torch.Tensor,
    logits_right: torch.Tensor,
    source_left: torch.Tensor,
    source_right: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    target_delta = logits_right - logits_left
    source_delta = source_right - source_left
    target_delta = target_delta - target_delta.mean(dim=-1, keepdim=True)
    source_delta = source_delta - source_delta.mean(dim=-1, keepdim=True)
    per_pair = (target_delta - source_delta).pow(2).mean(dim=-1)
    return weighted_mean(per_pair, weights)


def sequence_loss(
    logits_seq: torch.Tensor,
    source_seq: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    target_delta = logits_seq[:, 1:] - logits_seq[:, :-1]
    source_delta = source_seq[:, 1:] - source_seq[:, :-1]
    target_delta = target_delta - target_delta.mean(dim=-1, keepdim=True)
    source_delta = source_delta - source_delta.mean(dim=-1, keepdim=True)
    per_window = (target_delta - source_delta).pow(2).mean(dim=(1, 2))
    return weighted_mean(per_window, weights)


def subset_object(obj: dict[str, Any], indices: torch.Tensor) -> dict[str, Any]:
    out = dict(obj)
    out["h"] = obj["h"][indices]
    out["y"] = obj["y"][indices]
    if obj.get("sample_id") is not None:
        sample_ids = list(obj["sample_id"])
        out["sample_id"] = [sample_ids[int(idx)] for idx in indices]
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    source_obj = load_hidden_tensor(args.source_tensor)
    target_obj = load_hidden_tensor(args.target_tensor)
    rows = load_rows(args.decision_jsonl)
    if source_obj["h"].shape[0] != target_obj["h"].shape[0]:
        raise ValueError("source and target tensors must be paired and ordered identically")
    if rows and len(rows) != source_obj["h"].shape[0]:
        raise ValueError("decision rows and hidden tensors must have the same number of samples")
    if args.max_samples > 0 and args.max_samples < source_obj["h"].shape[0]:
        generator = torch.Generator().manual_seed(args.seed)
        subset = torch.randperm(source_obj["h"].shape[0], generator=generator)[: args.max_samples].sort().values
        source_obj = subset_object(source_obj, subset)
        target_obj = subset_object(target_obj, subset)
        rows = subset_rows(rows, subset)

    source_model, source_checkpoint = load_source_model(args.source_policy, device)
    source_z, source_logits = source_targets(source_model, source_obj["h"], device, args.batch_size)
    labels = target_obj["y"].long()
    source_pred = source_logits.argmax(dim=-1)
    teacher_correct = source_pred.eq(labels)
    train_idx, val_idx = split_indices(len(labels), args.val_frac, args.seed)
    transition_pairs = build_transition_pairs(rows, len(labels))
    train_transition_pairs = pairs_within_indices(transition_pairs, train_idx)
    transition_windows = build_transition_windows(rows, len(labels), args.sequence_window)
    train_transition_windows = windows_within_indices(transition_windows, train_idx)
    weights = confidence_weights(
        source_logits,
        args.confidence_gamma,
        args.confidence_floor,
        args.min_confidence,
        args.confidence_normalize_by,
        args.confidence_mode,
    )
    if args.teacher_filter == "oracle_correct":
        weights = weights * teacher_correct.float()
    transition_confidence_gamma = (
        args.confidence_gamma
        if args.transition_confidence_gamma < 0
        else args.transition_confidence_gamma
    )
    transition_confidence_floor = (
        args.confidence_floor
        if args.transition_confidence_floor < 0
        else args.transition_confidence_floor
    )
    min_transition_confidence = (
        args.min_confidence
        if args.min_transition_confidence < 0
        else args.min_transition_confidence
    )
    train_transition_weights = transition_confidence_weights(
        source_logits,
        train_transition_pairs,
        transition_confidence_gamma,
        transition_confidence_floor,
        min_transition_confidence,
        args.confidence_normalize_by,
        args.confidence_mode,
    )
    action_names = action_names_from_rows(rows) or source_checkpoint.get("action_names")
    if args.transition_weight_mode == "adaptive" and train_transition_pairs.numel():
        impact_weights = adaptive_transition_weights(
            source_logits,
            train_transition_pairs,
            action_names,
            args.transition_impact_strength,
            not args.no_normalize_transition_weights,
        )
        train_transition_weights = train_transition_weights * impact_weights
    if args.transition_delta_strength > 0 and train_transition_pairs.numel():
        delta_weights = transition_delta_weights(
            source_logits,
            train_transition_pairs,
            args.transition_delta_strength,
            not args.no_normalize_transition_weights,
        )
        train_transition_weights = train_transition_weights * delta_weights
    train_sequence_weights = transition_confidence_weights(
        source_logits,
        train_transition_windows[:, [0, -1]] if train_transition_windows.numel() else torch.empty(0, 2, dtype=torch.long),
        transition_confidence_gamma,
        transition_confidence_floor,
        min_transition_confidence,
        args.confidence_normalize_by,
        args.confidence_mode,
    )
    if args.transition_weight_mode == "adaptive" and train_transition_windows.numel():
        sequence_pairs = train_transition_windows[:, [0, -1]]
        sequence_impact = adaptive_transition_weights(
            source_logits,
            sequence_pairs,
            action_names,
            args.transition_impact_strength,
            not args.no_normalize_transition_weights,
        )
        train_sequence_weights = train_sequence_weights * sequence_impact
    if args.transition_delta_strength > 0 and train_transition_windows.numel():
        sequence_delta = transition_delta_weights(
            source_logits,
            train_transition_windows[:, [0, -1]],
            args.transition_delta_strength,
            not args.no_normalize_transition_weights,
        )
        train_sequence_weights = train_sequence_weights * sequence_delta
    if args.teacher_filter == "oracle_correct" and train_transition_pairs.numel():
        edge_correct = teacher_correct[train_transition_pairs[:, 0]] & teacher_correct[train_transition_pairs[:, 1]]
        train_transition_weights = train_transition_weights * edge_correct.float()
    if args.teacher_filter == "oracle_correct" and train_transition_windows.numel():
        window_correct = teacher_correct[train_transition_windows].all(dim=1)
        train_sequence_weights = train_sequence_weights * window_correct.float()

    adapter_config = AdapterConfig(
        args.adapter,
        int(target_obj["h"].shape[1]),
        int(source_checkpoint["z_dim"]),
        args.rank,
    )
    head = PolicyHead(int(source_checkpoint["z_dim"]), int(source_checkpoint["num_actions"]))
    head.load_state_dict(source_model.policy_head.cpu().state_dict())
    subspace_basis = (
        policy_subspace_basis(head, args.policy_subspace_rank)
        if args.subspace_weight > 0
        else None
    )
    model = FrozenHeadAdapter(build_adapter(adapter_config), head).to(device)
    if args.init_adapter:
        init_checkpoint = torch.load(args.init_adapter, map_location="cpu")
        init_config = init_checkpoint.get("adapter_config", {})
        expected_config = adapter_config.__dict__
        if init_config and init_config != expected_config:
            raise ValueError(f"init adapter config mismatch: {init_config} != {expected_config}")
        model.adapter.load_state_dict(init_checkpoint["adapter_state"])
    opt = torch.optim.AdamW(model.adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loader = DataLoader(
        TensorDataset(target_obj["h"][train_idx], source_z[train_idx], source_logits[train_idx], weights[train_idx]),
        batch_size=args.batch_size,
        shuffle=True,
    )
    transition_loader = None
    if args.transition_weight > 0:
        if train_transition_pairs.numel() == 0:
            raise ValueError("transition_weight requires transition pairs from --decision-jsonl")
        transition_loader = DataLoader(
            TensorDataset(
                target_obj["h"][train_transition_pairs[:, 0]],
                target_obj["h"][train_transition_pairs[:, 1]],
                source_logits[train_transition_pairs[:, 0]],
                source_logits[train_transition_pairs[:, 1]],
                train_transition_weights
                if train_transition_weights.numel()
                else torch.ones(train_transition_pairs.shape[0]),
            ),
            batch_size=args.batch_size,
            shuffle=True,
        )
    sequence_loader = None
    if args.sequence_weight > 0:
        if args.sequence_window <= 2:
            raise ValueError("sequence_weight requires --sequence-window > 2")
        if train_transition_windows.numel() == 0:
            raise ValueError("sequence_weight requires sequence windows from --decision-jsonl")
        sequence_loader = DataLoader(
            TensorDataset(
                target_obj["h"][train_transition_windows],
                source_logits[train_transition_windows],
                train_sequence_weights if train_sequence_weights.numel() else torch.ones(train_transition_windows.shape[0]),
            ),
            batch_size=max(1, args.batch_size // max(1, args.sequence_window)),
            shuffle=True,
        )
    best_state = None
    best_score = -1.0
    best_metrics: dict[str, float] = {}
    metrics = evaluate(model, target_obj["h"], labels, source_logits, val_idx, transition_pairs, device, args.batch_size)
    if args.selection_metric == "fidelity_composite":
        best_score = 0.5 * metrics["policy_consistency"] + 0.5 * metrics["transition_agreement"]
    else:
        best_score = metrics[args.selection_metric]
    best_metrics = metrics
    best_state = {key: value.detach().cpu() for key, value in model.adapter.state_dict().items()}
    for epoch in range(args.epochs):
        model.train()
        for h_t, z_s, logits_s, weight in loader:
            z_t, logits_t = model(h_t.to(device))
            loss = distillation_loss(
                z_t,
                logits_t,
                z_s.to(device),
                logits_s.to(device),
                weight.to(device),
                subspace_basis,
                args.temperature,
                args.state_weight,
                args.subspace_weight,
                args.margin_weight,
                args.kl_weight,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        if transition_loader is not None:
            model.train()
            for h_left, h_right, source_left, source_right, edge_weight in transition_loader:
                _, logits_left = model(h_left.to(device))
                _, logits_right = model(h_right.to(device))
                loss = args.transition_weight * transition_loss(
                    logits_left,
                    logits_right,
                    source_left.to(device),
                    source_right.to(device),
                    edge_weight.to(device),
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
        if sequence_loader is not None:
            model.train()
            for h_seq, source_seq, window_weight in sequence_loader:
                batch, window, hidden = h_seq.shape
                _, logits_flat = model(h_seq.reshape(batch * window, hidden).to(device))
                logits_seq = logits_flat.reshape(batch, window, -1)
                loss = args.sequence_weight * sequence_loss(
                    logits_seq,
                    source_seq.to(device),
                    window_weight.to(device),
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
        if args.eval_every > 1 and epoch != args.epochs - 1 and (epoch + 1) % args.eval_every != 0:
            continue
        metrics = evaluate(model, target_obj["h"], labels, source_logits, val_idx, transition_pairs, device, args.batch_size)
        if args.selection_metric == "fidelity_composite":
            score = 0.5 * metrics["policy_consistency"] + 0.5 * metrics["transition_agreement"]
        else:
            score = metrics[args.selection_metric]
        if score > best_score:
            best_score = score
            best_metrics = metrics
            best_state = {key: value.detach().cpu() for key, value in model.adapter.state_dict().items()}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter_state": best_state,
            "adapter_config": adapter_config.__dict__,
            "source_policy": str(Path(args.source_policy)),
            "init_adapter": args.init_adapter,
            "training_signal": "source_policy_distillation",
            "temperature": args.temperature,
            "state_weight": args.state_weight,
            "subspace_weight": args.subspace_weight,
            "margin_weight": args.margin_weight,
            "transition_weight": args.transition_weight,
            "transition_weight_mode": args.transition_weight_mode,
            "transition_impact_preset": args.transition_impact_preset,
            "transition_impact_strength": args.transition_impact_strength,
            "transition_delta_strength": args.transition_delta_strength,
            "normalize_transition_weights": not args.no_normalize_transition_weights,
            "sequence_window": args.sequence_window,
            "sequence_weight": args.sequence_weight,
            "kl_weight": args.kl_weight,
            "confidence_gamma": args.confidence_gamma,
            "confidence_floor": args.confidence_floor,
            "min_confidence": args.min_confidence,
            "confidence_mode": args.confidence_mode,
            "transition_confidence_gamma": transition_confidence_gamma,
            "transition_confidence_floor": transition_confidence_floor,
            "min_transition_confidence": min_transition_confidence,
            "confidence_normalize_by": args.confidence_normalize_by,
            "teacher_filter": args.teacher_filter,
            "policy_subspace_rank": None if subspace_basis is None else subspace_basis.shape[1],
            "max_samples": args.max_samples,
            "selection_metric": args.selection_metric,
            "num_transition_pairs": int(transition_pairs.shape[0]),
            "num_train_transition_pairs": int(train_transition_pairs.shape[0]),
            "num_transition_windows": int(transition_windows.shape[0]),
            "num_train_transition_windows": int(train_transition_windows.shape[0]),
            "mean_train_weight": float(weights[train_idx].mean().item()),
            "kept_train_fraction": float((weights[train_idx] > 0).float().mean().item()),
            "mean_train_transition_weight": float(train_transition_weights.mean().item())
            if train_transition_weights.numel()
            else 0.0,
            "kept_train_transition_fraction": float((train_transition_weights > 0).float().mean().item())
            if train_transition_weights.numel()
            else 0.0,
            "mean_train_sequence_weight": float(train_sequence_weights.mean().item())
            if train_sequence_weights.numel()
            else 0.0,
            "kept_train_sequence_fraction": float((train_sequence_weights > 0).float().mean().item())
            if train_sequence_weights.numel()
            else 0.0,
            "best_val_metrics": best_metrics,
            "uses_target_labels_for_training": args.teacher_filter != "none",
        },
        out_dir / "target_adapter.pt",
    )
    print(
        json.dumps(
            {
                "best_val_metrics": best_metrics,
                "uses_target_labels_for_training": args.teacher_filter != "none",
                "init_adapter": args.init_adapter,
                "state_weight": args.state_weight,
                "subspace_weight": args.subspace_weight,
                "margin_weight": args.margin_weight,
                "transition_weight": args.transition_weight,
                "transition_weight_mode": args.transition_weight_mode,
                "transition_impact_preset": args.transition_impact_preset,
                "transition_impact_strength": args.transition_impact_strength,
                "transition_delta_strength": args.transition_delta_strength,
                "normalize_transition_weights": not args.no_normalize_transition_weights,
                "sequence_window": args.sequence_window,
                "sequence_weight": args.sequence_weight,
                "kl_weight": args.kl_weight,
                "temperature": args.temperature,
                "confidence_gamma": args.confidence_gamma,
                "confidence_floor": args.confidence_floor,
                "min_confidence": args.min_confidence,
                "confidence_mode": args.confidence_mode,
                "transition_confidence_gamma": transition_confidence_gamma,
                "transition_confidence_floor": transition_confidence_floor,
                "min_transition_confidence": min_transition_confidence,
                "confidence_normalize_by": args.confidence_normalize_by,
                "teacher_filter": args.teacher_filter,
                "policy_subspace_rank": None if subspace_basis is None else subspace_basis.shape[1],
                "max_samples": args.max_samples,
                "selection_metric": args.selection_metric,
                "num_transition_pairs": int(transition_pairs.shape[0]),
                "num_train_transition_pairs": int(train_transition_pairs.shape[0]),
                "num_transition_windows": int(transition_windows.shape[0]),
                "num_train_transition_windows": int(train_transition_windows.shape[0]),
                "mean_train_weight": float(weights[train_idx].mean().item()),
                "kept_train_fraction": float((weights[train_idx] > 0).float().mean().item()),
                "mean_train_transition_weight": float(train_transition_weights.mean().item())
                if train_transition_weights.numel()
                else 0.0,
                "kept_train_transition_fraction": float((train_transition_weights > 0).float().mean().item())
                if train_transition_weights.numel()
                else 0.0,
                "mean_train_sequence_weight": float(train_sequence_weights.mean().item())
                if train_sequence_weights.numel()
                else 0.0,
                "kept_train_sequence_fraction": float((train_sequence_weights > 0).float().mean().item())
                if train_sequence_weights.numel()
                else 0.0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
