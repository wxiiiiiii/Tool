from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from universal_agent_policy.adapters import AdapterConfig, AdapterPolicy, PolicyHead, build_adapter
from universal_agent_policy.data import load_hidden_tensor, split_indices


@dataclass
class AnchorRunResult:
    strategy: str
    k: int
    normalization: str
    feature_mode: str
    target_calibration: str
    policy_model: str
    source_train_accuracy: float
    source_val_accuracy: float
    source_full_accuracy: float
    target_full_accuracy: float
    target_val_accuracy: float
    source_target_agreement: float
    source_target_val_agreement: float
    target_transition_accuracy: float
    source_target_transition_agreement: float
    target_trainable_params: int
    target_calibration_params: int
    target_gradient_steps: int
    target_labels_used: int
    source_online_queries: int
    num_anchors: int
    num_transition_anchors: int


class RelativeLinearPolicy(nn.Module):
    def __init__(self, input_dim: int, num_actions: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


class RelativeMLPPolicy(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Zero-target-training anchor-relative policy interface. A source policy is trained once "
            "over source anchor-relative coordinates; each target only caches anchor activations."
        )
    )
    parser.add_argument("--source-tensor", required=True)
    parser.add_argument("--target-tensor", required=True)
    parser.add_argument("--source-policy", default=None, help="Optional source checkpoint for confidence/margin anchor selection.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[
            "random",
            "semantic",
            "policy_prototypes",
            "decision_boundary",
            "policy_critical",
            "policy_logdet",
            "transition_critical",
            "policy_transition_mixture",
        ],
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument(
        "--feature-modes",
        nargs="+",
        default=["state"],
        choices=[
            "state",
            "state_pool",
            "state_delta",
            "state_delta_pool",
            "state_transition",
            "state_transition_pool",
            "all",
            "all_pool",
        ],
    )
    parser.add_argument("--target-calibrations", nargs="+", default=["none"], choices=["none", "ridge", "orthogonal"])
    parser.add_argument("--calibration-ridge", type=float, default=1e-3)
    parser.add_argument("--normalizations", nargs="+", default=["raw", "center", "whiten", "rank"])
    parser.add_argument("--transition-k", type=int, default=-1, help="Number of transition anchors; default uses K.")
    parser.add_argument("--policy-model", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--linear-lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _list_field(obj: dict[str, Any], key: str) -> list[Any] | None:
    value = obj.get(key)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return list(value)


def validate_target_alignment(source: dict[str, Any], target: dict[str, Any]) -> None:
    if len(source["y"]) != len(target["y"]):
        raise ValueError("source and target tensors must have the same number of paired decision states")
    source_ids = _list_field(source, "sample_id")
    target_ids = _list_field(target, "sample_id")
    if source_ids is not None and target_ids is not None and source_ids != target_ids:
        raise ValueError("source and target sample_id lists differ; anchor-relative coordinates need paired anchor prompts")


def load_source_logits(path: str | None, h: torch.Tensor, num_actions: int, device: torch.device) -> torch.Tensor | None:
    if not path:
        return None
    checkpoint = torch.load(path, map_location="cpu")
    adapter_config = AdapterConfig(**checkpoint["adapter_config"])
    model = AdapterPolicy(build_adapter(adapter_config), PolicyHead(checkpoint["z_dim"], num_actions))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, h.shape[0], 512):
            logits.append(model(h[start : start + 512].to(device)).cpu())
    return torch.cat(logits, dim=0)


def unique_preserve_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def fill_to_k(seed_indices: list[int], train_idx: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    selected = unique_preserve_order(seed_indices)
    if len(selected) >= k:
        return torch.tensor(selected[:k], dtype=torch.long)
    selected_set = set(selected)
    remaining = [int(i) for i in train_idx.tolist() if int(i) not in selected_set]
    generator = torch.Generator().manual_seed(seed)
    if remaining:
        order = torch.randperm(len(remaining), generator=generator).tolist()
        selected.extend(remaining[i] for i in order[: k - len(selected)])
    return torch.tensor(selected[: min(k, len(selected))], dtype=torch.long)


def farthest_first(h: torch.Tensor, candidates: torch.Tensor, k: int, seed: int) -> list[int]:
    if len(candidates) <= k:
        return [int(i) for i in candidates.tolist()]
    generator = torch.Generator().manual_seed(seed)
    cand = candidates[torch.randperm(len(candidates), generator=generator)]
    x = F.normalize(h[cand], dim=-1)
    selected_local = [0]
    min_dist = 1.0 - (x @ x[0])
    for _ in range(1, k):
        next_local = int(torch.argmax(min_dist).item())
        selected_local.append(next_local)
        dist = 1.0 - (x @ x[next_local])
        min_dist = torch.minimum(min_dist, dist)
    return [int(cand[i].item()) for i in selected_local]


def logdet_greedy(h: torch.Tensor, candidates: torch.Tensor, k: int, seed: int, eps: float = 1e-3) -> list[int]:
    if len(candidates) <= k:
        return [int(i) for i in candidates.tolist()]
    selected = farthest_first(h, candidates, 1, seed)
    selected_set = set(selected)
    remaining = [int(i) for i in candidates.tolist() if int(i) not in selected_set]
    x = F.normalize(h, dim=-1)
    for _ in range(1, k):
        best_idx = remaining[0]
        best_score = float("-inf")
        for idx in remaining:
            trial = torch.tensor(selected + [idx], dtype=torch.long)
            gram = x[trial] @ x[trial].T
            sign, score = torch.linalg.slogdet(gram + eps * torch.eye(len(trial)))
            value = float(score.item()) if sign.item() > 0 else float("-inf")
            if value > best_score:
                best_score = value
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)
        if not remaining:
            break
    return selected


def random_anchors(train_idx: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_idx), generator=generator)
    return train_idx[perm[: min(k, len(train_idx))]]


def semantic_anchors(source: dict[str, Any], train_idx: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    categories = _list_field(source, "category")
    if not categories:
        return torch.tensor(farthest_first(source["h"], train_idx, min(k, len(train_idx)), seed), dtype=torch.long)
    by_category: dict[str, list[int]] = {}
    train_set = set(int(i) for i in train_idx.tolist())
    for idx, category in enumerate(categories):
        if idx in train_set:
            by_category.setdefault(str(category), []).append(idx)
    if not by_category:
        return random_anchors(train_idx, k, seed)
    per_category = max(1, k // len(by_category))
    selected: list[int] = []
    for offset, (_, indices) in enumerate(sorted(by_category.items())):
        cand = torch.tensor(indices, dtype=torch.long)
        selected.extend(farthest_first(source["h"], cand, min(per_category, len(cand)), seed + offset))
    return fill_to_k(selected, train_idx, k, seed)


def confidence_and_margin(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = logits.softmax(dim=-1)
    confidence = probs.max(dim=-1).values
    pred = probs.argmax(dim=-1)
    top2 = probs.topk(min(2, probs.shape[-1]), dim=-1).values
    if top2.shape[-1] == 1:
        margin = top2[:, 0]
    else:
        margin = top2[:, 0] - top2[:, 1]
    return confidence, margin, pred


def policy_prototype_anchors(train_idx: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor | None, k: int, seed: int) -> torch.Tensor:
    if logits is None:
        return semantic_label_anchors(train_idx, labels, k, seed)
    confidence, _, pred = confidence_and_margin(logits)
    selected: list[int] = []
    classes = sorted(set(int(v) for v in labels[train_idx].tolist()))
    per_class = max(1, k // max(1, len(classes)))
    for cls in classes:
        mask = (labels[train_idx] == cls) & (pred[train_idx] == labels[train_idx])
        candidates = train_idx[mask]
        if len(candidates) == 0:
            candidates = train_idx[labels[train_idx] == cls]
        order = torch.argsort(confidence[candidates], descending=True)
        selected.extend(int(i) for i in candidates[order[:per_class]].tolist())
    if len(selected) < k:
        selected_set = set(selected)
        remaining = torch.tensor([int(i) for i in train_idx.tolist() if int(i) not in selected_set], dtype=torch.long)
        if len(remaining):
            order = torch.argsort(confidence[remaining], descending=True)
            selected.extend(int(i) for i in remaining[order[: k - len(selected)]].tolist())
    return fill_to_k(selected, train_idx, k, seed)


def semantic_label_anchors(train_idx: torch.Tensor, labels: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    selected: list[int] = []
    classes = sorted(set(int(v) for v in labels[train_idx].tolist()))
    per_class = max(1, k // max(1, len(classes)))
    for cls in classes:
        selected.extend(int(i) for i in train_idx[labels[train_idx] == cls][:per_class].tolist())
    return fill_to_k(selected, train_idx, k, seed)


def policy_critical_anchors(train_idx: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor | None, k: int, seed: int) -> torch.Tensor:
    if logits is None:
        return semantic_label_anchors(train_idx, labels, k, seed)
    _, margin, _ = confidence_and_margin(logits)
    prototype_budget = max(1, (2 * k) // 3)
    selected = [int(i) for i in policy_prototype_anchors(train_idx, labels, logits, prototype_budget, seed).tolist()]
    selected_set = set(selected)
    remaining = torch.tensor([int(i) for i in train_idx.tolist() if int(i) not in selected_set], dtype=torch.long)
    if len(remaining):
        boundary_budget = k - len(selected)
        order = torch.argsort(margin[remaining], descending=False)
        selected.extend(int(i) for i in remaining[order[:boundary_budget]].tolist())
    return fill_to_k(selected, train_idx, k, seed)


def decision_boundary_anchors(train_idx: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor | None, k: int, seed: int) -> torch.Tensor:
    if logits is None:
        return random_anchors(train_idx, k, seed)
    _, margin, pred = confidence_and_margin(logits)
    selected: list[int] = []
    classes = sorted(set(int(v) for v in labels[train_idx].tolist()))
    per_class = max(1, k // max(1, len(classes)))
    for cls in classes:
        candidates = train_idx[(labels[train_idx] == cls) & (pred[train_idx] == labels[train_idx])]
        if len(candidates) == 0:
            candidates = train_idx[labels[train_idx] == cls]
        if len(candidates) == 0:
            continue
        order = torch.argsort(margin[candidates], descending=False)
        selected.extend(int(i) for i in candidates[order[:per_class]].tolist())
    return fill_to_k(selected, train_idx, k, seed)


def policy_logdet_anchors(train_idx: torch.Tensor, labels: torch.Tensor, source: dict[str, Any], logits: torch.Tensor | None, k: int, seed: int) -> torch.Tensor:
    if logits is None:
        return torch.tensor(logdet_greedy(source["h"], train_idx, min(k, len(train_idx)), seed), dtype=torch.long)
    confidence, margin, pred = confidence_and_margin(logits)
    classes = sorted(set(int(v) for v in labels[train_idx].tolist()))
    selected: list[int] = []
    per_class = max(1, k // max(1, len(classes) * 2))
    for cls in classes:
        correct = train_idx[(labels[train_idx] == cls) & (pred[train_idx] == labels[train_idx])]
        if len(correct) == 0:
            correct = train_idx[labels[train_idx] == cls]
        high_conf = correct[torch.argsort(confidence[correct], descending=True)[:per_class]]
        boundary = correct[torch.argsort(margin[correct], descending=False)[:per_class]]
        selected.extend(int(i) for i in high_conf.tolist())
        selected.extend(int(i) for i in boundary.tolist())
    remaining_budget = k - len(unique_preserve_order(selected))
    if remaining_budget > 0:
        selected_set = set(selected)
        remaining = torch.tensor([int(i) for i in train_idx.tolist() if int(i) not in selected_set], dtype=torch.long)
        if len(remaining):
            selected.extend(logdet_greedy(source["h"], remaining, min(remaining_budget, len(remaining)), seed))
    return fill_to_k(selected, train_idx, k, seed)


def parse_sample_id(sample_id: Any) -> tuple[str, int, int, int] | None:
    text = str(sample_id)
    parts = text.rsplit("_", 3)
    if len(parts) != 4:
        return None
    domain, task_id, trial, turn = parts
    try:
        return domain, int(task_id), int(trial), int(turn)
    except ValueError:
        return None


def build_transition_pairs_from_tensor(obj: dict[str, Any]) -> list[tuple[int, int]]:
    sample_ids = _list_field(obj, "sample_id")
    if not sample_ids:
        return []
    groups: dict[tuple[str, int, int], list[tuple[int, int]]] = {}
    for idx, sample_id in enumerate(sample_ids):
        parsed = parse_sample_id(sample_id)
        if parsed is None:
            continue
        domain, task_id, trial, turn = parsed
        groups.setdefault((domain, task_id, trial), []).append((turn, idx))
    pairs: list[tuple[int, int]] = []
    for items in groups.values():
        ordered = [idx for _, idx in sorted(items)]
        pairs.extend((left, right) for left, right in zip(ordered, ordered[1:]))
    return pairs


def transition_critical_state_anchors(
    source: dict[str, Any],
    train_idx: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor | None,
    k: int,
    seed: int,
) -> torch.Tensor:
    train_set = set(int(i) for i in train_idx.tolist())
    pairs = [(l, r) for l, r in build_transition_pairs_from_tensor(source) if l in train_set and r in train_set]
    if not pairs:
        return policy_critical_anchors(train_idx, labels, logits, k, seed)
    scores: list[tuple[float, int]] = []
    if logits is not None:
        centered = logits - logits.mean(dim=-1, keepdim=True)
        for left, right in pairs:
            action_change = 1.0 if int(labels[left]) != int(labels[right]) else 0.0
            delta = float((centered[right] - centered[left]).norm().item())
            scores.append((delta + action_change, right))
    else:
        for left, right in pairs:
            action_change = 1.0 if int(labels[left]) != int(labels[right]) else 0.0
            delta = float((source["h"][right] - source["h"][left]).norm().item())
            scores.append((delta + action_change, right))
    selected = [idx for _, idx in sorted(scores, reverse=True)]
    return fill_to_k(selected, train_idx, k, seed)


def policy_transition_mixture_anchors(
    source: dict[str, Any],
    train_idx: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor | None,
    k: int,
    seed: int,
) -> torch.Tensor:
    prototype_budget = max(1, k // 3)
    boundary_budget = max(1, k // 3)
    transition_budget = max(1, k - prototype_budget - boundary_budget)
    selected: list[int] = []
    selected.extend(int(i) for i in policy_prototype_anchors(train_idx, labels, logits, prototype_budget, seed).tolist())
    selected.extend(int(i) for i in decision_boundary_anchors(train_idx, labels, logits, boundary_budget, seed + 1).tolist())
    selected.extend(int(i) for i in transition_critical_state_anchors(source, train_idx, labels, logits, transition_budget, seed + 2).tolist())
    return fill_to_k(selected, train_idx, k, seed)


def select_anchors(
    strategy: str,
    source: dict[str, Any],
    train_idx: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor | None,
    k: int,
    seed: int,
) -> torch.Tensor:
    if strategy == "random":
        return random_anchors(train_idx, k, seed)
    if strategy == "semantic":
        return semantic_anchors(source, train_idx, k, seed)
    if strategy == "policy_prototypes":
        return policy_prototype_anchors(train_idx, labels, logits, k, seed)
    if strategy == "decision_boundary":
        return decision_boundary_anchors(train_idx, labels, logits, k, seed)
    if strategy == "policy_critical":
        return policy_critical_anchors(train_idx, labels, logits, k, seed)
    if strategy == "policy_logdet":
        return policy_logdet_anchors(train_idx, labels, source, logits, k, seed)
    if strategy == "transition_critical":
        return transition_critical_state_anchors(source, train_idx, labels, logits, k, seed)
    if strategy == "policy_transition_mixture":
        return policy_transition_mixture_anchors(source, train_idx, labels, logits, k, seed)
    raise ValueError(f"Unknown anchor strategy: {strategy}")


def relative_coordinates(h: torch.Tensor, anchor_idx: torch.Tensor, normalization: str) -> torch.Tensor:
    x = F.normalize(h, dim=-1)
    anchors = F.normalize(h[anchor_idx], dim=-1)
    coords = x @ anchors.T
    gram = anchors @ anchors.T
    if normalization == "raw":
        return coords
    if normalization == "center":
        mean = gram.mean(dim=0, keepdim=True)
        std = gram.std(dim=0, keepdim=True).clamp_min(1e-4)
        return (coords - mean) / std
    if normalization == "whiten":
        mean = gram.mean(dim=0, keepdim=True)
        centered_coords = coords - mean
        centered_gram = gram - mean
        cov = (centered_gram.T @ centered_gram) / max(1, centered_gram.shape[0] - 1)
        cov = cov + 1e-3 * torch.eye(cov.shape[0])
        evals, evecs = torch.linalg.eigh(cov)
        whitening = evecs @ torch.diag(evals.clamp_min(1e-5).rsqrt()) @ evecs.T
        return centered_coords @ whitening
    if normalization == "rank":
        if coords.shape[1] <= 1:
            return torch.zeros_like(coords)
        ranks = torch.argsort(torch.argsort(coords, dim=1), dim=1).float()
        return ranks / float(coords.shape[1] - 1)
    raise ValueError(f"Unknown normalization: {normalization}")


def delta_relative_coordinates(coords: torch.Tensor, pairs: list[tuple[int, int]]) -> torch.Tensor:
    out = torch.zeros_like(coords)
    for left, right in pairs:
        out[right] = coords[right] - coords[left]
    return out


def select_transition_anchor_pairs(
    source: dict[str, Any],
    train_idx: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor | None,
    k: int,
) -> list[tuple[int, int]]:
    train_set = set(int(i) for i in train_idx.tolist())
    pairs = [(l, r) for l, r in build_transition_pairs_from_tensor(source) if l in train_set and r in train_set]
    if not pairs:
        return []
    if logits is not None:
        centered = logits - logits.mean(dim=-1, keepdim=True)
        scored = []
        for left, right in pairs:
            action_change = 1.0 if int(labels[left]) != int(labels[right]) else 0.0
            score = float((centered[right] - centered[left]).norm().item()) + action_change
            scored.append((score, left, right))
    else:
        scored = [
            (
                float((source["h"][right] - source["h"][left]).norm().item()) + (1.0 if int(labels[left]) != int(labels[right]) else 0.0),
                left,
                right,
            )
            for left, right in pairs
        ]
    return [(left, right) for _, left, right in sorted(scored, reverse=True)[:k]]


def transition_anchor_coordinates(h: torch.Tensor, pairs: list[tuple[int, int]], transition_anchors: list[tuple[int, int]]) -> torch.Tensor:
    if not transition_anchors:
        return h.new_zeros((h.shape[0], 0))
    anchor_deltas = torch.stack([h[right] - h[left] for left, right in transition_anchors])
    anchor_deltas = F.normalize(anchor_deltas, dim=-1)
    out = h.new_zeros((h.shape[0], len(transition_anchors)))
    if not pairs:
        return out
    state_delta = torch.stack([h[right] - h[left] for left, right in pairs])
    state_delta = F.normalize(state_delta, dim=-1)
    sims = state_delta @ anchor_deltas.T
    for row, (_, right) in enumerate(pairs):
        out[right] = sims[row]
    return out


def action_pool_coordinates(coords: torch.Tensor, anchor_labels: torch.Tensor, num_actions: int) -> torch.Tensor:
    pooled: list[torch.Tensor] = []
    for action in range(num_actions):
        cols = anchor_labels == action
        if bool(cols.any()):
            action_coords = coords[:, cols]
            pooled.append(action_coords.mean(dim=1, keepdim=True))
            pooled.append(action_coords.max(dim=1, keepdim=True).values)
        else:
            pooled.append(coords.new_zeros((coords.shape[0], 1)))
            pooled.append(coords.new_zeros((coords.shape[0], 1)))
    return torch.cat(pooled, dim=-1)


def build_features(
    obj: dict[str, Any],
    anchor_idx: torch.Tensor,
    normalization: str,
    feature_mode: str,
    transition_anchors: list[tuple[int, int]],
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    coords = relative_coordinates(obj["h"], anchor_idx, normalization)
    use_pool = feature_mode.endswith("_pool")
    base_mode = feature_mode.removesuffix("_pool")
    parts = [coords]
    if use_pool:
        if labels is None:
            raise ValueError("labels are required for pooled policy-anchor coordinates")
        parts.append(action_pool_coordinates(coords, labels[anchor_idx], int(labels.max().item()) + 1))
    state_features = torch.cat(parts, dim=-1)
    if base_mode == "state":
        return state_features
    pairs = build_transition_pairs_from_tensor(obj)
    delta_coords = delta_relative_coordinates(coords, pairs)
    if base_mode == "state_delta":
        return torch.cat([state_features, delta_coords], dim=-1)
    transition_coords = transition_anchor_coordinates(obj["h"], pairs, transition_anchors)
    if base_mode == "state_transition":
        return torch.cat([state_features, transition_coords], dim=-1)
    if base_mode == "all":
        return torch.cat([state_features, delta_coords, transition_coords], dim=-1)
    raise ValueError(f"Unknown feature mode: {feature_mode}")


def calibration_indices(anchor_idx: torch.Tensor, transition_anchors: list[tuple[int, int]], feature_mode: str) -> torch.Tensor:
    indices = [int(i) for i in anchor_idx.tolist()]
    if "transition" in feature_mode or feature_mode.startswith("all"):
        for left, right in transition_anchors:
            indices.append(int(left))
            indices.append(int(right))
    return torch.tensor(unique_preserve_order(indices), dtype=torch.long)


def calibrate_target_features(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    calibration_idx: torch.Tensor,
    method: str,
    ridge: float,
) -> tuple[torch.Tensor, int]:
    if method == "none":
        return target_features, 0
    x = target_features[calibration_idx]
    y = source_features[calibration_idx]
    if x.shape[1] != y.shape[1]:
        raise ValueError("source and target feature dimensions must match for anchor-only calibration")
    if method == "ridge":
        xtx = x.T @ x
        eye = torch.eye(xtx.shape[0], dtype=xtx.dtype)
        coef = torch.linalg.solve(xtx + ridge * eye, x.T @ y)
        return target_features @ coef, int(coef.numel())
    if method == "orthogonal":
        xty = x.T @ y
        u, _, vh = torch.linalg.svd(xty, full_matrices=False)
        rotation = u @ vh
        return target_features @ rotation, int(rotation.numel())
    raise ValueError(f"Unknown target calibration: {method}")


def build_policy(input_dim: int, num_actions: int, kind: str, hidden_dim: int) -> nn.Module:
    if kind == "linear":
        return RelativeLinearPolicy(input_dim, num_actions)
    if kind == "mlp":
        return RelativeMLPPolicy(input_dim, num_actions, hidden_dim)
    raise ValueError(f"Unknown policy model: {kind}")


def accuracy(pred: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor | None = None) -> float:
    if idx is not None:
        pred = pred[idx]
        labels = labels[idx]
    if len(labels) == 0:
        return 0.0
    return (pred == labels).float().mean().item()


def train_source_relative_policy(
    source_coords: torch.Tensor,
    target_coords: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[AnchorRunResult, nn.Module]:
    num_actions = int(labels.max().item()) + 1
    model = build_policy(source_coords.shape[1], num_actions, args.policy_model, args.hidden_dim).to(device)
    lr = args.linear_lr if args.policy_model == "linear" else args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    x_source = source_coords.to(device)
    y = labels.to(device)
    best_state = None
    best_val = -1.0
    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x_source[train_idx.to(device)]), y[train_idx.to(device)])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            pred_val = model(x_source[val_idx.to(device)]).argmax(dim=-1).cpu()
            val_acc = accuracy(pred_val, labels[val_idx])
        if val_acc > best_val:
            best_val = val_acc
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_run(
    model: nn.Module,
    source_coords: torch.Tensor,
    target_coords: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    transition_pairs: list[tuple[int, int]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        source_pred = model(source_coords.to(device)).argmax(dim=-1).cpu()
        target_pred = model(target_coords.to(device)).argmax(dim=-1).cpu()
    target_transition_accuracy = 0.0
    source_target_transition_agreement = 0.0
    if transition_pairs:
        target_hits = 0
        agreement_hits = 0
        for left, right in transition_pairs:
            target_hits += int(target_pred[left] == labels[left] and target_pred[right] == labels[right])
            agreement_hits += int(target_pred[left] == source_pred[left] and target_pred[right] == source_pred[right])
        target_transition_accuracy = target_hits / len(transition_pairs)
        source_target_transition_agreement = agreement_hits / len(transition_pairs)
    return {
        "source_train_accuracy": accuracy(source_pred, labels, train_idx),
        "source_val_accuracy": accuracy(source_pred, labels, val_idx),
        "source_full_accuracy": accuracy(source_pred, labels),
        "target_full_accuracy": accuracy(target_pred, labels),
        "target_val_accuracy": accuracy(target_pred, labels, val_idx),
        "source_target_agreement": accuracy(target_pred, source_pred),
        "source_target_val_agreement": accuracy(target_pred, source_pred, val_idx),
        "target_transition_accuracy": target_transition_accuracy,
        "source_target_transition_agreement": source_target_transition_agreement,
    }


def write_report(results: list[AnchorRunResult], out_dir: Path, source_tensor: str, target_tensor: str) -> None:
    best = max(results, key=lambda r: (r.target_full_accuracy, r.source_target_agreement))
    rows = sorted(results, key=lambda r: (r.strategy, r.k, r.normalization))
    lines = [
        "# Anchor-Relative Policy Interface Results",
        "",
        "This experiment trains the policy interface once on source anchor-relative coordinates and performs target onboarding with anchor forward/cache only.",
        "",
        f"- Source tensor: `{source_tensor}`",
        f"- Target tensor: `{target_tensor}`",
        "- Target labels used: `0`",
        "- Target gradient steps: `0`",
        "- Source online queries during target onboarding: `0`",
        "",
        "## Best Run",
        "",
        "| Strategy | K | Feature | Calibration | Normalization | Policy | Source Val Acc | Target Acc | Agreement | Transition Agreement |",
        "|---|---:|---|---|---|---|---:|---:|---:|---:|",
        (
            f"| {best.strategy} | {best.k} | {best.feature_mode} | {best.target_calibration} | "
            f"{best.normalization} | {best.policy_model} | "
            f"{best.source_val_accuracy:.2%} | {best.target_full_accuracy:.2%} | {best.source_target_agreement:.2%} | "
            f"{best.source_target_transition_agreement:.2%} |"
        ),
        "",
        "## All Runs",
        "",
        "| Strategy | K | Feature | Calibration | Norm | Source Val | Source Full | Target Full | Target Val | Agreement | Trans Acc | Trans Agreement |",
        "|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.strategy} | {r.k} | {r.feature_mode} | {r.target_calibration} | {r.normalization} | {r.source_val_accuracy:.2%} | "
            f"{r.source_full_accuracy:.2%} | {r.target_full_accuracy:.2%} | "
            f"{r.target_val_accuracy:.2%} | {r.source_target_agreement:.2%} | "
            f"{r.target_transition_accuracy:.2%} | {r.source_target_transition_agreement:.2%} |"
        )
    (out_dir / "anchor_relative_policy_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    source = load_hidden_tensor(args.source_tensor)
    target = load_hidden_tensor(args.target_tensor)
    validate_target_alignment(source, target)

    labels = source["y"]
    if not torch.equal(labels, target["y"]):
        raise ValueError("source and target labels differ; paired target evaluation would be ambiguous")
    train_idx, val_idx = split_indices(len(labels), args.val_frac, args.seed)
    num_actions = int(labels.max().item()) + 1
    source_logits = load_source_logits(args.source_policy, source["h"], num_actions, device)
    transition_pairs = build_transition_pairs_from_tensor(source)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[AnchorRunResult] = []
    best_payload: dict[str, Any] | None = None
    best_score = -1.0

    for strategy in args.strategies:
        for k in args.k_values:
            anchor_idx = select_anchors(strategy, source, train_idx, labels, source_logits, k, args.seed)
            if len(anchor_idx) == 0:
                continue
            transition_k = k if args.transition_k < 0 else args.transition_k
            transition_anchors = select_transition_anchor_pairs(source, train_idx, labels, source_logits, transition_k)
            for normalization in args.normalizations:
                for feature_mode in args.feature_modes:
                    source_coords = build_features(source, anchor_idx, normalization, feature_mode, transition_anchors, labels)
                    raw_target_coords = build_features(target, anchor_idx, normalization, feature_mode, transition_anchors, labels)
                    model = train_source_relative_policy(source_coords, raw_target_coords, labels, train_idx, val_idx, args, device)
                    calib_idx = calibration_indices(anchor_idx, transition_anchors, feature_mode)
                    for target_calibration in args.target_calibrations:
                        target_coords, calibration_params = calibrate_target_features(
                            source_coords,
                            raw_target_coords,
                            calib_idx,
                            target_calibration,
                            args.calibration_ridge,
                        )
                        metrics = evaluate_run(model, source_coords, target_coords, labels, train_idx, val_idx, transition_pairs, device)
                        result = AnchorRunResult(
                            strategy=strategy,
                            k=int(k),
                            normalization=normalization,
                            feature_mode=feature_mode,
                            target_calibration=target_calibration,
                            policy_model=args.policy_model,
                            target_trainable_params=0,
                            target_calibration_params=calibration_params,
                            target_gradient_steps=0,
                            target_labels_used=0,
                            source_online_queries=0,
                            num_anchors=int(len(anchor_idx)),
                            num_transition_anchors=int(len(transition_anchors)),
                            **metrics,
                        )
                        results.append(result)
                        score = result.target_full_accuracy + 0.1 * result.source_target_agreement + 0.05 * result.source_target_transition_agreement
                        if score > best_score:
                            best_score = score
                            best_payload = {
                                "result": asdict(result),
                                "anchor_indices": anchor_idx.tolist(),
                                "transition_anchor_pairs": transition_anchors,
                                "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                            }

    payload = {
        "metadata": {
            "source_tensor": args.source_tensor,
            "target_tensor": args.target_tensor,
            "source_policy": args.source_policy,
            "seed": args.seed,
            "val_frac": args.val_frac,
            "policy_model": args.policy_model,
            "hidden_dim": args.hidden_dim,
            "feature_modes": args.feature_modes,
            "target_calibrations": args.target_calibrations,
            "target_labels_used": 0,
            "target_gradient_steps": 0,
            "source_online_queries": 0,
        },
        "results": [asdict(r) for r in results],
    }
    (out_dir / "anchor_relative_policy_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if best_payload is not None:
        torch.save(best_payload, out_dir / "best_anchor_relative_policy.pt")
    write_report(results, out_dir, args.source_tensor, args.target_tensor)
    best = max(results, key=lambda r: (r.target_full_accuracy, r.source_target_agreement))
    print(json.dumps(asdict(best), indent=2))


if __name__ == "__main__":
    main()
