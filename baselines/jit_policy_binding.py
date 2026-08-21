from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from baselines import eval_protocol
from baselines.anchor_relative_policy import (
    build_features,
    build_transition_pairs_from_tensor,
    calibrate_target_features,
    load_source_logits,
    select_anchors,
    select_transition_anchor_pairs,
    train_source_relative_policy,
)
from baselines.tool_policy_utils import load_rows
from universal_agent_policy.data import load_hidden_tensor


@dataclass
class BindingResult:
    target: str
    method: str
    memory_mode: str
    k: int
    memory_k: int
    beta: float
    gate_margin: float
    temperature: float
    source_val_accuracy: float
    source_full_accuracy: float
    target_full_accuracy: float
    target_val_accuracy: float
    source_target_agreement: float
    source_target_transition_agreement: float
    target_transition_accuracy: float
    target_labels_used: int
    target_gradient_steps: int
    source_online_queries: int
    memory_size: int
    memory_filter: str = "all"
    gate_open_rate: float = 0.0
    changed_rate: float = 0.0
    improved_rate: float = 0.0
    harmed_rate: float = 0.0
    quadrant_a_count: int = 0
    quadrant_b_count: int = 0
    quadrant_c_count: int = 0
    quadrant_d_count: int = 0
    quadrant_b_harmed_rate: float = 0.0
    quadrant_c_improved_rate: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Just-in-time non-parametric policy binding over anchor-relative coordinates. "
            "Builds source policy memory once, then corrects target Universal Action logits at inference."
        )
    )
    parser.add_argument("--source-tensor", required=True)
    parser.add_argument("--target-tensor", required=True)
    parser.add_argument("--source-policy", required=True)
    parser.add_argument("--decision-jsonl", help="Decision rows used for trajectory/task-disjoint split.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-name", default="target")
    parser.add_argument("--strategy", default="transition_critical")
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--feature-mode", default="state", choices=["state", "state_delta", "state_transition", "all"])
    parser.add_argument("--normalization", default="center")
    parser.add_argument("--target-calibration", default="ridge", choices=["none", "ridge"])
    parser.add_argument("--calibration-ridge", type=float, default=1e-3)
    parser.add_argument("--policy-model", default="mlp", choices=["linear", "mlp"])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--linear-lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--split-unit", choices=["decision", "trajectory", "task"], default="trajectory")
    parser.add_argument("--split-json", help="Load a precomputed split JSON.")
    parser.add_argument("--write-split-json", help="Write the resolved train/val split for exact reuse.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-modes", nargs="+", default=["raw", "capsule"], choices=["raw", "capsule"])
    parser.add_argument("--memory-filters", nargs="+", default=["all"], choices=["all", "source_correct", "source_reliable"])
    parser.add_argument("--memory-min-confidence", type=float, default=0.55)
    parser.add_argument("--memory-min-margin", type=float, default=0.05)
    parser.add_argument("--memory-require-stable-transition", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["retrieval_only", "jit_residual", "jit_blend", "jit_gated", "jit_reliable", "jit_selective"],
        choices=["retrieval_only", "jit_residual", "jit_blend", "jit_gated", "jit_reliable", "jit_selective", "jit_counterfactual"],
    )
    parser.add_argument("--memory-ks", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    parser.add_argument("--selection-metric", choices=["source_val", "target_unsupervised"], default="source_val")
    parser.add_argument("--gate-margins", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    parser.add_argument("--reliability-sim-margins", nargs="+", type=float, default=[0.0, 0.01, 0.03])
    parser.add_argument("--reliability-max-entropies", nargs="+", type=float, default=[0.4, 0.6, 0.8])
    parser.add_argument("--reliability-base-margins", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    parser.add_argument("--selective-base-margins", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    parser.add_argument("--selective-source-margins", nargs="+", type=float, default=[0.05, 0.1])
    parser.add_argument("--selective-override-margins", nargs="+", type=float, default=[0.0, 0.05, 0.1])
    parser.add_argument("--selective-sim-margins", nargs="+", type=float, default=[0.0, 0.01, 0.03])
    parser.add_argument("--selective-max-entropies", nargs="+", type=float, default=[0.4, 0.6, 0.8])
    parser.add_argument("--consistency-trials", type=int, default=8)
    parser.add_argument("--consistency-drop-rate", type=float, default=0.2)
    parser.add_argument("--consistency-max-stabilities", nargs="+", type=float, default=[0.5, 0.7, 0.9])
    return parser.parse_args()


def accuracy(pred: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor | None = None) -> float:
    if idx is not None:
        pred = pred[idx]
        labels = labels[idx]
    if len(labels) == 0:
        return 0.0
    return (pred == labels).float().mean().item()


def transition_metrics(
    target_pred: torch.Tensor,
    source_pred: torch.Tensor,
    labels: torch.Tensor,
    pairs: list[tuple[int, int]],
) -> tuple[float, float]:
    if not pairs:
        return 0.0, 0.0
    target_hits = 0
    agreement_hits = 0
    for left, right in pairs:
        target_hits += int(target_pred[left] == labels[left] and target_pred[right] == labels[right])
        agreement_hits += int(target_pred[left] == source_pred[left] and target_pred[right] == source_pred[right])
    return target_hits / len(pairs), agreement_hits / len(pairs)


def centered_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits - logits.mean(dim=-1, keepdim=True)


def predict_model(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(x.to(device)).cpu()


def perturbation_stability(
    model: torch.nn.Module,
    features: torch.Tensor,
    base_pred: torch.Tensor,
    device: torch.device,
    trials: int,
    drop_rate: float,
    seed: int,
) -> torch.Tensor:
    if trials <= 0 or drop_rate <= 0.0:
        return torch.ones(features.shape[0])
    keep_prob = max(1e-3, 1.0 - drop_rate)
    generator = torch.Generator().manual_seed(seed)
    hits = torch.zeros(features.shape[0])
    model.eval()
    with torch.no_grad():
        for _ in range(trials):
            mask = (torch.rand(features.shape[1], generator=generator) < keep_prob).float() / keep_prob
            logits = model((features * mask).to(device)).cpu()
            hits += logits.argmax(dim=-1).eq(base_pred).float()
    return hits / float(trials)


def topk_retrieval_logits(
    query: torch.Tensor,
    memory_keys: torch.Tensor,
    memory_logits: torch.Tensor,
    memory_k: int,
    temperature: float,
) -> torch.Tensor:
    q = F.normalize(query, dim=-1)
    keys = F.normalize(memory_keys, dim=-1)
    sims = q @ keys.T
    k = min(memory_k, memory_keys.shape[0])
    values, indices = sims.topk(k, dim=-1)
    weights = F.softmax(values / max(temperature, 1e-6), dim=-1)
    neighbor_logits = memory_logits[indices]
    return (weights.unsqueeze(-1) * neighbor_logits).sum(dim=1)


def topk_retrieval_with_stats(
    query: torch.Tensor,
    memory_keys: torch.Tensor,
    memory_logits: torch.Tensor,
    memory_k: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = F.normalize(query, dim=-1)
    keys = F.normalize(memory_keys, dim=-1)
    sims = q @ keys.T
    k = min(memory_k, memory_keys.shape[0])
    values, indices = sims.topk(k, dim=-1)
    weights = F.softmax(values / max(temperature, 1e-6), dim=-1)
    neighbor_logits = memory_logits[indices]
    retrieved = (weights.unsqueeze(-1) * neighbor_logits).sum(dim=1)
    if k == 1:
        sim_margin = values[:, 0].new_zeros(values.shape[0])
    else:
        sim_margin = values[:, 0] - values[:, 1]
    num_actions = memory_logits.shape[-1]
    neighbor_actions = neighbor_logits.argmax(dim=-1)
    action_votes = query.new_zeros((query.shape[0], num_actions))
    action_votes.scatter_add_(1, neighbor_actions, weights)
    action_entropy = -(action_votes * action_votes.clamp_min(1e-8).log()).sum(dim=-1)
    action_entropy = action_entropy / torch.log(torch.tensor(float(num_actions), dtype=action_entropy.dtype))
    return retrieved, sim_margin, action_entropy


def build_capsules(
    features: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    memory_idx: torch.Tensor,
    per_action: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = logits.softmax(dim=-1)
    pred = probs.argmax(dim=-1)
    confidence = probs.max(dim=-1).values
    keys: list[torch.Tensor] = []
    vals: list[torch.Tensor] = []
    for action in sorted(set(int(v) for v in labels[memory_idx].tolist())):
        candidates = memory_idx[(labels[memory_idx] == action) & (pred[memory_idx] == labels[memory_idx])]
        if len(candidates) == 0:
            candidates = memory_idx[labels[memory_idx] == action]
        if len(candidates) == 0:
            continue
        ordered = candidates[torch.argsort(confidence[candidates], descending=True)]
        chunks = torch.chunk(ordered[: max(1, per_action)], min(max(1, per_action), len(ordered)))
        for chunk in chunks:
            keys.append(features[chunk].mean(dim=0))
            vals.append(logits[chunk].mean(dim=0))
    if not keys:
        return features[memory_idx], logits[memory_idx]
    return torch.stack(keys), torch.stack(vals)


def jit_correct_logits(
    base_logits: torch.Tensor,
    query_features: torch.Tensor,
    memory_keys: torch.Tensor,
    memory_logits: torch.Tensor,
    memory_k: int,
    beta: float,
    temperature: float,
    method: str,
) -> torch.Tensor:
    retrieved = topk_retrieval_logits(query_features, memory_keys, memory_logits, memory_k, temperature)
    if method == "retrieval_only":
        return retrieved
    if method == "jit_residual":
        return centered_logits(base_logits) + beta * centered_logits(retrieved)
    if method == "jit_blend":
        return (1.0 - beta) * centered_logits(base_logits) + beta * centered_logits(retrieved)
    raise ValueError(f"Unknown correction method: {method}")


def jit_gated_logits(base_logits: torch.Tensor, retrieved_logits: torch.Tensor, beta: float, gate_margin: float) -> torch.Tensor:
    base_probs = base_logits.softmax(dim=-1)
    retrieved_probs = retrieved_logits.softmax(dim=-1)
    base_conf, base_pred = base_probs.max(dim=-1)
    retrieved_conf, retrieved_pred = retrieved_probs.max(dim=-1)
    compatible = retrieved_pred == base_pred
    confident_override = retrieved_conf > (base_conf + gate_margin)
    gate = (compatible | confident_override).float().unsqueeze(-1)
    return centered_logits(base_logits) + gate * beta * centered_logits(retrieved_logits)


def prediction_margin(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = logits.softmax(dim=-1)
    top2 = probs.topk(min(2, probs.shape[-1]), dim=-1)
    conf = top2.values[:, 0]
    pred = top2.indices[:, 0]
    if top2.values.shape[-1] == 1:
        margin = conf
    else:
        margin = top2.values[:, 0] - top2.values[:, 1]
    return conf, margin, pred


def source_reliable_indices(
    train_idx: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    pairs: list[tuple[int, int]],
    memory_filter: str,
    min_confidence: float,
    min_margin: float,
    require_stable_transition: bool,
) -> torch.Tensor:
    if memory_filter == "all":
        return train_idx
    confidence, margin, pred = prediction_margin(logits)
    reliable = pred.eq(labels)
    if memory_filter == "source_reliable":
        reliable &= confidence.ge(min_confidence) & margin.ge(min_margin)
        if require_stable_transition:
            stable = torch.zeros_like(reliable)
            train_set = set(int(i) for i in train_idx.tolist())
            for left, right in pairs:
                if left in train_set and right in train_set and bool(reliable[left]) and bool(reliable[right]):
                    stable[left] = True
                    stable[right] = True
            reliable &= stable
    selected = train_idx[reliable[train_idx]]
    if len(selected) == 0:
        return train_idx[pred[train_idx].eq(labels[train_idx])]
    return selected


def jit_reliable_logits(
    base_logits: torch.Tensor,
    retrieved_logits: torch.Tensor,
    sim_margin: torch.Tensor,
    action_entropy: torch.Tensor,
    beta: float,
    min_sim_margin: float,
    max_entropy: float,
    base_uncertain_margin: float,
    override_margin: float,
) -> torch.Tensor:
    _, base_margin, base_pred = prediction_margin(base_logits)
    _, retrieved_margin, retrieved_pred = prediction_margin(retrieved_logits)
    retrieval_reliable = (sim_margin >= min_sim_margin) & (action_entropy <= max_entropy)
    compatible = retrieved_pred == base_pred
    useful_override = (base_margin <= base_uncertain_margin) & (retrieved_margin >= base_margin + override_margin)
    gate = (retrieval_reliable & (compatible | useful_override)).float().unsqueeze(-1)
    return centered_logits(base_logits) + gate * beta * centered_logits(retrieved_logits)


def jit_selective_logits(
    base_logits: torch.Tensor,
    retrieved_logits: torch.Tensor,
    sim_margin: torch.Tensor,
    action_entropy: torch.Tensor,
    beta: float,
    min_sim_margin: float,
    max_entropy: float,
    base_uncertain_margin: float,
    source_margin_floor: float,
    override_margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, base_margin, base_pred = prediction_margin(base_logits)
    _, retrieved_margin, retrieved_pred = prediction_margin(retrieved_logits)
    retrieval_reliable = (sim_margin >= min_sim_margin) & (action_entropy <= max_entropy)
    disagreement = retrieved_pred != base_pred
    base_uncertain = base_margin <= base_uncertain_margin
    source_confident = retrieved_margin >= source_margin_floor
    stronger_than_base = retrieved_margin >= base_margin + override_margin
    gate = retrieval_reliable & disagreement & base_uncertain & source_confident & stronger_than_base
    gate_f = gate.float().unsqueeze(-1)
    corrected = centered_logits(base_logits) + gate_f * beta * centered_logits(retrieved_logits)
    return corrected, gate


def jit_counterfactual_logits(
    base_logits: torch.Tensor,
    retrieved_logits: torch.Tensor,
    sim_margin: torch.Tensor,
    action_entropy: torch.Tensor,
    base_stability: torch.Tensor,
    beta: float,
    min_sim_margin: float,
    max_entropy: float,
    base_uncertain_margin: float,
    source_margin_floor: float,
    override_margin: float,
    max_base_stability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, base_margin, base_pred = prediction_margin(base_logits)
    _, retrieved_margin, retrieved_pred = prediction_margin(retrieved_logits)
    retrieval_reliable = (sim_margin >= min_sim_margin) & (action_entropy <= max_entropy)
    disagreement = retrieved_pred != base_pred
    base_uncertain = base_margin <= base_uncertain_margin
    base_unstable = base_stability <= max_base_stability
    source_confident = retrieved_margin >= source_margin_floor
    stronger_than_base = retrieved_margin >= base_margin + override_margin
    gate = retrieval_reliable & disagreement & base_uncertain & base_unstable & source_confident & stronger_than_base
    gate_f = gate.float().unsqueeze(-1)
    corrected = centered_logits(base_logits) + gate_f * beta * centered_logits(retrieved_logits)
    return corrected, gate


def correction_diagnostics(
    target_base_pred: torch.Tensor,
    target_pred: torch.Tensor,
    source_teacher_pred: torch.Tensor,
    labels: torch.Tensor,
    gate: torch.Tensor | None,
) -> dict[str, float | int]:
    changed = target_pred != target_base_pred
    improved = target_base_pred.ne(labels) & target_pred.eq(labels)
    harmed = target_base_pred.eq(labels) & target_pred.ne(labels)
    source_correct = source_teacher_pred.eq(labels)
    target_base_correct = target_base_pred.eq(labels)
    qa = target_base_correct & source_correct
    qb = target_base_correct & ~source_correct
    qc = ~target_base_correct & source_correct
    qd = ~target_base_correct & ~source_correct

    def rate(mask: torch.Tensor) -> float:
        if len(mask) == 0:
            return 0.0
        return mask.float().mean().item()

    def conditional_rate(event: torch.Tensor, condition: torch.Tensor) -> float:
        denom = int(condition.sum().item())
        if denom == 0:
            return 0.0
        return (event & condition).float().sum().item() / denom

    return {
        "gate_open_rate": rate(gate) if gate is not None else rate(changed),
        "changed_rate": rate(changed),
        "improved_rate": rate(improved),
        "harmed_rate": rate(harmed),
        "quadrant_a_count": int(qa.sum().item()),
        "quadrant_b_count": int(qb.sum().item()),
        "quadrant_c_count": int(qc.sum().item()),
        "quadrant_d_count": int(qd.sum().item()),
        "quadrant_b_harmed_rate": conditional_rate(harmed, qb),
        "quadrant_c_improved_rate": conditional_rate(improved, qc),
    }


def candidate_score(
    logits: torch.Tensor,
    labels: torch.Tensor,
    idx: torch.Tensor,
    selection_metric: str,
) -> float:
    pred = logits.argmax(dim=-1)
    if selection_metric == "source_val":
        return accuracy(pred, labels, idx)
    probs = logits.softmax(dim=-1)
    confidence = probs.max(dim=-1).values.mean().item()
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean().item()
    return confidence - 0.02 * entropy


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    source = load_hidden_tensor(args.source_tensor)
    target = load_hidden_tensor(args.target_tensor)
    rows = load_rows(args.decision_jsonl) if args.decision_jsonl else []
    labels = source["y"]
    if len(labels) != len(target["y"]) or not torch.equal(labels, target["y"]):
        raise ValueError("source and target tensors must be paired with identical labels for evaluation")
    if args.split_json:
        train_idx, val_idx, split_metadata = eval_protocol.load_split(args.split_json, len(labels))
    else:
        train_idx, val_idx, split_metadata = eval_protocol.split_indices_for_protocol(
            len(labels), args.val_frac, args.seed, rows, args.split_unit
        )
    if args.write_split_json:
        eval_protocol.save_split(args.write_split_json, train_idx, val_idx, split_metadata)
    source_logits = load_source_logits(args.source_policy, source["h"], int(labels.max().item()) + 1, device)
    if source_logits is None:
        raise ValueError("source logits are required")

    anchor_idx = select_anchors(args.strategy, source, train_idx, labels, source_logits, args.k, args.seed)
    transition_anchors = select_transition_anchor_pairs(source, train_idx, labels, source_logits, args.k)
    source_features = build_features(source, anchor_idx, args.normalization, args.feature_mode, transition_anchors)
    raw_target_features = build_features(target, anchor_idx, args.normalization, args.feature_mode, transition_anchors)
    target_features, calibration_params = calibrate_target_features(
        source_features, raw_target_features, anchor_idx, args.target_calibration, args.calibration_ridge
    )

    relative_policy = train_source_relative_policy(source_features, target_features, labels, train_idx, val_idx, args, device)
    source_base_logits = predict_model(relative_policy, source_features, device)
    target_base_logits = predict_model(relative_policy, target_features, device)
    source_base_pred = source_base_logits.argmax(dim=-1)
    target_base_pred = target_base_logits.argmax(dim=-1)
    source_teacher_pred = source_logits.argmax(dim=-1)
    source_base_stability = torch.ones_like(labels, dtype=torch.float)
    target_base_stability = torch.ones_like(labels, dtype=torch.float)
    if "jit_counterfactual" in args.methods:
        source_base_stability = perturbation_stability(
            relative_policy,
            source_features,
            source_base_pred,
            device,
            args.consistency_trials,
            args.consistency_drop_rate,
            args.seed + 101,
        )
        target_base_stability = perturbation_stability(
            relative_policy,
            target_features,
            target_base_pred,
            device,
            args.consistency_trials,
            args.consistency_drop_rate,
            args.seed + 202,
        )
    val_set = {int(i) for i in val_idx.tolist()}
    pairs = (
        eval_protocol.transition_pairs(rows, val_idx)
        if rows and len(rows) == len(labels)
        else [(left, right) for left, right in build_transition_pairs_from_tensor(source) if left in val_set and right in val_set]
    )

    results: list[BindingResult] = []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_trans_acc, base_trans_agree = transition_metrics(target_base_pred, source_base_pred, labels, pairs)
    results.append(
        BindingResult(
            target=args.target_name,
            method="anchor_relative_base",
            memory_mode="none",
            k=args.k,
            memory_k=0,
            beta=0.0,
            gate_margin=0.0,
            temperature=0.0,
            source_val_accuracy=accuracy(source_base_pred, labels, val_idx),
            source_full_accuracy=accuracy(source_base_pred, labels),
            target_full_accuracy=accuracy(target_base_pred, labels),
            target_val_accuracy=accuracy(target_base_pred, labels, val_idx),
            source_target_agreement=accuracy(target_base_pred, source_base_pred),
            source_target_transition_agreement=base_trans_agree,
            target_transition_accuracy=base_trans_acc,
            target_labels_used=0,
            target_gradient_steps=0,
            source_online_queries=0,
            memory_size=0,
            memory_filter="all",
        )
    )

    for memory_mode in args.memory_modes:
        for memory_filter in args.memory_filters:
            memory_idx = source_reliable_indices(
                train_idx,
                source_logits,
                labels,
                pairs,
                memory_filter,
                args.memory_min_confidence,
                args.memory_min_margin,
                args.memory_require_stable_transition,
            )
            if memory_mode == "raw":
                memory_keys = source_features[memory_idx]
                memory_logits = source_logits[memory_idx]
            else:
                memory_keys, memory_logits = build_capsules(source_features, source_logits, labels, memory_idx, per_action=8)
            best_local: tuple[float, dict[str, Any]] | None = None
            for memory_k in args.memory_ks:
                for temperature in args.temperatures:
                    source_retrieved, source_sim_margin, source_action_entropy = topk_retrieval_with_stats(
                        source_features, memory_keys, memory_logits, memory_k, temperature
                    )
                    for method in args.methods:
                        beta_values = [1.0] if method == "retrieval_only" else args.betas
                        for beta in beta_values:
                            gate_margins = args.gate_margins if method == "jit_gated" else [0.0]
                            if method == "jit_reliable":
                                gate_margins = args.gate_margins
                            sim_margins = args.reliability_sim_margins if method == "jit_reliable" else [0.0]
                            max_entropies = args.reliability_max_entropies if method == "jit_reliable" else [1.0]
                            base_margins = args.reliability_base_margins if method == "jit_reliable" else [1.0]
                            source_margins = [0.0]
                            if method in {"jit_selective", "jit_counterfactual"}:
                                gate_margins = [0.0]
                                sim_margins = args.selective_sim_margins
                                max_entropies = args.selective_max_entropies
                                base_margins = args.selective_base_margins
                                source_margins = args.selective_source_margins
                                override_margins = args.selective_override_margins
                            else:
                                override_margins = [0.0]
                            stability_values = args.consistency_max_stabilities if method == "jit_counterfactual" else [1.0]
                            for gate_margin in gate_margins:
                                for min_sim_margin in sim_margins:
                                    for max_entropy in max_entropies:
                                        for base_uncertain_margin in base_margins:
                                            for source_margin_floor in source_margins:
                                                for override_margin in override_margins:
                                                    for max_base_stability in stability_values:
                                                        source_gate = None
                                                        if method == "retrieval_only":
                                                            source_corrected = source_retrieved
                                                        elif method == "jit_residual":
                                                            source_corrected = centered_logits(source_base_logits) + beta * centered_logits(source_retrieved)
                                                        elif method == "jit_blend":
                                                            source_corrected = (1.0 - beta) * centered_logits(source_base_logits) + beta * centered_logits(source_retrieved)
                                                        elif method == "jit_gated":
                                                            source_corrected = jit_gated_logits(source_base_logits, source_retrieved, beta, gate_margin)
                                                        elif method == "jit_reliable":
                                                            source_corrected = jit_reliable_logits(
                                                                source_base_logits,
                                                                source_retrieved,
                                                                source_sim_margin,
                                                                source_action_entropy,
                                                                beta,
                                                                min_sim_margin,
                                                                max_entropy,
                                                                base_uncertain_margin,
                                                                gate_margin,
                                                            )
                                                        elif method == "jit_selective":
                                                            source_corrected, source_gate = jit_selective_logits(
                                                                source_base_logits,
                                                                source_retrieved,
                                                                source_sim_margin,
                                                                source_action_entropy,
                                                                beta,
                                                                min_sim_margin,
                                                                max_entropy,
                                                                base_uncertain_margin,
                                                                source_margin_floor,
                                                                override_margin,
                                                            )
                                                        elif method == "jit_counterfactual":
                                                            source_corrected, source_gate = jit_counterfactual_logits(
                                                                source_base_logits,
                                                                source_retrieved,
                                                                source_sim_margin,
                                                                source_action_entropy,
                                                                source_base_stability,
                                                                beta,
                                                                min_sim_margin,
                                                                max_entropy,
                                                                base_uncertain_margin,
                                                                source_margin_floor,
                                                                override_margin,
                                                                max_base_stability,
                                                            )
                                                        else:
                                                            raise ValueError(f"Unknown correction method: {method}")
                                                        score = candidate_score(source_corrected, labels, val_idx, args.selection_metric)
                                                        if source_gate is not None:
                                                            score -= 0.001 * abs(float(source_gate.float().mean().item()) - 0.1)
                                                        payload = {
                                                            "method": method,
                                                            "memory_filter": memory_filter,
                                                            "memory_k": memory_k,
                                                            "temperature": temperature,
                                                            "beta": beta,
                                                            "gate_margin": gate_margin,
                                                            "min_sim_margin": min_sim_margin,
                                                            "max_entropy": max_entropy,
                                                            "base_uncertain_margin": base_uncertain_margin,
                                                            "source_margin_floor": source_margin_floor,
                                                            "override_margin": override_margin,
                                                            "max_base_stability": max_base_stability,
                                                            "source_corrected": source_corrected,
                                                            "memory_size": int(memory_keys.shape[0]),
                                                        }
                                                        if best_local is None or score > best_local[0]:
                                                            best_local = (score, payload)
            if best_local is None:
                continue
            choice = best_local[1]
            target_retrieved, target_sim_margin, target_action_entropy = topk_retrieval_with_stats(
                target_features,
                memory_keys,
                memory_logits,
                int(choice["memory_k"]),
                float(choice["temperature"]),
            )
            target_gate = None
            if str(choice["method"]) == "retrieval_only":
                target_corrected = target_retrieved
            elif str(choice["method"]) == "jit_residual":
                target_corrected = centered_logits(target_base_logits) + float(choice["beta"]) * centered_logits(target_retrieved)
            elif str(choice["method"]) == "jit_blend":
                target_corrected = (1.0 - float(choice["beta"])) * centered_logits(target_base_logits) + float(choice["beta"]) * centered_logits(target_retrieved)
            elif str(choice["method"]) == "jit_gated":
                target_corrected = jit_gated_logits(target_base_logits, target_retrieved, float(choice["beta"]), float(choice["gate_margin"]))
            elif str(choice["method"]) == "jit_reliable":
                target_corrected = jit_reliable_logits(
                    target_base_logits,
                    target_retrieved,
                    target_sim_margin,
                    target_action_entropy,
                    float(choice["beta"]),
                    float(choice.get("min_sim_margin", 0.0)),
                    float(choice.get("max_entropy", 1.0)),
                    float(choice.get("base_uncertain_margin", 1.0)),
                    float(choice["gate_margin"]),
                )
            elif str(choice["method"]) == "jit_selective":
                target_corrected, target_gate = jit_selective_logits(
                    target_base_logits,
                    target_retrieved,
                    target_sim_margin,
                    target_action_entropy,
                    float(choice["beta"]),
                    float(choice.get("min_sim_margin", 0.0)),
                    float(choice.get("max_entropy", 1.0)),
                    float(choice.get("base_uncertain_margin", 1.0)),
                    float(choice.get("source_margin_floor", 0.0)),
                    float(choice.get("override_margin", 0.0)),
                )
            elif str(choice["method"]) == "jit_counterfactual":
                target_corrected, target_gate = jit_counterfactual_logits(
                    target_base_logits,
                    target_retrieved,
                    target_sim_margin,
                    target_action_entropy,
                    target_base_stability,
                    float(choice["beta"]),
                    float(choice.get("min_sim_margin", 0.0)),
                    float(choice.get("max_entropy", 1.0)),
                    float(choice.get("base_uncertain_margin", 1.0)),
                    float(choice.get("source_margin_floor", 0.0)),
                    float(choice.get("override_margin", 0.0)),
                    float(choice.get("max_base_stability", 1.0)),
                )
            else:
                raise ValueError(f"Unknown correction method: {choice['method']}")
            source_corrected = choice["source_corrected"]
            source_pred = source_corrected.argmax(dim=-1)
            target_pred = target_corrected.argmax(dim=-1)
            trans_acc, trans_agree = transition_metrics(target_pred, source_pred, labels, pairs)
            diag = correction_diagnostics(target_base_pred, target_pred, source_teacher_pred, labels, target_gate)
            results.append(
                BindingResult(
                    target=args.target_name,
                    method=str(choice["method"]),
                    memory_mode=memory_mode,
                    k=args.k,
                    memory_k=int(choice["memory_k"]),
                    beta=float(choice["beta"]),
                    gate_margin=float(choice.get("gate_margin", 0.0)),
                    temperature=float(choice["temperature"]),
                    source_val_accuracy=accuracy(source_pred, labels, val_idx),
                    source_full_accuracy=accuracy(source_pred, labels),
                    target_full_accuracy=accuracy(target_pred, labels),
                    target_val_accuracy=accuracy(target_pred, labels, val_idx),
                    source_target_agreement=accuracy(target_pred, source_pred),
                    source_target_transition_agreement=trans_agree,
                    target_transition_accuracy=trans_acc,
                    target_labels_used=0,
                    target_gradient_steps=0,
                    source_online_queries=0,
                    memory_size=int(choice["memory_size"]),
                    memory_filter=str(choice["memory_filter"]),
                    gate_open_rate=float(diag["gate_open_rate"]),
                    changed_rate=float(diag["changed_rate"]),
                    improved_rate=float(diag["improved_rate"]),
                    harmed_rate=float(diag["harmed_rate"]),
                    quadrant_a_count=int(diag["quadrant_a_count"]),
                    quadrant_b_count=int(diag["quadrant_b_count"]),
                    quadrant_c_count=int(diag["quadrant_c_count"]),
                    quadrant_d_count=int(diag["quadrant_d_count"]),
                    quadrant_b_harmed_rate=float(diag["quadrant_b_harmed_rate"]),
                    quadrant_c_improved_rate=float(diag["quadrant_c_improved_rate"]),
                )
            )

    metadata = {
        "source_tensor": args.source_tensor,
        "target_tensor": args.target_tensor,
        "source_policy": args.source_policy,
        "strategy": args.strategy,
        "k": args.k,
        "feature_mode": args.feature_mode,
        "normalization": args.normalization,
        "target_calibration": args.target_calibration,
        "target_calibration_params": calibration_params,
        "split": split_metadata,
        "selection_metric": args.selection_metric,
        "memory_filters": args.memory_filters,
        "memory_min_confidence": args.memory_min_confidence,
        "memory_min_margin": args.memory_min_margin,
        "memory_require_stable_transition": args.memory_require_stable_transition,
        "methods": args.methods,
        "consistency_trials": args.consistency_trials,
        "consistency_drop_rate": args.consistency_drop_rate,
        "consistency_max_stabilities": args.consistency_max_stabilities,
        "target_labels_used": 0,
        "target_gradient_steps": 0,
        "source_online_queries": 0,
    }
    obj = {"metadata": metadata, "results": [asdict(r) for r in results]}
    (out_dir / "jit_policy_binding_results.json").write_text(json.dumps(obj, indent=2), encoding="utf-8")

    lines = [
        "# Just-in-Time Non-Parametric Policy Binding",
        "",
        f"- Target: `{args.target_name}`",
        f"- Target labels used: `0`",
        f"- Target gradient steps: `0`",
        f"- Source online queries: `0`",
        f"- Anchor-only calibration: `{args.target_calibration}`",
        f"- Split: `{split_metadata.get('split_unit', 'unknown')}`",
        f"- Train samples: {split_metadata.get('num_train_samples', 'unknown')}",
        f"- Held-out samples: {split_metadata.get('num_val_samples', 'unknown')}",
        "",
        "| Method | Memory | Filter | Memory Size | kNN | beta | gate | temp | Source Held-out | Target Held-out | Target Full Diagnostic | Policy Agreement | Transition Agreement | Gate Open | Changed | Improved | Harmed | B Harm | C Improve |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.method} | {r.memory_mode} | {r.memory_filter} | {r.memory_size} | {r.memory_k} | {r.beta:.2f} | {r.gate_margin:.2f} | {r.temperature:.2f} | "
            f"{r.source_val_accuracy:.2%} | {r.target_val_accuracy:.2%} | {r.target_full_accuracy:.2%} | "
            f"{r.source_target_agreement:.2%} | {r.source_target_transition_agreement:.2%} | "
            f"{r.gate_open_rate:.2%} | {r.changed_rate:.2%} | {r.improved_rate:.2%} | {r.harmed_rate:.2%} | "
            f"{r.quadrant_b_harmed_rate:.2%} | {r.quadrant_c_improved_rate:.2%} |"
        )
    best = max(results, key=lambda r: (r.target_val_accuracy, r.source_target_agreement))
    lines.extend(
        [
            "",
            "## Best",
            "",
            (
                f"`{best.method}` with `{best.memory_mode}` memory reached "
                f"{best.target_val_accuracy:.2%} held-out target action accuracy and "
                f"{best.source_target_agreement:.2%} policy agreement."
            ),
        ]
    )
    (out_dir / "jit_policy_binding_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(asdict(best), indent=2))


if __name__ == "__main__":
    main()
