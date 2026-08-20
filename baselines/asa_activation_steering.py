from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from baselines.tool_policy_utils import (
    action_names_from_rows,
    add_reference_agreement,
    is_tool_action,
    load_rows,
    prediction_records,
    predictions_by_sample_id,
    save_json,
)
from universal_agent_policy.data import load_hidden_tensor, split_indices


@dataclass
class ASACalibration:
    method: str
    calibration_fraction: float
    seed: int
    alpha_tool: float
    alpha_domain: float
    target_trainable_parameters: int
    stored_parameters: int
    uses_target_action_labels: bool
    uses_source_policy: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ASA-style target-side activation steering baseline over saved target hidden states. "
            "This is a paper-faithful offline proxy: construct tool/domain/action directions and "
            "a lightweight router from target calibration states, then predict actions without "
            "training or updating the target backbone."
        )
    )
    parser.add_argument("--decision-jsonl", required=True)
    parser.add_argument("--target-tensor", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference-predictions", help="Optional source-policy prediction file for agreement metrics")
    parser.add_argument("--calibration-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--alpha-tool", type=float, default=0.75)
    parser.add_argument("--alpha-domain", type=float, default=0.25)
    parser.add_argument("--normalize", action="store_true")
    return parser.parse_args()


def _mean_or_zero(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.numel() == 0:
        return torch.zeros(dim)
    return x.float().mean(dim=0)


def _standardize(h: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (h.float() - mean) / std.clamp_min(1e-6)


def fit_controller(
    h: torch.Tensor,
    labels: torch.Tensor,
    rows: list[dict[str, Any]],
    action_names: list[str],
    calibration_idx: torch.Tensor,
    alpha_tool: float,
    alpha_domain: float,
) -> dict[str, Any]:
    h_cal = h[calibration_idx].float()
    y_cal = labels[calibration_idx].long()
    mean = h_cal.mean(dim=0)
    std = h_cal.std(dim=0).clamp_min(1e-6)
    h_cal = _standardize(h_cal, mean, std)
    dim = h.shape[1]
    global_mean = h_cal.mean(dim=0)

    action_centroids = []
    for action_id in range(len(action_names)):
        mask = y_cal == action_id
        centroid = _mean_or_zero(h_cal[mask], dim)
        if not bool(mask.any()):
            centroid = global_mean.clone()
        action_centroids.append(centroid)
    action_centroids_t = torch.stack(action_centroids)

    tool_mask = torch.tensor([is_tool_action(action_names[int(label)]) for label in y_cal], dtype=torch.bool)
    tool_direction = _mean_or_zero(h_cal[tool_mask], dim) - _mean_or_zero(h_cal[~tool_mask], dim)
    if tool_direction.norm() > 0:
        tool_direction = F.normalize(tool_direction, dim=0)

    domain_centroids: dict[str, torch.Tensor] = {}
    for domain in sorted({str(row.get("domain", "unknown")) for row in rows}):
        local_positions = [
            pos for pos, row_idx in enumerate(calibration_idx.tolist()) if str(rows[row_idx].get("domain", "unknown")) == domain
        ]
        if local_positions:
            domain_centroids[domain] = h_cal[torch.tensor(local_positions)].mean(dim=0) - global_mean
        else:
            domain_centroids[domain] = torch.zeros(dim)

    return {
        "mean": mean,
        "std": std,
        "action_centroids": action_centroids_t,
        "tool_direction": tool_direction,
        "domain_centroids": domain_centroids,
        "alpha_tool": alpha_tool,
        "alpha_domain": alpha_domain,
    }


def predict(
    h: torch.Tensor,
    rows: list[dict[str, Any]],
    action_names: list[str],
    controller: dict[str, Any],
    normalize: bool,
) -> list[str]:
    x = _standardize(h, controller["mean"], controller["std"])
    centroids = controller["action_centroids"]
    if normalize:
        x_for_router = F.normalize(x, dim=-1)
        c_for_router = F.normalize(centroids, dim=-1)
    else:
        x_for_router = x
        c_for_router = centroids

    first_ids = torch.cdist(x_for_router, c_for_router).argmin(dim=-1)
    steered = x.clone()
    tool_dir = controller["tool_direction"]
    for idx, action_id in enumerate(first_ids.tolist()):
        action_name = action_names[int(action_id)]
        sign = 1.0 if is_tool_action(action_name) else -1.0
        steered[idx] = steered[idx] + float(controller["alpha_tool"]) * sign * tool_dir
        domain = str(rows[idx].get("domain", "unknown"))
        steered[idx] = steered[idx] + float(controller["alpha_domain"]) * controller["domain_centroids"].get(
            domain, torch.zeros_like(tool_dir)
        )

    if normalize:
        steered_for_cls = F.normalize(steered, dim=-1)
        centroids_for_cls = F.normalize(centroids, dim=-1)
    else:
        steered_for_cls = steered
        centroids_for_cls = centroids
    pred_ids = torch.cdist(steered_for_cls, centroids_for_cls).argmin(dim=-1)
    return [action_names[int(idx)] for idx in pred_ids.tolist()]


def main() -> None:
    args = parse_args()
    rows = load_rows(args.decision_jsonl)
    tensor = load_hidden_tensor(args.target_tensor)
    if len(rows) != tensor["h"].shape[0]:
        raise ValueError("decision rows and target tensor must have the same length")
    action_names = action_names_from_rows(rows)
    labels = tensor["y"].long()
    calibration_idx, _ = split_indices(len(labels), 1.0 - args.calibration_frac, args.seed)
    controller = fit_controller(
        tensor["h"],
        labels,
        rows,
        action_names,
        calibration_idx,
        args.alpha_tool,
        args.alpha_domain,
    )
    predictions = predict(tensor["h"], rows, action_names, controller, args.normalize)
    stored_params = (
        int(controller["action_centroids"].numel())
        + int(controller["tool_direction"].numel())
        + sum(int(value.numel()) for value in controller["domain_centroids"].values())
    )
    calibration = ASACalibration(
        method="asa_target_activation_steering",
        calibration_fraction=args.calibration_frac,
        seed=args.seed,
        alpha_tool=args.alpha_tool,
        alpha_domain=args.alpha_domain,
        target_trainable_parameters=0,
        stored_parameters=stored_params,
        uses_target_action_labels=True,
        uses_source_policy=False,
    )
    artifact = prediction_records(rows, predictions, method="asa_target_activation_steering", metadata=asdict(calibration))
    for idx, record in enumerate(artifact["records"]):
        record["calibration_state"] = int(idx) in set(calibration_idx.tolist())
    if args.reference_predictions:
        add_reference_agreement(artifact, rows, predictions_by_sample_id(args.reference_predictions), "source")
    artifact["controller"] = {
        "note": "Tensor values are not serialized; rerun this script to rebuild the ASA controller.",
        "action_names": action_names,
        "calibration_indices": calibration_idx.tolist(),
    }
    save_json(artifact, args.out)
    print(json.dumps(artifact["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
