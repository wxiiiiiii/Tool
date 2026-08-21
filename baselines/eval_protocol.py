from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def row_trajectory_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row.get("domain", "unknown")), int(row.get("task_id", -1)), int(row.get("trial", 0)))


def row_task_key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row.get("domain", "unknown")), int(row.get("task_id", -1)))


def _decision_split(n: int, val_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    val_size = max(1, int(n * val_frac))
    return perm[val_size:], perm[:val_size]


def grouped_split_indices(
    rows: list[dict[str, Any]],
    val_frac: float,
    seed: int,
    split_unit: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if split_unit not in {"trajectory", "task"}:
        raise ValueError(f"Unsupported grouped split unit: {split_unit}")
    if not rows:
        raise ValueError(f"{split_unit} split requires decision rows")

    key_fn = row_trajectory_key if split_unit == "trajectory" else row_task_key
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[key_fn(row)].append(idx)

    keys = sorted(groups)
    if len(keys) < 2:
        train_idx, val_idx = _decision_split(len(rows), val_frac, seed)
        return train_idx, val_idx, {
            "split_unit": "decision_fallback_single_group",
            "requested_split_unit": split_unit,
            "seed": seed,
            "val_frac": val_frac,
            "num_groups": len(keys),
            "num_train_samples": int(train_idx.numel()),
            "num_val_samples": int(val_idx.numel()),
        }
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(keys), generator=generator).tolist()
    val_group_size = min(len(keys) - 1, max(1, int(len(keys) * val_frac)))
    val_keys = {keys[i] for i in perm[:val_group_size]}

    train_idx: list[int] = []
    val_idx: list[int] = []
    for key in keys:
        if key in val_keys:
            val_idx.extend(groups[key])
        else:
            train_idx.extend(groups[key])

    metadata = {
        "split_unit": split_unit,
        "seed": seed,
        "val_frac": val_frac,
        "num_groups": len(keys),
        "num_train_groups": len(keys) - len(val_keys),
        "num_val_groups": len(val_keys),
        "num_train_samples": len(train_idx),
        "num_val_samples": len(val_idx),
    }
    return torch.tensor(train_idx, dtype=torch.long), torch.tensor(val_idx, dtype=torch.long), metadata


def split_indices_for_protocol(
    n: int,
    val_frac: float,
    seed: int,
    rows: list[dict[str, Any]] | None = None,
    split_unit: str = "trajectory",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if split_unit == "decision":
        train_idx, val_idx = _decision_split(n, val_frac, seed)
        return train_idx, val_idx, {
            "split_unit": "decision",
            "seed": seed,
            "val_frac": val_frac,
            "num_train_samples": int(train_idx.numel()),
            "num_val_samples": int(val_idx.numel()),
        }

    if rows and len(rows) == n:
        return grouped_split_indices(rows, val_frac, seed, split_unit)

    train_idx, val_idx = _decision_split(n, val_frac, seed)
    return train_idx, val_idx, {
        "split_unit": "decision_fallback_no_rows",
        "requested_split_unit": split_unit,
        "seed": seed,
        "val_frac": val_frac,
        "num_train_samples": int(train_idx.numel()),
        "num_val_samples": int(val_idx.numel()),
    }


def save_split(path: str | Path, train_idx: torch.Tensor, val_idx: torch.Tensor, metadata: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "train_idx": [int(i) for i in train_idx.cpu().tolist()],
        "val_idx": [int(i) for i in val_idx.cpu().tolist()],
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_split(path: str | Path, n: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    train_idx = torch.tensor(obj["train_idx"], dtype=torch.long)
    val_idx = torch.tensor(obj["val_idx"], dtype=torch.long)
    all_idx = set(train_idx.tolist()) | set(val_idx.tolist())
    if min(all_idx, default=0) < 0 or max(all_idx, default=-1) >= n:
        raise ValueError("Loaded split contains out-of-range indices")
    if set(train_idx.tolist()) & set(val_idx.tolist()):
        raise ValueError("Loaded split train/val indices overlap")
    return train_idx, val_idx, obj.get("metadata", {})


def accuracy_on_indices(pred: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor) -> float:
    idx = idx.cpu()
    if idx.numel() == 0:
        return float("nan")
    return float((pred[idx].cpu() == labels[idx].cpu()).float().mean().item())


def macro_action_accuracy(pred: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor, num_actions: int) -> float:
    idx = idx.cpu()
    per_action: list[float] = []
    for action_id in range(num_actions):
        action_idx = idx[labels[idx].cpu() == action_id]
        if action_idx.numel() > 0:
            per_action.append(accuracy_on_indices(pred, labels, action_idx))
    if not per_action:
        return float("nan")
    return float(sum(per_action) / len(per_action))


def transition_pairs(rows: list[dict[str, Any]], allowed_idx: torch.Tensor | None = None) -> list[tuple[int, int]]:
    allowed: set[int] | None = None
    if allowed_idx is not None:
        allowed = {int(i) for i in allowed_idx.cpu().tolist()}
    groups: dict[tuple[str, int, int], list[tuple[int, int]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        if allowed is not None and idx not in allowed:
            continue
        groups[row_trajectory_key(row)].append((int(row.get("turn_index", idx)), idx))
    pairs: list[tuple[int, int]] = []
    for items in groups.values():
        ordered = [idx for _, idx in sorted(items)]
        pairs.extend((left, right) for left, right in zip(ordered, ordered[1:]))
    return pairs


def transition_state_accuracy(
    pred: torch.Tensor,
    labels: torch.Tensor,
    rows: list[dict[str, Any]],
    idx: torch.Tensor,
) -> float:
    pairs = transition_pairs(rows, idx)
    if not pairs:
        return float("nan")
    correct = 0
    for left, right in pairs:
        correct += int(pred[left].cpu() == labels[left].cpu() and pred[right].cpu() == labels[right].cpu())
    return correct / len(pairs)
