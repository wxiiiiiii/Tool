from __future__ import annotations

import torch


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()


def per_class_accuracy(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> dict[int, float]:
    pred = logits.argmax(dim=-1)
    scores: dict[int, float] = {}
    for cls in range(num_classes):
        mask = y == cls
        if mask.any():
            scores[cls] = (pred[mask] == y[mask]).float().mean().item()
    return scores

