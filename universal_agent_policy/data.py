from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class HiddenStateDataset(Dataset):
    def __init__(self, h: torch.Tensor, y: torch.Tensor) -> None:
        if h.ndim != 2:
            raise ValueError(f"Expected h to be rank-2, got shape {tuple(h.shape)}")
        if y.ndim != 1:
            raise ValueError(f"Expected y to be rank-1, got shape {tuple(y.shape)}")
        if h.shape[0] != y.shape[0]:
            raise ValueError("h and y must contain the same number of samples")
        self.h = h.float()
        self.y = y.long()

    def __len__(self) -> int:
        return self.h.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.h[idx], self.y[idx]


def load_hidden_tensor(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
    elif path.suffix == ".npz":
        npz = np.load(path, allow_pickle=True)
        obj = {key: npz[key] for key in npz.files}
    else:
        raise ValueError(f"Unsupported tensor file suffix: {path.suffix}")

    h = obj.get("h", obj.get("hidden_states"))
    y = obj.get("y", obj.get("labels"))
    if h is None or y is None:
        raise KeyError("Tensor file must contain h/y or hidden_states/labels")

    obj["h"] = torch.as_tensor(h).float()
    obj["y"] = torch.as_tensor(y).long()
    return obj


def split_indices(n: int, val_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    val_size = max(1, int(n * val_frac))
    return perm[val_size:], perm[:val_size]


def subset_dataset(obj: dict[str, Any], indices: torch.Tensor) -> HiddenStateDataset:
    return HiddenStateDataset(obj["h"][indices], obj["y"][indices])

