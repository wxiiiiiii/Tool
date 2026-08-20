from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .adapters import AdapterConfig, AdapterPolicy, PolicyHead, build_adapter
from .data import load_hidden_tensor, split_indices
from .metrics import accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-tensor", required=True)
    parser.add_argument("--target-tensor", required=True)
    parser.add_argument("--source-policy", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_obj = load_hidden_tensor(args.source_tensor)
    target_obj = load_hidden_tensor(args.target_tensor)
    if source_obj["h"].shape[0] != target_obj["h"].shape[0]:
        raise ValueError("Source and target tensors must be paired and ordered identically")

    checkpoint = torch.load(args.source_policy, map_location="cpu")
    cfg = checkpoint["adapter_config"]
    source_adapter = build_adapter(AdapterConfig(**cfg))
    head = PolicyHead(checkpoint["z_dim"], checkpoint["num_actions"])
    model = AdapterPolicy(source_adapter, head)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    train_idx, val_idx = split_indices(len(target_obj["y"]), args.val_frac, args.seed)
    with torch.no_grad():
        z_source = source_adapter(source_obj["h"])

    x = target_obj["h"][train_idx]
    y = z_source[train_idx]
    ones = torch.ones(x.shape[0], 1)
    x_aug = torch.cat([x, ones], dim=1)
    eye = torch.eye(x_aug.shape[1])
    coef = torch.linalg.solve(x_aug.T @ x_aug + args.ridge * eye, x_aug.T @ y)

    x_val = target_obj["h"][val_idx]
    x_val_aug = torch.cat([x_val, torch.ones(x_val.shape[0], 1)], dim=1)
    z_val = torch.nn.functional.normalize(x_val_aug @ coef, dim=-1)
    logits = head(z_val)
    val_acc = accuracy(logits, target_obj["y"][val_idx])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "linear_coef": coef,
            "source_policy": str(Path(args.source_policy)),
            "ridge": args.ridge,
            "val_accuracy": val_acc,
        },
        out_dir / "procrustes_adapter.pt",
    )
    print(json.dumps({"val_accuracy": val_acc}, indent=2))


if __name__ == "__main__":
    main()

