from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .adapters import AdapterConfig, AdapterPolicy, PolicyHead, build_adapter
from .data import load_hidden_tensor, split_indices, subset_dataset
from .metrics import accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--z-dim", type=int, default=256)
    parser.add_argument("--adapter", choices=["linear", "low_rank_mlp"], default="low_rank_mlp")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    logits_all, y_all = [], []
    with torch.no_grad():
        for h, y in loader:
            logits_all.append(model(h.to(device)).cpu())
            y_all.append(y)
    return accuracy(torch.cat(logits_all), torch.cat(y_all))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    obj = load_hidden_tensor(args.tensor)
    train_idx, val_idx = split_indices(len(obj["y"]), args.val_frac, args.seed)
    train_ds = subset_dataset(obj, train_idx)
    val_ds = subset_dataset(obj, val_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    input_dim = obj["h"].shape[1]
    action_names = obj.get("action_names")
    num_actions = len(action_names) if action_names else int(obj["y"].max().item()) + 1
    adapter_config = AdapterConfig(args.adapter, input_dim, args.z_dim, args.rank)
    model = AdapterPolicy(build_adapter(adapter_config), PolicyHead(args.z_dim, num_actions)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = -1.0
    for epoch in range(args.epochs):
        model.train()
        for h, y in train_loader:
            loss = nn.functional.cross_entropy(model(h.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": best_state,
        "adapter_config": adapter_config.__dict__,
        "z_dim": args.z_dim,
        "num_actions": num_actions,
        "action_names": action_names,
        "best_val_accuracy": best_val,
    }
    torch.save(checkpoint, out_dir / "source_policy.pt")
    print(json.dumps({"best_val_accuracy": best_val, "num_actions": num_actions}, indent=2))


if __name__ == "__main__":
    main()
