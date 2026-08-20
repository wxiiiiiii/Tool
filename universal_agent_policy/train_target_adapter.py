from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .adapters import AdapterConfig, PolicyHead, build_adapter
from .data import load_hidden_tensor, split_indices, subset_dataset
from .metrics import accuracy


class FrozenHeadAdapter(nn.Module):
    def __init__(self, adapter: nn.Module, head: PolicyHead) -> None:
        super().__init__()
        self.adapter = adapter
        self.head = head
        for param in self.head.parameters():
            param.requires_grad = False

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(self.adapter(h))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--source-policy", required=True)
    parser.add_argument("--out-dir", required=True)
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
    source = torch.load(args.source_policy, map_location="cpu")
    train_idx, val_idx = split_indices(len(obj["y"]), args.val_frac, args.seed)
    train_loader = DataLoader(subset_dataset(obj, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(subset_dataset(obj, val_idx), batch_size=args.batch_size)

    input_dim = obj["h"].shape[1]
    z_dim = int(source["z_dim"])
    num_actions = int(source["num_actions"])

    head = PolicyHead(z_dim, num_actions)
    head_state = {
        key.removeprefix("policy_head."): value
        for key, value in source["model_state"].items()
        if key.startswith("policy_head.")
    }
    head.load_state_dict(head_state)

    adapter_config = AdapterConfig(args.adapter, input_dim, z_dim, args.rank)
    model = FrozenHeadAdapter(build_adapter(adapter_config), head).to(device)
    opt = torch.optim.AdamW(model.adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
            best_state = {key: value.detach().cpu() for key, value in model.adapter.state_dict().items()}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter_state": best_state,
            "adapter_config": adapter_config.__dict__,
            "source_policy": str(Path(args.source_policy)),
            "best_val_accuracy": best_val,
        },
        out_dir / "target_adapter.pt",
    )
    print(json.dumps({"best_val_accuracy": best_val}, indent=2))


if __name__ == "__main__":
    main()
