from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class LinearAdapter(nn.Module):
    def __init__(self, input_dim: int, z_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, z_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(h), dim=-1)


class LowRankMLPAdapter(nn.Module):
    def __init__(self, input_dim: int, z_dim: int, rank: int = 64) -> None:
        super().__init__()
        self.down = nn.Linear(input_dim, rank)
        self.up = nn.Linear(rank, z_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.up(F.gelu(self.down(h)))
        return F.normalize(z, dim=-1)


class PolicyHead(nn.Module):
    def __init__(self, z_dim: int, num_actions: int) -> None:
        super().__init__()
        self.linear = nn.Linear(z_dim, num_actions)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)


class AdapterPolicy(nn.Module):
    def __init__(self, adapter: nn.Module, policy_head: PolicyHead) -> None:
        super().__init__()
        self.adapter = adapter
        self.policy_head = policy_head

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.policy_head(self.adapter(h))


@dataclass(frozen=True)
class AdapterConfig:
    kind: str
    input_dim: int
    z_dim: int
    rank: int = 64


def build_adapter(config: AdapterConfig) -> nn.Module:
    if config.kind == "linear":
        return LinearAdapter(config.input_dim, config.z_dim)
    if config.kind == "low_rank_mlp":
        return LowRankMLPAdapter(config.input_dim, config.z_dim, config.rank)
    raise ValueError(f"Unknown adapter kind: {config.kind}")

