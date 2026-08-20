from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from universal_agent_policy.adapters import AdapterConfig, PolicyHead, build_adapter
from universal_agent_policy.policy.action_space import AbstractAction, ActionSpace, SlotActionSpace


def dtype_from_name(name: str) -> str | torch.dtype:
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def row_to_text(tokenizer: Any, row: dict[str, Any]) -> str:
    if "messages" in row:
        return tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
    return str(row["prompt"])


class FrozenAdapterPolicy(nn.Module):
    def __init__(self, adapter: nn.Module, head: PolicyHead) -> None:
        super().__init__()
        self.adapter = adapter
        self.head = head
        for param in self.head.parameters():
            param.requires_grad = False

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(self.adapter(h))


@dataclass
class PolicyPrediction:
    action: AbstractAction
    logits: torch.Tensor
    probabilities: torch.Tensor
    z: torch.Tensor


def action_space_from_checkpoint(checkpoint: dict[str, Any]) -> ActionSpace | SlotActionSpace:
    action_names = checkpoint.get("action_names")
    if action_names:
        return ActionSpace(list(action_names))
    return SlotActionSpace.from_num_actions(int(checkpoint["num_actions"]))


class FrozenPolicyRuntime:
    def __init__(self, model: FrozenAdapterPolicy, action_space: ActionSpace | SlotActionSpace, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.action_space = action_space

    @classmethod
    def from_target_adapter(
        cls,
        source_policy_path: str,
        target_adapter_path: str,
        device: str = "cpu",
    ) -> "FrozenPolicyRuntime":
        source = torch.load(source_policy_path, map_location="cpu")
        target = torch.load(target_adapter_path, map_location="cpu")
        z_dim = int(source["z_dim"])
        num_actions = int(source["num_actions"])

        head = PolicyHead(z_dim, num_actions)
        head_state = {
            key.removeprefix("policy_head."): value
            for key, value in source["model_state"].items()
            if key.startswith("policy_head.")
        }
        head.load_state_dict(head_state)

        adapter = build_adapter(AdapterConfig(**target["adapter_config"]))
        adapter.load_state_dict(target["adapter_state"])
        action_space = action_space_from_checkpoint(source)
        return cls(FrozenAdapterPolicy(adapter, head), action_space, device)

    @classmethod
    def from_source_policy(cls, source_policy_path: str, device: str = "cpu") -> "FrozenPolicyRuntime":
        source = torch.load(source_policy_path, map_location="cpu")
        adapter = build_adapter(AdapterConfig(**source["adapter_config"]))
        head = PolicyHead(int(source["z_dim"]), int(source["num_actions"]))
        adapter.load_state_dict(
            {
                key.removeprefix("adapter."): value
                for key, value in source["model_state"].items()
                if key.startswith("adapter.")
            }
        )
        head.load_state_dict(
            {
                key.removeprefix("policy_head."): value
                for key, value in source["model_state"].items()
                if key.startswith("policy_head.")
            }
        )
        model = FrozenAdapterPolicy(adapter, head)
        action_space = action_space_from_checkpoint(source)
        return cls(model, action_space, device)

    def predict_from_hidden(self, h: torch.Tensor) -> PolicyPrediction:
        if h.ndim == 1:
            h = h.unsqueeze(0)
        with torch.no_grad():
            h = h.to(self.device).float()
            z = self.model.adapter(h)
            logits = self.model.head(z)
            probs = torch.softmax(logits, dim=-1)
        action_id = int(probs.argmax(dim=-1).item())
        return PolicyPrediction(
            action=self.action_space.action_from_id(action_id),
            logits=logits.squeeze(0).cpu(),
            probabilities=probs.squeeze(0).cpu(),
            z=z.squeeze(0).cpu(),
        )


class HFHiddenStateExtractor:
    def __init__(self, model_id: str, layer: int, device: str = "cpu", torch_dtype: str = "float32") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype_from_name(torch_dtype),
            device_map=device if device != "cpu" else None,
        )
        if device == "cpu":
            self.model.to("cpu")
        self.model.eval()
        self.layer = layer

    @property
    def device(self) -> torch.device:
        return self.model.device

    def extract(self, row: dict[str, Any]) -> torch.Tensor:
        text = row_to_text(self.tokenizer, row)
        encoded = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**encoded, output_hidden_states=True, use_cache=False)
        return outputs.hidden_states[self.layer][0, -1].detach().float().cpu()
