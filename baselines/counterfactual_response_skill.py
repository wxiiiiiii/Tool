from __future__ import annotations

import argparse
import json
import math
import random
import re
import gc
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from baselines import eval_protocol
from baselines.hidden_policy_baselines import RidgeProjection, load_source_parts


OBS_TYPES = ["success", "failure", "empty", "conflict"]
DELTA_OBS_TYPES = ["failure", "empty", "conflict"]
BRANCHES = ["finish", "recover", "verify", "acquire", "execute"]
BRANCH_TO_ID = {name: idx for idx, name in enumerate(BRANCHES)}

GENERIC_OBSERVATIONS = {
    "success": {
        "status": "success",
        "message": "The tool completed successfully and the requested state change is now committed.",
        "result": {"ok": True, "updated": True},
    },
    "failure": {
        "status": "error",
        "error": "tool_execution_failed",
        "message": "The tool failed and no state change was committed.",
    },
    "empty": {
        "status": "success",
        "message": "The tool returned no matching records.",
        "result": [],
    },
    "conflict": {
        "status": "conflict",
        "message": "The tool result conflicts with the requested operation and requires verification.",
        "result": {"ok": False, "requires_verification": True},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Counterfactual policy-response Skill transfer. Skills are discovered from "
            "same-prefix Source policy responses to success/failure/empty/conflict observations."
        )
    )
    parser.add_argument("--decision-jsonl", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--source-policy", required=True)
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--source-model-id", required=True)
    parser.add_argument("--target-model-id", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-skills", type=int, default=8)
    parser.add_argument("--skill-dim", type=int, default=32)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--response-rank", type=int, default=8)
    parser.add_argument("--binder-rank", type=int, default=32)
    parser.add_argument("--binder-epochs", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--critical-impact-threshold", type=float, default=0.5)
    parser.add_argument("--branch-margin-threshold", type=float, default=0.5)
    parser.add_argument("--critical-top-quantile", type=float, default=0.75)
    parser.add_argument("--write-pair-top-quantile", type=float, default=0.8)
    parser.add_argument("--write-branch-strength", type=float, default=12.0)
    parser.add_argument("--write-pair-calibration-margin", type=float, default=1.0)
    parser.add_argument("--write-pair-max-calibration", type=float, default=8.0)
    parser.add_argument("--branch-aux-weight", type=float, default=0.05)
    parser.add_argument("--obs-variants", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-cos", type=float, default=0.25)
    parser.add_argument("--lambda-rank", type=float, default=0.5)
    parser.add_argument("--rank-margin", type=float, default=0.5)
    parser.add_argument("--skill-temperature", type=float, default=0.1)
    parser.add_argument("--fingerprint", choices=["mean_ref", "success_ref"], default="mean_ref")
    parser.add_argument("--gate-static-margin", type=float, default=0.08)
    parser.add_argument("--max-train-prefixes", type=int, default=-1)
    parser.add_argument("--max-eval-prefixes", type=int, default=-1)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def branch_for_action(action: str) -> str:
    if action in {"ANSWER", "STOP"}:
        return "finish"
    if action in {"ASK_USER", "THINK", "TRANSFER"}:
        return "recover"
    if action == "VERIFY":
        return "verify"
    if action.startswith(("ACT_RETRIEVE", "ACT_SEARCH", "ACT_COMPUTE")):
        return "acquire"
    if action.startswith(("ACT_UPDATE", "ACT_SEND")):
        return "execute"
    return "recover"


def action_to_branch_id(action: str) -> int:
    return BRANCH_TO_ID[branch_for_action(action)]


def expected_branches(observation_type: str, tool_kind: str) -> set[str]:
    if observation_type == "success":
        if tool_kind == "write":
            return {"finish"}
        if tool_kind in {"retrieve", "search", "compute"}:
            return {"verify", "execute", "finish", "acquire"}
        return {"finish", "verify"}
    if observation_type == "failure":
        return {"recover", "verify", "acquire"}
    if observation_type == "empty":
        return {"recover", "verify", "acquire"}
    if observation_type == "conflict":
        return {"verify", "recover"}
    return set(BRANCHES)


def action_logits_to_branch_logits(action_logits: torch.Tensor, action_names: list[str], normalize_size: bool = True) -> torch.Tensor:
    original_shape = action_logits.shape[:-1]
    flat = action_logits.reshape(-1, action_logits.shape[-1])
    out = flat.new_full((flat.shape[0], len(BRANCHES)), -1e9)
    for branch_id in range(len(BRANCHES)):
        ids = [idx for idx, action in enumerate(action_names) if action_to_branch_id(action) == branch_id]
        if ids:
            idx = torch.tensor(ids, dtype=torch.long, device=flat.device)
            score = torch.logsumexp(flat.index_select(1, idx), dim=1)
            if normalize_size:
                score = score - math.log(len(ids))
            out[:, branch_id] = score
    out = out.reshape(*original_shape, len(BRANCHES))
    return out


def branch_id_set(names: set[str]) -> list[int]:
    return [BRANCH_TO_ID[name] for name in names if name in BRANCH_TO_ID]


def branch_good_bad(obs_type: str, tool_kind: str) -> tuple[list[int], list[int]]:
    if obs_type == "success":
        good = ["finish"]
        bad = ["recover"]
        if tool_kind != "write":
            good = sorted(expected_branches(obs_type, tool_kind))
            bad = ["recover"]
    elif obs_type in {"failure", "empty"}:
        good = ["recover", "verify", "acquire"]
        bad = ["finish", "execute"]
    elif obs_type == "conflict":
        good = ["recover", "verify"]
        bad = ["execute", "finish"]
    else:
        good = sorted(expected_branches(obs_type, tool_kind))
        bad = [name for name in BRANCHES if name not in good]
    return branch_id_set(set(good)), branch_id_set(set(bad))


def branch_margin(branch_logits: torch.Tensor, good: list[int], bad: list[int]) -> torch.Tensor:
    good_idx = torch.tensor(good, dtype=torch.long, device=branch_logits.device)
    bad_idx = torch.tensor(bad, dtype=torch.long, device=branch_logits.device)
    good_score = torch.logsumexp(branch_logits.index_select(-1, good_idx), dim=-1)
    bad_score = torch.logsumexp(branch_logits.index_select(-1, bad_idx), dim=-1)
    return good_score - bad_score


def branch_group_score(branch_logits: torch.Tensor, branch_names: list[str] | tuple[str, ...]) -> torch.Tensor:
    ids = branch_id_set(set(branch_names))
    idx = torch.tensor(ids, dtype=torch.long, device=branch_logits.device)
    return torch.logsumexp(branch_logits.index_select(-1, idx), dim=-1)


def write_pair_loss(success_branch: torch.Tensor, failure_branch: torch.Tensor, margin: float) -> torch.Tensor:
    success_finish = branch_group_score(success_branch, ["finish"])
    success_recover = branch_group_score(success_branch, ["recover", "verify"])
    failure_recover = branch_group_score(failure_branch, ["recover", "verify"])
    failure_finish = branch_group_score(failure_branch, ["finish"])
    loss_success = F.relu(success_branch.new_tensor(float(margin)) - success_finish + success_recover)
    loss_failure = F.relu(failure_branch.new_tensor(float(margin)) - failure_recover + failure_finish)
    return (loss_success + loss_failure).mean()


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pct(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{100.0 * value:.2f}%"


def conversation_bounds(prompt: str) -> tuple[int, int] | None:
    marker = "Conversation state before the next assistant decision:\n"
    end_marker = "\n\nNext policy action:"
    start = prompt.find(marker)
    end = prompt.find(end_marker, start + len(marker))
    if start < 0 or end < 0:
        return None
    return start + len(marker), end


def conversation_lines(prompt: str) -> list[tuple[int, int, str]]:
    bounds = conversation_bounds(prompt)
    if bounds is None:
        return []
    start, end = bounds
    context = prompt[start:end]
    out = []
    cursor = 0
    for line in context.splitlines(keepends=True):
        stripped = line.strip()
        if stripped:
            absolute_start = start + cursor
            out.append((absolute_start, absolute_start + len(line.rstrip("\n")), stripped))
        cursor += len(line)
    return out


def last_context_event_is_tool_observation(prompt: str) -> bool:
    lines = conversation_lines(prompt)
    if len(lines) < 2:
        return False
    last = lines[-1][2].lower()
    previous = lines[-2][2].lower()
    return last.startswith("tool ") and "assistant tool_call:" in previous


def last_tool_observation_line(prompt: str) -> tuple[int, int, str] | None:
    lines = conversation_lines(prompt)
    if not lines or not lines[-1][2].lower().startswith("tool "):
        return None
    return lines[-1]


def last_tool_name(prompt: str) -> str:
    lines = conversation_lines(prompt)
    for _, _, line in reversed(lines):
        match = re.search(r"assistant tool_call:\s*([A-Za-z0-9_]+)\(", line)
        if match:
            return match.group(1)
    return "unknown_tool"


def tool_family(tool_name: str) -> str:
    lower = tool_name.lower()
    if lower.startswith(("book_", "cancel_", "exchange_", "modify_", "return_", "send_", "update_")):
        return "write"
    if lower.startswith(("find_", "get_", "list_")):
        return "retrieve"
    if lower.startswith("search_"):
        return "search"
    if lower == "calculate":
        return "compute"
    if lower == "think":
        return "think"
    if "transfer" in lower:
        return "transfer"
    return "other"


def infer_observation_type(text: str) -> str:
    lower = text.lower()
    if any(term in lower for term in ("conflict", "inconsistent", "mismatch", "requires verification")):
        return "conflict"
    if any(term in lower for term in ("error", "failed", "failure", "exception", "invalid", "denied", "cannot", "can't")):
        return "failure"
    if any(term in lower for term in ("not found", "no matching", "no result", "empty", "none found", "[]")):
        return "empty"
    return "success"


def build_observation_bank(rows: list[dict[str, Any]], train_idx: list[int]) -> dict[tuple[str, str], list[str]]:
    bank: dict[tuple[str, str], list[str]] = defaultdict(list)
    family_bank: dict[tuple[str, str], list[str]] = defaultdict(list)
    for idx in train_idx:
        prompt = str(rows[idx].get("prompt") or "")
        if not last_context_event_is_tool_observation(prompt):
            continue
        info = last_tool_observation_line(prompt)
        if info is None:
            continue
        obs = info[2]
        obs_type = infer_observation_type(obs)
        tool = last_tool_name(prompt)
        bank[(tool, obs_type)].append(obs)
        family_bank[(tool_family(tool), obs_type)].append(obs)
    for (family, obs_type), values in family_bank.items():
        bank[(f"__family__:{family}", obs_type)].extend(values)
    return dict(bank)


def generic_observation_line(original_tool_line: str, obs_type: str) -> str:
    prefix = original_tool_line.split(":", 1)[0] if ":" in original_tool_line else "tool counterfactual"
    payload = json.dumps(GENERIC_OBSERVATIONS[obs_type], ensure_ascii=False, sort_keys=True)
    return f"{prefix}: {payload}"


def sample_schema_compatible_observation(
    prompt: str,
    obs_type: str,
    bank: dict[tuple[str, str], list[str]],
    rng: random.Random,
) -> str:
    info = last_tool_observation_line(prompt)
    if info is None:
        raise ValueError("prompt does not end with a tool observation")
    tool = last_tool_name(prompt)
    candidates = bank.get((tool, obs_type), [])
    if not candidates:
        candidates = bank.get((f"__family__:{tool_family(tool)}", obs_type), [])
    if candidates:
        return rng.choice(candidates)
    return generic_observation_line(info[2], obs_type)


def replace_last_tool_observation_with_line(prompt: str, replacement: str) -> str:
    info = last_tool_observation_line(prompt)
    if info is None:
        raise ValueError("prompt does not end with a tool observation")
    start, end, _ = info
    return prompt[:start] + replacement + prompt[end:]


def build_pre_observation_prompt(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt") or "")
    if not last_context_event_is_tool_observation(prompt):
        raise ValueError("pre-observation prompt requires assistant tool_call followed by tool observation")
    info = last_tool_observation_line(prompt)
    if info is None:
        raise ValueError("prompt does not end with a tool observation")
    start, end, _ = info
    next_marker = "\n\nNext policy action:"
    next_start = prompt.find(next_marker, end)
    if next_start < 0:
        next_start = end
    return prompt[:start].rstrip() + prompt[next_start:]


def build_cf_bundle(
    row: dict[str, Any],
    bank: dict[tuple[str, str], list[str]],
    rng: random.Random,
    obs_variants: int = 1,
) -> dict[str, list[str]]:
    prompt = str(row.get("prompt") or "")
    if not last_context_event_is_tool_observation(prompt):
        raise ValueError("counterfactual bundle requires a post-tool branch point")
    bundle = {}
    for obs_type in OBS_TYPES:
        variants = []
        for _ in range(max(1, int(obs_variants))):
            obs_line = sample_schema_compatible_observation(prompt, obs_type, bank, rng)
            variants.append(replace_last_tool_observation_with_line(prompt, obs_line))
        bundle[obs_type] = variants
    return bundle


def dtype_from_name(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


class BackboneRunner:
    def __init__(self, model_id: str, layer: int, max_input_tokens: int, torch_dtype: str) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=dtype_from_name(torch_dtype),
        )
        self.model.eval()
        self.layer = layer
        self.max_input_tokens = max_input_tokens

    @torch.no_grad()
    def extract(self, prompts: list[str], batch_size: int) -> torch.Tensor:
        outputs = []
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
            )
            encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
            result = self.model(**encoded, output_hidden_states=True, use_cache=False)
            hidden = result.hidden_states[self.layer]
            last_idx = encoded["attention_mask"].sum(dim=1) - 1
            states = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_idx]
            outputs.append(states.detach().float().cpu())
        return torch.cat(outputs, dim=0)


def normalize_response_delta(x: torch.Tensor) -> torch.Tensor:
    centered = x - x.mean(dim=-1, keepdim=True)
    return centered / centered.std(dim=-1, keepdim=True).clamp_min(1e-6)


def build_response_fingerprint(logits_by_obs: dict[str, torch.Tensor], mode: str = "mean_ref") -> torch.Tensor:
    if mode == "mean_ref":
        stacked = torch.stack([logits_by_obs[obs_type] for obs_type in OBS_TYPES], dim=0)
        ref = stacked.mean(dim=0)
        return torch.cat([normalize_response_delta(logits_by_obs[obs_type] - ref) for obs_type in OBS_TYPES], dim=-1)
    ref = logits_by_obs["success"]
    parts = []
    for obs_type in DELTA_OBS_TYPES:
        parts.append(normalize_response_delta(logits_by_obs[obs_type] - ref))
    return torch.cat(parts, dim=-1)


def build_branch_logits_by_obs(
    logits_by_obs: dict[str, torch.Tensor],
    action_names: list[str],
) -> dict[str, torch.Tensor]:
    return {obs_type: action_logits_to_branch_logits(logits, action_names) for obs_type, logits in logits_by_obs.items()}


def build_branch_response_fingerprint(
    logits_by_obs: dict[str, torch.Tensor],
    action_names: list[str],
) -> torch.Tensor:
    branch_logits = build_branch_logits_by_obs(logits_by_obs, action_names)
    stacked = torch.stack([branch_logits[obs_type] for obs_type in OBS_TYPES], dim=0)
    ref = stacked.mean(dim=0)
    parts = []
    for obs_type in OBS_TYPES:
        parts.append(normalize_response_delta(branch_logits[obs_type] - ref))
    return torch.cat(parts, dim=-1)


def branch_response_delta_mean_ref(
    logits_by_obs: dict[str, torch.Tensor],
    obs_type: str,
    action_names: list[str],
) -> torch.Tensor:
    branch_logits = build_branch_logits_by_obs(logits_by_obs, action_names)
    mean_logits = torch.stack([branch_logits[obs] for obs in OBS_TYPES], dim=0).mean(dim=0)
    return branch_logits[obs_type] - mean_logits


def response_delta(logits_by_obs: dict[str, torch.Tensor], obs_type: str) -> torch.Tensor:
    return logits_by_obs[obs_type] - logits_by_obs["success"]


def response_delta_mean_ref(logits_by_obs: dict[str, torch.Tensor], obs_type: str) -> torch.Tensor:
    mean_logits = torch.stack([logits_by_obs[obs] for obs in OBS_TYPES], dim=0).mean(dim=0)
    return logits_by_obs[obs_type] - mean_logits


def fit_kmeans(features: torch.Tensor, k: int, iters: int = 50) -> torch.Tensor:
    if features.shape[0] < k:
        raise ValueError(f"Need at least {k} fingerprints, got {features.shape[0]}")
    features = F.normalize(features.float(), dim=1)
    init = torch.linspace(0, features.shape[0] - 1, steps=k).round().long()
    centroids = features.index_select(0, init).clone()
    for _ in range(iters):
        sims = features @ F.normalize(centroids, dim=1).T
        labels = sims.argmax(dim=1)
        new = centroids.clone()
        for cluster_id in range(k):
            mask = labels == cluster_id
            if bool(mask.any()):
                new[cluster_id] = features[mask].mean(dim=0)
        new = F.normalize(new, dim=1)
        if torch.allclose(new, centroids, atol=1e-5):
            break
        centroids = new
    return centroids


def bind_target_skill(fingerprint: torch.Tensor, centroids: torch.Tensor, temperature: float) -> torch.Tensor:
    sims = F.normalize(fingerprint.float(), dim=-1) @ F.normalize(centroids.float(), dim=-1).T
    return torch.softmax(sims / float(temperature), dim=-1)


@dataclass
class PolicySkill:
    skill_id: int
    response_prototype: torch.Tensor
    response_variance: torch.Tensor | None
    embedding: torch.Tensor
    transition_table: torch.Tensor


class ResponseGapAdapter(nn.Module):
    def __init__(self, hidden_dim: int, skill_dim: int, num_obs: int, num_actions: int, rank: int = 64) -> None:
        super().__init__()
        in_dim = hidden_dim + skill_dim + num_obs
        self.down = nn.Linear(in_dim, rank)
        self.up = nn.Linear(rank, num_actions)

    def forward(self, delta_h: torch.Tensor, skill_embedding: torch.Tensor, obs_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([delta_h, skill_embedding, obs_onehot], dim=-1)
        return self.up(F.gelu(self.down(x)))


class ResponseSkillLibrary(nn.Module):
    def __init__(
        self,
        prototypes: torch.Tensor,
        hidden_dim: int,
        skill_dim: int,
        num_actions: int,
        num_obs_types: int,
        rank: int = 64,
    ) -> None:
        super().__init__()
        self.register_buffer("prototypes", prototypes.float())
        self.skill_embedding = nn.Embedding(prototypes.shape[0], skill_dim)
        self.response_adapter = ResponseGapAdapter(
            hidden_dim=hidden_dim,
            skill_dim=skill_dim,
            num_obs=num_obs_types,
            num_actions=num_actions,
            rank=rank,
        )
        self.register_buffer("transition", torch.zeros(prototypes.shape[0], num_obs_types, prototypes.shape[0]))

    def mixed_skill_embedding(self, skill_probs: torch.Tensor) -> torch.Tensor:
        weights = self.skill_embedding.weight
        return skill_probs @ weights

    def predict_gap(self, delta_h: torch.Tensor, skill_probs: torch.Tensor, obs_onehot: torch.Tensor) -> torch.Tensor:
        return self.response_adapter(delta_h, self.mixed_skill_embedding(skill_probs), obs_onehot)


class BranchResponseOperator(nn.Module):
    def __init__(self, hidden_dim: int, skill_dim: int, num_obs_types: int, num_branches: int, rank: int = 64) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_dim + skill_dim + num_obs_types, rank)
        self.up = nn.Linear(rank, num_branches)

    def forward(self, delta_h_raw: torch.Tensor, skill_embedding: torch.Tensor, obs_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([delta_h_raw, skill_embedding, obs_onehot], dim=-1)
        return self.up(F.gelu(self.down(x)))


class BranchSkillLibrary(nn.Module):
    def __init__(
        self,
        prototypes: torch.Tensor,
        hidden_dim: int,
        skill_dim: int,
        num_obs_types: int,
        rank: int = 64,
    ) -> None:
        super().__init__()
        self.register_buffer("prototypes", prototypes.float())
        self.skill_embedding = nn.Embedding(prototypes.shape[0], skill_dim)
        self.branch_operator = BranchResponseOperator(
            hidden_dim=hidden_dim,
            skill_dim=skill_dim,
            num_obs_types=num_obs_types,
            num_branches=len(BRANCHES),
            rank=rank,
        )

    def mixed_skill_embedding(self, skill_probs: torch.Tensor) -> torch.Tensor:
        return skill_probs @ self.skill_embedding.weight

    def predict_branch_residual(
        self,
        delta_h_raw: torch.Tensor,
        skill_probs: torch.Tensor,
        obs_onehot: torch.Tensor,
    ) -> torch.Tensor:
        return self.branch_operator(delta_h_raw, self.mixed_skill_embedding(skill_probs), obs_onehot)


class FunctionalSkillBinder(nn.Module):
    def __init__(self, z_dim: int, num_skills: int, rank: int = 32) -> None:
        super().__init__()
        self.down = nn.Linear(z_dim, rank)
        self.up = nn.Linear(rank, num_skills)

    def forward(self, z_pre: torch.Tensor) -> torch.Tensor:
        return self.up(F.gelu(self.down(z_pre)))


def center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=-1, keepdim=True)


def action_indices(action_names: list[str], predicate) -> torch.Tensor:
    idxs = [idx for idx, name in enumerate(action_names) if predicate(name)]
    if not idxs:
        idxs = list(range(len(action_names)))
    return torch.tensor(idxs, dtype=torch.long)


def branch_ranking_loss(logits: torch.Tensor, action_names: list[str], obs_type: str, tool_kind_name: str, margin: float) -> torch.Tensor:
    tool_kind = tool_family(tool_kind_name)
    if obs_type == "success" and tool_kind == "write":
        good = action_indices(action_names, lambda a: a in {"STOP", "ANSWER"})
        bad = action_indices(action_names, lambda a: a in {"ASK_USER", "TRANSFER", "THINK"} or a.startswith(("ACT_RETRIEVE", "ACT_SEARCH")))
    elif obs_type in {"failure", "empty"}:
        good = action_indices(
            action_names,
            lambda a: a in {"VERIFY", "ASK_USER", "THINK", "TRANSFER"} or a.startswith(("ACT_RETRIEVE", "ACT_SEARCH")),
        )
        bad = action_indices(action_names, lambda a: a in {"STOP", "ANSWER"} or a.startswith(("ACT_UPDATE", "ACT_SEND")))
    elif obs_type == "conflict":
        good = action_indices(action_names, lambda a: a in {"VERIFY", "ASK_USER", "THINK"})
        bad = action_indices(action_names, lambda a: a.startswith(("ACT_UPDATE", "ACT_SEND")))
    else:
        return logits.new_zeros(())
    good_score = logits.index_select(0, good.to(logits.device)).max()
    bad_score = logits.index_select(0, bad.to(logits.device)).max()
    return F.relu(logits.new_tensor(float(margin)) - good_score + bad_score)


def policy_margin(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
    if top.shape[-1] == 1:
        return top[..., 0]
    return top[..., 0] - top[..., 1]


def response_gate(obs_type: str, static_margin: torch.Tensor, threshold: float) -> torch.Tensor:
    if obs_type in {"failure", "empty", "conflict"}:
        return torch.ones_like(static_margin)
    return (static_margin < float(threshold)).float()


def forward_bundles(
    runner: BackboneRunner,
    bundles: list[dict[str, list[str]]],
    projection: nn.Module,
    head: nn.Module,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    hidden_by_obs = {}
    logits_by_obs = {}
    for obs_type in OBS_TYPES:
        prompts = []
        counts = []
        for bundle in bundles:
            variants = bundle[obs_type]
            prompts.extend(variants)
            counts.append(len(variants))
        hidden_flat = runner.extract(prompts, batch_size)
        pieces = []
        cursor = 0
        for count in counts:
            pieces.append(hidden_flat[cursor : cursor + count].mean(dim=0))
            cursor += count
        hidden = torch.stack(pieces, dim=0)
        with torch.no_grad():
            z = projection(hidden.to(device))
            logits = head(z).detach().cpu()
        hidden_by_obs[obs_type] = hidden
        logits_by_obs[obs_type] = logits
    return hidden_by_obs, logits_by_obs


def project_hidden_by_obs(
    hidden_by_obs: dict[str, torch.Tensor],
    projection: nn.Module,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    out = {}
    with torch.no_grad():
        for obs_type, hidden in hidden_by_obs.items():
            out[obs_type] = projection(hidden.to(device)).detach().cpu()
    return out


def forward_pre_observation(
    runner: BackboneRunner,
    rows: list[dict[str, Any]],
    indices: list[int],
    projection: nn.Module,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompts = [build_pre_observation_prompt(rows[idx]) for idx in indices]
    hidden = runner.extract(prompts, batch_size)
    with torch.no_grad():
        z = projection(hidden.to(device)).detach().cpu()
    return hidden, z


def select_valid_indices(rows: list[dict[str, Any]], indices: list[int], max_items: int) -> list[int]:
    selected = [idx for idx in indices if last_context_event_is_tool_observation(str(rows[idx].get("prompt") or ""))]
    if max_items > 0:
        selected = selected[:max_items]
    return selected


def build_bundles(
    rows: list[dict[str, Any]],
    indices: list[int],
    bank: dict[tuple[str, str], list[str]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, list[str]]], list[int]]:
    rng = random.Random(args.seed)
    bundles = [build_cf_bundle(rows[idx], bank, rng, args.obs_variants) for idx in indices]
    return bundles, indices


def source_skill_ids_from_centroids(source_fingerprint: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    sims = F.normalize(source_fingerprint.float(), dim=-1) @ F.normalize(centroids.float(), dim=-1).T
    return sims.argmax(dim=-1)


def fit_response_basis(train_examples: dict[str, Any], rank: int) -> torch.Tensor:
    rows = []
    for obs_type in OBS_TYPES:
        rows.append(response_delta_mean_ref(train_examples["source_logits"], obs_type))
    matrix = torch.cat(rows, dim=0).float()
    matrix = center(matrix)
    _, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    return F.normalize(vh[: min(rank, vh.shape[0])].T.contiguous(), dim=0)


def true_response_gap_mean_ref(examples: dict[str, Any], obs_type: str) -> torch.Tensor:
    source_delta = response_delta_mean_ref(examples["source_logits"], obs_type)
    target_delta = response_delta_mean_ref(examples["target_logits"], obs_type)
    return source_delta - target_delta


def build_response_operator(
    train_examples: dict[str, Any],
    source_skill_ids: torch.Tensor,
    response_basis: torch.Tensor,
    num_skills: int,
) -> torch.Tensor:
    operator = torch.zeros(num_skills, len(OBS_TYPES), response_basis.shape[1], dtype=torch.float32)
    counts = torch.zeros(num_skills, len(OBS_TYPES), dtype=torch.float32)
    global_coords = []
    for obs_id, obs_type in enumerate(OBS_TYPES):
        gap = true_response_gap_mean_ref(train_examples, obs_type).float()
        coords = gap @ response_basis.float()
        global_coords.append(coords)
        for skill_id in range(num_skills):
            mask = source_skill_ids == skill_id
            if bool(mask.any()):
                operator[skill_id, obs_id] = coords[mask].mean(dim=0)
                counts[skill_id, obs_id] = float(mask.sum().item())
    fallback = torch.cat(global_coords, dim=0).mean(dim=0)
    for skill_id in range(num_skills):
        for obs_id in range(len(OBS_TYPES)):
            if counts[skill_id, obs_id] == 0:
                operator[skill_id, obs_id] = fallback
    return operator


def train_functional_skill_binder(
    z_pre: torch.Tensor,
    source_skill_ids: torch.Tensor,
    num_skills: int,
    args: argparse.Namespace,
    device: torch.device,
) -> FunctionalSkillBinder:
    binder = FunctionalSkillBinder(z_pre.shape[1], num_skills, args.binder_rank).to(device)
    z = z_pre.to(device)
    labels = source_skill_ids.to(device)
    optimizer = torch.optim.AdamW(binder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for _ in range(args.binder_epochs):
        logits = binder(z)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return binder.eval()


@torch.no_grad()
def binder_skill_probs(binder: FunctionalSkillBinder, z_pre: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.softmax(binder(z_pre.to(device)).detach().cpu(), dim=-1)


@torch.no_grad()
def prototype_binder_skill_probs(
    train_z_pre: torch.Tensor,
    train_skill_ids: torch.Tensor,
    eval_z_pre: torch.Tensor,
    num_skills: int,
    temperature: float,
) -> torch.Tensor:
    centroids = []
    global_mean = train_z_pre.float().mean(dim=0)
    for skill_id in range(num_skills):
        mask = train_skill_ids == skill_id
        centroids.append(train_z_pre.float()[mask].mean(dim=0) if bool(mask.any()) else global_mean)
    centroid_tensor = torch.stack(centroids, dim=0)
    sims = F.normalize(eval_z_pre.float(), dim=1) @ F.normalize(centroid_tensor.float(), dim=1).T
    return torch.softmax(sims / float(temperature), dim=1)


@torch.no_grad()
def ridge_binder_skill_probs(
    train_z_pre: torch.Tensor,
    train_skill_ids: torch.Tensor,
    eval_z_pre: torch.Tensor,
    num_skills: int,
    ridge: float = 1e-2,
) -> torch.Tensor:
    x = torch.cat([train_z_pre.float(), torch.ones(train_z_pre.shape[0], 1)], dim=1)
    y = one_hot_skill_probs(train_skill_ids, num_skills).float()
    eye = torch.eye(x.shape[1], dtype=x.dtype)
    eye[-1, -1] = 0.0
    coef = torch.linalg.solve(x.T @ x + float(ridge) * eye, x.T @ y)
    eval_x = torch.cat([eval_z_pre.float(), torch.ones(eval_z_pre.shape[0], 1)], dim=1)
    return torch.softmax(eval_x @ coef, dim=1)


def combine_examples(
    indices: list[int],
    bundles: list[dict[str, str]],
    h_s: dict[str, torch.Tensor],
    l_s: dict[str, torch.Tensor],
    h_t: dict[str, torch.Tensor],
    l_t: dict[str, torch.Tensor],
    z_s: dict[str, torch.Tensor] | None = None,
    z_t: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    return {
        "indices": indices,
        "bundles": bundles,
        "source_hidden": h_s,
        "source_z": z_s,
        "target_hidden": h_t,
        "target_z": z_t,
        "source_logits": l_s,
        "target_logits": l_t,
        "source_fingerprint": build_response_fingerprint(l_s),
        "target_fingerprint": build_response_fingerprint(l_t),
    }


def forward_examples(
    indices: list[int],
    bundles: list[dict[str, str]],
    source_runner: BackboneRunner,
    target_runner: BackboneRunner,
    source_projection: nn.Module,
    target_projection: nn.Module,
    head: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    h_s, l_s = forward_bundles(source_runner, bundles, source_projection, head, args.batch_size, device)
    h_t, l_t = forward_bundles(target_runner, bundles, target_projection, head, args.batch_size, device)
    f_s = build_response_fingerprint(l_s, args.fingerprint)
    f_t = build_response_fingerprint(l_t, args.fingerprint)
    return {
        "indices": indices,
        "bundles": bundles,
        "source_hidden": h_s,
        "target_hidden": h_t,
        "source_logits": l_s,
        "target_logits": l_t,
        "source_fingerprint": f_s,
        "target_fingerprint": f_t,
    }


def examples_to_training_tensors(
    examples: dict[str, Any],
    centroids: torch.Tensor,
    temperature: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[int]]:
    f_t = examples["target_fingerprint"]
    skill_probs = bind_target_skill(f_t, centroids, temperature)
    delta_h_items = []
    skill_items = []
    obs_items = []
    gap_items = []
    obs_names = []
    row_positions = []
    for row_pos in range(f_t.shape[0]):
        for obs_id, obs_type in enumerate(DELTA_OBS_TYPES):
            delta_h = examples["target_hidden"][obs_type][row_pos] - examples["target_hidden"]["success"][row_pos]
            source_delta = response_delta({key: value[row_pos] for key, value in examples["source_logits"].items()}, obs_type)
            target_delta = response_delta({key: value[row_pos] for key, value in examples["target_logits"].items()}, obs_type)
            delta_h_items.append(delta_h)
            skill_items.append(skill_probs[row_pos])
            obs_items.append(F.one_hot(torch.tensor(obs_id), num_classes=len(DELTA_OBS_TYPES)).float())
            gap_items.append(source_delta - target_delta)
            obs_names.append(obs_type)
            row_positions.append(row_pos)
    return (
        torch.stack(delta_h_items).to(device),
        torch.stack(skill_items).to(device),
        torch.stack(obs_items).to(device),
        torch.stack(gap_items).to(device),
        obs_names,
        row_positions,
    )


def train_response_gap_adapter(
    rows: list[dict[str, Any]],
    examples: dict[str, Any],
    centroids: torch.Tensor,
    action_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> ResponseSkillLibrary:
    delta_h, skill_probs, obs_onehot, true_gap, obs_names, row_positions = examples_to_training_tensors(
        examples, centroids, args.skill_temperature, device
    )
    library = ResponseSkillLibrary(
        prototypes=centroids,
        hidden_dim=delta_h.shape[1],
        skill_dim=args.skill_dim,
        num_actions=len(action_names),
        num_obs_types=len(DELTA_OBS_TYPES),
        rank=args.rank,
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(library.skill_embedding.parameters()) + list(library.response_adapter.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    order = torch.arange(delta_h.shape[0], device=device)
    for _ in range(args.epochs):
        perm = order[torch.randperm(order.numel(), device=device)]
        for start in range(0, perm.numel(), args.train_batch_size):
            batch = perm[start : start + args.train_batch_size]
            pred_gap = library.predict_gap(delta_h[batch], skill_probs[batch], obs_onehot[batch])
            target_gap = true_gap[batch]
            loss = F.mse_loss(center(pred_gap), center(target_gap))
            loss = loss + float(args.lambda_cos) * (1.0 - F.cosine_similarity(center(pred_gap), center(target_gap), dim=1)).mean()
            if args.lambda_rank > 0:
                rank_losses = []
                for local_pos, global_pos in enumerate(batch.tolist()):
                    obs_type = obs_names[global_pos]
                    row_pos = row_positions[global_pos]
                    row_idx = examples["indices"][row_pos]
                    base_logits = examples["target_logits"][obs_type][row_pos].to(device)
                    corrected = base_logits + pred_gap[local_pos]
                    tool = last_tool_name(str(rows[row_idx].get("prompt") or ""))
                    rank_losses.append(branch_ranking_loss(corrected, action_names, obs_type, tool, args.rank_margin))
                if rank_losses:
                    loss = loss + float(args.lambda_rank) * torch.stack(rank_losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return library.eval()


def broadcast_branch_residual(branch_residual: torch.Tensor, action_names: list[str]) -> torch.Tensor:
    columns = [action_to_branch_id(action) for action in action_names]
    idx = torch.tensor(columns, dtype=torch.long, device=branch_residual.device)
    return branch_residual.index_select(1, idx)


def actions_in_branch(action_names: list[str], branch_id: int) -> list[int]:
    ids = [idx for idx, action in enumerate(action_names) if action_to_branch_id(action) == branch_id]
    return ids or list(range(len(action_names)))


def corrected_branch_logits_from_residual(
    static_logits: torch.Tensor,
    branch_residual: torch.Tensor,
    action_names: list[str],
    alpha: float,
) -> torch.Tensor:
    return action_logits_to_branch_logits(static_logits, action_names) + float(alpha) * branch_residual


def hard_branch_first_logits(
    static_logits: torch.Tensor,
    branch_residual: torch.Tensor,
    action_names: list[str],
    alpha: float,
) -> torch.Tensor:
    corrected_branch = corrected_branch_logits_from_residual(static_logits, branch_residual, action_names, alpha)
    selected = corrected_branch.argmax(dim=1)
    out = static_logits.new_full(static_logits.shape, -1e9)
    for row_pos, branch_id in enumerate(selected.tolist()):
        ids = actions_in_branch(action_names, int(branch_id))
        out[row_pos, ids] = static_logits[row_pos, ids]
    return out


def soft_hierarchical_logits(
    static_logits: torch.Tensor,
    branch_residual: torch.Tensor,
    action_names: list[str],
    alpha: float,
) -> torch.Tensor:
    corrected_branch = corrected_branch_logits_from_residual(static_logits, branch_residual, action_names, alpha)
    branch_log_probs = F.log_softmax(corrected_branch, dim=1)
    out = static_logits.new_full(static_logits.shape, -1e9)
    for branch_id in range(len(BRANCHES)):
        ids = actions_in_branch(action_names, branch_id)
        idx = torch.tensor(ids, dtype=torch.long, device=static_logits.device)
        within = F.log_softmax(static_logits.index_select(1, idx), dim=1)
        out.index_copy_(1, idx, within + branch_log_probs[:, branch_id : branch_id + 1])
    return out


def source_validated_branch_critical_mask(
    rows: list[dict[str, Any]],
    examples: dict[str, Any],
    action_names: list[str],
    threshold: float,
    top_quantile: float = 0.75,
) -> tuple[torch.Tensor, dict[str, float]]:
    source_branches = build_branch_logits_by_obs(examples["source_logits"], action_names)
    target_branches = build_branch_logits_by_obs(examples["target_logits"], action_names)
    success_source = source_branches["success"].argmax(dim=1)
    failure_source = source_branches["failure"].argmax(dim=1)
    success_target = target_branches["success"].argmax(dim=1)
    failure_target = target_branches["failure"].argmax(dim=1)
    source_flip = success_source != failure_source
    target_flip = success_target != failure_target
    source_margin_deltas: list[float] = []
    target_margin_deltas = []
    for row_pos, row_idx in enumerate(examples["indices"]):
        family = tool_family(last_tool_name(str(rows[row_idx].get("prompt") or "")))
        good_s, bad_s = branch_good_bad("success", family)
        good_f, bad_f = branch_good_bad("failure", family)
        s_success_margin = branch_margin(source_branches["success"][row_pos], good_s, bad_s)
        s_failure_margin = branch_margin(source_branches["failure"][row_pos], good_f, bad_f)
        t_success_margin = branch_margin(target_branches["success"][row_pos], good_s, bad_s)
        t_failure_margin = branch_margin(target_branches["failure"][row_pos], good_f, bad_f)
        s_delta = s_failure_margin - s_success_margin
        t_delta = t_failure_margin - t_success_margin
        source_margin_deltas.append(float(s_delta.item()))
        target_margin_deltas.append(float(t_delta.item()))
    abs_delta = torch.tensor([abs(value) for value in source_margin_deltas], dtype=torch.float32)
    if abs_delta.numel():
        q = min(max(float(top_quantile), 0.0), 1.0)
        quantile_threshold = float(torch.quantile(abs_delta, q).item())
        soft_threshold = max(float(threshold), quantile_threshold)
        margin_critical = abs_delta >= soft_threshold
    else:
        soft_threshold = float(threshold)
        margin_critical = torch.zeros_like(source_flip)
    critical = source_flip | margin_critical
    stats = {
        "num_total_prefixes": float(len(examples["indices"])),
        "num_source_validated_critical_prefixes": float(critical.sum().item()),
        "critical_prefix_rate": float(critical.float().mean().item()) if critical.numel() else 0.0,
        "source_branch_flip_rate": float(source_flip.float().mean().item()) if source_flip.numel() else 0.0,
        "target_branch_flip_rate": float(target_flip.float().mean().item()) if target_flip.numel() else 0.0,
        "source_branch_margin_delta": float(torch.tensor(source_margin_deltas).mean().item()) if source_margin_deltas else 0.0,
        "target_branch_margin_delta": float(torch.tensor(target_margin_deltas).mean().item()) if target_margin_deltas else 0.0,
        "source_branch_margin_abs_top_quantile_threshold": soft_threshold,
    }
    return critical, stats


def write_pair_critical_mask(
    rows: list[dict[str, Any]],
    examples: dict[str, Any],
    action_names: list[str],
    top_quantile: float = 0.8,
) -> tuple[torch.Tensor, dict[str, float]]:
    source_branches = build_branch_logits_by_obs(examples["source_logits"], action_names)
    success_source = source_branches["success"].argmax(dim=1)
    failure_source = source_branches["failure"].argmax(dim=1)
    is_write = []
    source_flip = []
    pair_strength = []
    for row_pos, row_idx in enumerate(examples["indices"]):
        family = tool_family(last_tool_name(str(rows[row_idx].get("prompt") or "")))
        write = family == "write"
        is_write.append(write)
        source_flip.append(bool(success_source[row_pos].item() != failure_source[row_pos].item()))
        success_margin = branch_group_score(source_branches["success"][row_pos], ["finish"]) - branch_group_score(
            source_branches["success"][row_pos], ["recover", "verify"]
        )
        failure_margin = branch_group_score(source_branches["failure"][row_pos], ["recover", "verify"]) - branch_group_score(
            source_branches["failure"][row_pos], ["finish"]
        )
        pair_strength.append(float((success_margin + failure_margin).item()))
    is_write_t = torch.tensor(is_write, dtype=torch.bool)
    source_flip_t = torch.tensor(source_flip, dtype=torch.bool)
    strengths = torch.tensor(pair_strength, dtype=torch.float32)
    write_strengths = strengths[is_write_t]
    if write_strengths.numel():
        q = min(max(float(top_quantile), 0.0), 1.0)
        threshold = float(torch.quantile(write_strengths.abs(), q).item())
        high_strength = strengths.abs() >= threshold
    else:
        threshold = 0.0
        high_strength = torch.zeros_like(is_write_t)
    mask = is_write_t & (source_flip_t | high_strength)
    fallback_mask = is_write_t
    if not bool(mask.any()) and bool(fallback_mask.any()):
        mask = fallback_mask
    stats = {
        "num_total_prefixes": float(len(examples["indices"])),
        "num_write_prefixes": float(is_write_t.sum().item()),
        "num_write_pair_critical_prefixes": float(mask.sum().item()),
        "write_pair_critical_rate": float(mask.float().mean().item()) if mask.numel() else 0.0,
        "write_source_flip_rate": float((source_flip_t & is_write_t).float().sum().item() / max(1, int(is_write_t.sum().item()))),
        "write_pair_strength_abs_top_quantile_threshold": threshold,
        "mean_write_pair_strength": float(write_strengths.mean().item()) if write_strengths.numel() else 0.0,
    }
    return mask, stats


def train_branch_response_operator(
    rows: list[dict[str, Any]],
    examples: dict[str, Any],
    h_pre: torch.Tensor,
    branch_centroids: torch.Tensor,
    source_skill_ids: torch.Tensor,
    action_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> BranchSkillLibrary:
    library = BranchSkillLibrary(
        prototypes=branch_centroids,
        hidden_dim=h_pre.shape[1],
        skill_dim=args.skill_dim,
        num_obs_types=len(OBS_TYPES),
        rank=args.rank,
    ).to(device)
    optimizer = torch.optim.AdamW(library.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = len(examples["indices"])
    row_ids = torch.arange(n, device=device)
    skill_probs = one_hot_skill_probs(source_skill_ids, args.num_skills).to(device)
    h_pre_dev = h_pre.to(device)
    branch_source = build_branch_logits_by_obs(examples["source_logits"], action_names)
    branch_target = build_branch_logits_by_obs(examples["target_logits"], action_names)
    for _ in range(args.epochs):
        perm = row_ids[torch.randperm(row_ids.numel(), device=device)]
        for start in range(0, perm.numel(), args.train_batch_size):
            batch = perm[start : start + args.train_batch_size]
            losses = []
            for obs_id, obs_type in enumerate(OBS_TYPES):
                delta_h = examples["target_hidden"][obs_type].to(device).index_select(0, batch) - h_pre_dev.index_select(0, batch)
                obs_onehot = F.one_hot(
                    torch.full((batch.numel(),), obs_id, dtype=torch.long, device=device),
                    num_classes=len(OBS_TYPES),
                ).float()
                pred = library.predict_branch_residual(delta_h, skill_probs.index_select(0, batch), obs_onehot)
                base_branch = branch_target[obs_type].to(device).index_select(0, batch)
                final_branch = base_branch + pred
                branch_loss_items = []
                aux_items = []
                for local_pos, global_pos in enumerate(batch.tolist()):
                    row_idx = examples["indices"][global_pos]
                    family = tool_family(last_tool_name(str(rows[row_idx].get("prompt") or "")))
                    good, bad = branch_good_bad(obs_type, family)
                    branch_loss_items.append(F.relu(final_branch.new_tensor(args.rank_margin) - branch_margin(final_branch[local_pos], good, bad)))
                    source_delta = branch_source[obs_type][global_pos] - torch.stack(
                        [branch_source[o][global_pos] for o in OBS_TYPES], dim=0
                    ).mean(dim=0)
                    target_delta = branch_target[obs_type][global_pos] - torch.stack(
                        [branch_target[o][global_pos] for o in OBS_TYPES], dim=0
                    ).mean(dim=0)
                    aux_items.append(F.mse_loss(pred[local_pos], (source_delta - target_delta).to(device)))
                loss = torch.stack(branch_loss_items).mean()
                if args.branch_aux_weight > 0:
                    loss = loss + float(args.branch_aux_weight) * torch.stack(aux_items).mean()
                losses.append(loss)
            total_loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    return library.eval()


def train_write_pair_branch_operator(
    rows: list[dict[str, Any]],
    examples: dict[str, Any],
    h_pre: torch.Tensor,
    branch_centroids: torch.Tensor,
    source_skill_ids: torch.Tensor,
    write_pair_mask: torch.Tensor,
    action_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> BranchSkillLibrary:
    library = BranchSkillLibrary(
        prototypes=branch_centroids,
        hidden_dim=h_pre.shape[1],
        skill_dim=args.skill_dim,
        num_obs_types=len(OBS_TYPES),
        rank=args.rank,
    ).to(device)
    optimizer = torch.optim.AdamW(library.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    selected = torch.where(write_pair_mask.to(torch.bool))[0].to(device)
    if selected.numel() == 0:
        selected = torch.arange(len(examples["indices"]), device=device)
    skill_probs = one_hot_skill_probs(source_skill_ids, args.num_skills).to(device)
    h_pre_dev = h_pre.to(device)
    branch_target = build_branch_logits_by_obs(examples["target_logits"], action_names)
    success_obs = OBS_TYPES.index("success")
    failure_obs = OBS_TYPES.index("failure")
    success_onehot = F.one_hot(
        torch.full((1,), success_obs, dtype=torch.long, device=device),
        num_classes=len(OBS_TYPES),
    ).float()
    failure_onehot = F.one_hot(
        torch.full((1,), failure_obs, dtype=torch.long, device=device),
        num_classes=len(OBS_TYPES),
    ).float()
    for _ in range(args.epochs):
        perm = selected[torch.randperm(selected.numel(), device=device)]
        for start in range(0, perm.numel(), args.train_batch_size):
            batch = perm[start : start + args.train_batch_size]
            batch_skills = skill_probs.index_select(0, batch)
            success_delta_h = examples["target_hidden"]["success"].to(device).index_select(0, batch) - h_pre_dev.index_select(0, batch)
            failure_delta_h = examples["target_hidden"]["failure"].to(device).index_select(0, batch) - h_pre_dev.index_select(0, batch)
            success_pred = library.predict_branch_residual(
                success_delta_h,
                batch_skills,
                success_onehot.expand(batch.numel(), -1),
            )
            failure_pred = library.predict_branch_residual(
                failure_delta_h,
                batch_skills,
                failure_onehot.expand(batch.numel(), -1),
            )
            success_branch = branch_target["success"].to(device).index_select(0, batch) + success_pred
            failure_branch = branch_target["failure"].to(device).index_select(0, batch) + failure_pred
            loss = write_pair_loss(success_branch, failure_branch, args.rank_margin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return library.eval()


@torch.no_grad()
def source_branch_residual_by_obs(
    examples: dict[str, Any],
    action_names: list[str],
) -> dict[str, torch.Tensor]:
    return {obs: branch_response_delta_mean_ref(examples["source_logits"], obs, action_names) for obs in OBS_TYPES}


def apply_branch_residual_decoder(
    examples: dict[str, Any],
    residual_by_obs: dict[str, torch.Tensor],
    action_names: list[str],
    args: argparse.Namespace,
    mode: str,
) -> dict[str, torch.Tensor]:
    corrected = {}
    for obs_type in OBS_TYPES:
        static_logits = examples["target_logits"][obs_type]
        residual = residual_by_obs[obs_type]
        if mode == "broadcast":
            corrected[obs_type] = static_logits + float(args.alpha) * broadcast_branch_residual(residual, action_names)
        elif mode == "hard":
            corrected[obs_type] = hard_branch_first_logits(static_logits, residual, action_names, args.alpha)
        elif mode == "soft":
            corrected[obs_type] = soft_hierarchical_logits(static_logits, residual, action_names, args.alpha)
        else:
            raise ValueError(f"Unknown branch decoder mode: {mode}")
    return corrected


@torch.no_grad()
def branch_response_operator_residual_by_obs(
    examples: dict[str, Any],
    h_pre: torch.Tensor,
    skill_probs: torch.Tensor,
    library: BranchSkillLibrary,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    residuals = {}
    h_pre_dev = h_pre.to(device)
    skill_probs_dev = skill_probs.to(device)
    for obs_id, obs_type in enumerate(OBS_TYPES):
        delta_h = examples["target_hidden"][obs_type].to(device) - h_pre_dev
        obs_onehot = F.one_hot(
            torch.full((delta_h.shape[0],), obs_id, dtype=torch.long, device=device),
            num_classes=len(OBS_TYPES),
        ).float()
        branch_residual = library.predict_branch_residual(delta_h, skill_probs_dev, obs_onehot)
        residuals[obs_type] = branch_residual.cpu()
    return residuals


@torch.no_grad()
def write_pair_operator_residual_by_obs(
    rows: list[dict[str, Any]],
    examples: dict[str, Any],
    h_pre: torch.Tensor,
    skill_probs: torch.Tensor,
    library: BranchSkillLibrary,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    n = len(examples["indices"])
    residuals = {
        obs_type: torch.zeros(n, len(BRANCHES), dtype=examples["target_logits"][obs_type].dtype)
        for obs_type in OBS_TYPES
    }
    write_positions = [
        row_pos
        for row_pos, row_idx in enumerate(examples["indices"])
        if tool_family(last_tool_name(str(rows[row_idx].get("prompt") or ""))) == "write"
    ]
    if not write_positions:
        return residuals
    batch = torch.tensor(write_positions, dtype=torch.long, device=device)
    h_pre_dev = h_pre.to(device)
    skill_probs_dev = skill_probs.to(device).index_select(0, batch)
    for obs_type in ("success", "failure"):
        obs_id = OBS_TYPES.index(obs_type)
        delta_h = examples["target_hidden"][obs_type].to(device).index_select(0, batch) - h_pre_dev.index_select(0, batch)
        obs_onehot = F.one_hot(
            torch.full((batch.numel(),), obs_id, dtype=torch.long, device=device),
            num_classes=len(OBS_TYPES),
        ).float()
        pred = library.predict_branch_residual(delta_h, skill_probs_dev, obs_onehot).cpu()
        residuals[obs_type].index_copy_(0, batch.cpu(), pred)
    return residuals


def oracle_correct_write_branch_residual(
    rows: list[dict[str, Any]],
    examples: dict[str, Any],
    action_names: list[str],
    strength: float,
) -> dict[str, torch.Tensor]:
    n = len(examples["indices"])
    residuals = {
        obs_type: torch.zeros(n, len(BRANCHES), dtype=examples["target_logits"][obs_type].dtype)
        for obs_type in OBS_TYPES
    }
    finish = BRANCH_TO_ID["finish"]
    recover = BRANCH_TO_ID["recover"]
    verify = BRANCH_TO_ID["verify"]
    execute = BRANCH_TO_ID["execute"]
    for row_pos, row_idx in enumerate(examples["indices"]):
        family = tool_family(last_tool_name(str(rows[row_idx].get("prompt") or "")))
        if family != "write":
            continue
        residuals["success"][row_pos, finish] += float(strength)
        residuals["success"][row_pos, recover] -= float(strength)
        residuals["success"][row_pos, verify] -= 0.5 * float(strength)
        residuals["failure"][row_pos, recover] += float(strength)
        residuals["failure"][row_pos, verify] += float(strength)
        residuals["failure"][row_pos, finish] -= float(strength)
        residuals["failure"][row_pos, execute] -= 0.5 * float(strength)
    return residuals


def calibrated_write_pair_residual(
    rows: list[dict[str, Any]],
    examples: dict[str, Any],
    action_names: list[str],
    margin: float,
    max_strength: float,
) -> dict[str, torch.Tensor]:
    residuals = {
        obs_type: torch.zeros(
            len(examples["indices"]),
            len(BRANCHES),
            dtype=examples["target_logits"][obs_type].dtype,
        )
        for obs_type in OBS_TYPES
    }
    branch_target = build_branch_logits_by_obs(examples["target_logits"], action_names)
    finish = BRANCH_TO_ID["finish"]
    recover = BRANCH_TO_ID["recover"]
    verify = BRANCH_TO_ID["verify"]
    execute = BRANCH_TO_ID["execute"]
    acquire = BRANCH_TO_ID["acquire"]
    for row_pos, row_idx in enumerate(examples["indices"]):
        family = tool_family(last_tool_name(str(rows[row_idx].get("prompt") or "")))
        if family != "write":
            continue

        success = branch_target["success"][row_pos]
        success_bad = torch.tensor([recover, verify, acquire, execute], dtype=torch.long, device=success.device)
        success_gap = success[finish] - torch.logsumexp(success.index_select(0, success_bad), dim=0)
        success_need = min(float(max_strength), max(0.0, float(margin) - float(success_gap.item())))
        residuals["success"][row_pos, finish] += 0.6 * success_need
        residuals["success"][row_pos, recover] -= 0.15 * success_need
        residuals["success"][row_pos, verify] -= 0.15 * success_need
        residuals["success"][row_pos, acquire] -= 0.15 * success_need
        residuals["success"][row_pos, execute] -= 0.15 * success_need

        failure = branch_target["failure"][row_pos]
        failure_gap = torch.logsumexp(failure[[recover, verify]], dim=0) - failure[finish]
        failure_need = min(float(max_strength), max(0.0, float(margin) - float(failure_gap.item())))
        residuals["failure"][row_pos, recover] += 0.4 * failure_need
        residuals["failure"][row_pos, verify] += 0.4 * failure_need
        residuals["failure"][row_pos, finish] -= 0.4 * failure_need
        residuals["failure"][row_pos, execute] -= 0.2 * failure_need
    return residuals


def add_residuals(*items: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not items:
        return {}
    out = {}
    for obs_type in OBS_TYPES:
        value = items[0][obs_type].clone()
        for item in items[1:]:
            value = value + item[obs_type]
        out[obs_type] = value
    return out


def logits_to_actions(logits: torch.Tensor, action_names: list[str]) -> list[str]:
    return [action_names[int(idx)] for idx in logits.argmax(dim=1).tolist()]


def branch_metrics(rows: list[dict[str, Any]], examples: dict[str, Any], logits_by_obs: dict[str, torch.Tensor], action_names: list[str]) -> dict[str, Any]:
    actions_by_obs = {obs: logits_to_actions(logits, action_names) for obs, logits in logits_by_obs.items()}
    total = 0
    branch_correct = 0
    write_total = 0
    write_correct = 0
    write_success_total = 0
    write_success_correct = 0
    write_failure_total = 0
    write_failure_correct = 0
    pair_total = 0
    pair_correct = 0
    write_pair_total = 0
    write_pair_correct = 0
    unique_count = 0
    unique_write_count = 0
    for row_pos, row_idx in enumerate(examples["indices"]):
        prompt = str(rows[row_idx].get("prompt") or "")
        tool = last_tool_name(prompt)
        family = tool_family(tool)
        unique_actions = {actions_by_obs[obs][row_pos] for obs in OBS_TYPES}
        if len(unique_actions) > 1:
            unique_count += 1
            if family == "write":
                unique_write_count += 1
        success_branch = branch_for_action(actions_by_obs["success"][row_pos])
        failure_branch = branch_for_action(actions_by_obs["failure"][row_pos])
        success_ok = success_branch == "finish" if family == "write" else success_branch != "recover"
        failure_ok = failure_branch in {"recover", "verify", "acquire"}
        pair_total += 1
        pair_correct += int(success_ok and failure_ok)
        if family == "write":
            write_pair_total += 1
            write_pair_correct += int(success_ok and failure_ok)
        for obs in OBS_TYPES:
            action = actions_by_obs[obs][row_pos]
            ok = branch_for_action(action) in expected_branches(obs, family)
            total += 1
            branch_correct += int(ok)
            if family == "write":
                write_total += 1
                write_correct += int(ok)
                if obs == "success":
                    write_success_total += 1
                    write_success_correct += int(ok)
                elif obs == "failure":
                    write_failure_total += 1
                    write_failure_correct += int(ok)
    return {
        "branch_acc_all": branch_correct / max(1, total),
        "branch_acc_write": write_correct / max(1, write_total),
        "write_success_branch_acc": write_success_correct / max(1, write_success_total),
        "write_failure_branch_acc": write_failure_correct / max(1, write_failure_total),
        "success_failure_pair_acc": pair_correct / max(1, pair_total),
        "write_success_failure_pair_acc": write_pair_correct / max(1, write_pair_total),
        "unique_action_prefix_rate": unique_count / max(1, len(examples["indices"])),
        "unique_action_write_prefix_rate": unique_write_count / max(1, write_pair_total),
        "action_distribution_by_observation": {
            obs: dict(Counter(actions_by_obs[obs]).most_common(10)) for obs in OBS_TYPES
        },
    }


@torch.no_grad()
def apply_response_gap_adapter(
    examples: dict[str, Any],
    centroids: torch.Tensor,
    library: ResponseSkillLibrary,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    corrected = {obs: logits.clone() for obs, logits in examples["target_logits"].items()}
    skill_probs = bind_target_skill(examples["target_fingerprint"], centroids, args.skill_temperature).to(device)
    for obs_id, obs_type in enumerate(DELTA_OBS_TYPES):
        delta_h = (examples["target_hidden"][obs_type] - examples["target_hidden"]["success"]).to(device)
        obs_onehot = F.one_hot(torch.full((delta_h.shape[0],), obs_id, dtype=torch.long, device=device), len(DELTA_OBS_TYPES)).float()
        pred_gap = library.predict_gap(delta_h, skill_probs, obs_onehot).cpu()
        margin = policy_margin(examples["target_logits"][obs_type])
        gate = response_gate(obs_type, margin, args.gate_static_margin).cpu()
        corrected[obs_type] = examples["target_logits"][obs_type] + gate[:, None] * pred_gap
    return corrected


def apply_true_response_gap_oracle(examples: dict[str, Any], args: argparse.Namespace) -> dict[str, torch.Tensor]:
    corrected = {obs: logits.clone() for obs, logits in examples["target_logits"].items()}
    for obs_type in OBS_TYPES:
        gap = true_response_gap_mean_ref(examples, obs_type)
        margin = policy_margin(examples["target_logits"][obs_type])
        gate = response_gate(obs_type, margin, args.gate_static_margin).cpu()
        corrected[obs_type] = examples["target_logits"][obs_type] + float(args.alpha) * gate[:, None] * gap
    return corrected


def apply_prototype_operator(
    examples: dict[str, Any],
    skill_probs: torch.Tensor,
    response_basis: torch.Tensor,
    operator: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    corrected = {obs: logits.clone() for obs, logits in examples["target_logits"].items()}
    for obs_id, obs_type in enumerate(OBS_TYPES):
        coords = torch.einsum("nk,kr->nr", skill_probs.float(), operator[:, obs_id, :].float())
        pred_gap = coords @ response_basis.float().T
        margin = policy_margin(examples["target_logits"][obs_type])
        gate = response_gate(obs_type, margin, args.gate_static_margin).cpu()
        corrected[obs_type] = examples["target_logits"][obs_type] + float(args.alpha) * gate[:, None] * pred_gap
    return corrected


def one_hot_skill_probs(skill_ids: torch.Tensor, num_skills: int) -> torch.Tensor:
    return F.one_hot(skill_ids.long(), num_classes=num_skills).float()


def gap_diagnostics(
    examples: dict[str, Any],
    corrected: dict[str, torch.Tensor],
    base: dict[str, torch.Tensor],
) -> dict[str, float]:
    true_gaps = []
    pred_gaps = []
    source_norms = []
    target_norms = []
    for obs_type in OBS_TYPES:
        source_delta = response_delta_mean_ref(examples["source_logits"], obs_type)
        target_delta = response_delta_mean_ref(examples["target_logits"], obs_type)
        true_gap = source_delta - target_delta
        pred_gap = corrected[obs_type] - base[obs_type]
        true_gaps.append(true_gap)
        pred_gaps.append(pred_gap)
        source_norms.append(source_delta.norm(dim=1))
        target_norms.append(target_delta.norm(dim=1))
    true_mat = torch.cat(true_gaps, dim=0)
    pred_mat = torch.cat(pred_gaps, dim=0)
    cos = F.cosine_similarity(center(pred_mat), center(true_mat), dim=1)
    base_actions = torch.cat([base[obs].argmax(dim=1) for obs in OBS_TYPES], dim=0)
    pred_actions = torch.cat([corrected[obs].argmax(dim=1) for obs in OBS_TYPES], dim=0)
    source_actions = torch.cat([examples["source_logits"][obs].argmax(dim=1) for obs in OBS_TYPES], dim=0)
    flipped = pred_actions != base_actions
    return {
        "mean_source_delta_norm": float(torch.cat(source_norms).mean().item()),
        "mean_target_delta_norm": float(torch.cat(target_norms).mean().item()),
        "mean_true_gap_norm": float(true_mat.norm(dim=1).mean().item()),
        "mean_pred_gap_norm": float(pred_mat.norm(dim=1).mean().item()),
        "gap_cosine": float(cos.mean().item()),
        "argmax_flip_rate": float(flipped.float().mean().item()),
        "correct_flip_rate": float(((pred_actions == source_actions) & flipped).float().mean().item()),
        "harmful_flip_rate": float(((base_actions == source_actions) & flipped).float().mean().item()),
    }


def branch_flip_diagnostics(
    rows: list[dict[str, Any]],
    examples: dict[str, Any],
    corrected: dict[str, torch.Tensor],
    base: dict[str, torch.Tensor],
    action_names: list[str],
) -> dict[str, float]:
    correct_flips = 0
    harmful_flips = 0
    total = 0
    write_correct_flips = 0
    write_harmful_flips = 0
    write_total = 0
    for row_pos, row_idx in enumerate(examples["indices"]):
        family = tool_family(last_tool_name(str(rows[row_idx].get("prompt") or "")))
        for obs_type in OBS_TYPES:
            allowed = expected_branches(obs_type, family)
            base_action = action_names[int(base[obs_type][row_pos].argmax().item())]
            new_action = action_names[int(corrected[obs_type][row_pos].argmax().item())]
            base_ok = branch_for_action(base_action) in allowed
            new_ok = branch_for_action(new_action) in allowed
            changed = base_action != new_action
            correct_flips += int(changed and (not base_ok) and new_ok)
            harmful_flips += int(changed and base_ok and (not new_ok))
            total += 1
            if family == "write":
                write_correct_flips += int(changed and (not base_ok) and new_ok)
                write_harmful_flips += int(changed and base_ok and (not new_ok))
                write_total += 1
    return {
        "correct_flip_rate": correct_flips / max(1, total),
        "harmful_flip_rate": harmful_flips / max(1, total),
        "write_correct_flip_rate": write_correct_flips / max(1, write_total),
        "write_harmful_flip_rate": write_harmful_flips / max(1, write_total),
    }


def branch_decoder_diagnostics(
    rows: list[dict[str, Any]],
    examples: dict[str, Any],
    corrected: dict[str, torch.Tensor],
    base: dict[str, torch.Tensor],
    action_names: list[str],
) -> dict[str, float]:
    branch_total = 0
    branch_correct = 0
    branch_correct_flip = 0
    branch_harmful_flip = 0
    within_total = 0
    within_static_match = 0
    primitive_correct_given_branch_correct = 0
    primitive_total_given_branch_correct = 0
    static_branch_counter: Counter[str] = Counter()
    corrected_branch_counter: Counter[str] = Counter()
    oracle_branch_counter: Counter[str] = Counter()
    for row_pos, row_idx in enumerate(examples["indices"]):
        family = tool_family(last_tool_name(str(rows[row_idx].get("prompt") or "")))
        for obs_type in OBS_TYPES:
            allowed = expected_branches(obs_type, family)
            oracle = sorted(allowed)[0] if allowed else "recover"
            base_action = action_names[int(base[obs_type][row_pos].argmax().item())]
            new_action = action_names[int(corrected[obs_type][row_pos].argmax().item())]
            base_branch = branch_for_action(base_action)
            new_branch = branch_for_action(new_action)
            base_ok = base_branch in allowed
            new_ok = new_branch in allowed
            changed = base_branch != new_branch
            branch_correct += int(new_ok)
            branch_correct_flip += int(changed and (not base_ok) and new_ok)
            branch_harmful_flip += int(changed and base_ok and (not new_ok))
            branch_total += 1
            static_branch_counter[base_branch] += 1
            corrected_branch_counter[new_branch] += 1
            oracle_branch_counter[oracle] += 1
            if new_ok:
                ids = actions_in_branch(action_names, action_to_branch_id(new_action))
                static_within = max(ids, key=lambda idx: float(base[obs_type][row_pos, idx].item()))
                within_total += 1
                within_static_match += int(action_names[static_within] == new_action)
                primitive_total_given_branch_correct += 1
                primitive_correct_given_branch_correct += int(new_ok)
    return {
        "branch_selection_accuracy": branch_correct / max(1, branch_total),
        "branch_correct_flip_rate": branch_correct_flip / max(1, branch_total),
        "branch_harmful_flip_rate": branch_harmful_flip / max(1, branch_total),
        "within_branch_static_top1_match": within_static_match / max(1, within_total),
        "primitive_correct_flip_given_branch_correct": primitive_correct_given_branch_correct / max(1, primitive_total_given_branch_correct),
        "static_branch_distribution": dict(static_branch_counter),
        "corrected_branch_distribution": dict(corrected_branch_counter),
        "oracle_branch_distribution": dict(oracle_branch_counter),
    }


def branch_response_direction_audit(
    rows: list[dict[str, Any]],
    examples: dict[str, Any],
    branch_residual_by_obs: dict[str, torch.Tensor],
    action_names: list[str],
) -> dict[str, float]:
    branch_target = build_branch_logits_by_obs(examples["target_logits"], action_names)
    direction = []
    scales = []
    for row_pos, row_idx in enumerate(examples["indices"]):
        family = tool_family(last_tool_name(str(rows[row_idx].get("prompt") or "")))
        for obs_type in OBS_TYPES:
            good, bad = branch_good_bad(obs_type, family)
            base = branch_target[obs_type][row_pos].float()
            residual = branch_residual_by_obs[obs_type][row_pos].float()
            good_idx = torch.tensor(good, dtype=torch.long)
            bad_idx = torch.tensor(bad, dtype=torch.long)
            base_good = torch.logsumexp(base.index_select(0, good_idx), dim=0)
            base_bad = torch.logsumexp(base.index_select(0, bad_idx), dim=0)
            res_good = torch.logsumexp(residual.index_select(0, good_idx), dim=0)
            res_bad = torch.logsumexp(residual.index_select(0, bad_idx), dim=0)
            denom = float((res_good - res_bad).item())
            direction.append(denom > 0)
            if denom > 1e-9:
                scales.append(max(0.0, float((base_bad - base_good).item())) / denom)
    alpha = torch.tensor(scales, dtype=torch.float32) if scales else torch.empty(0)
    return {
        "branch_direction_correct_rate": float(sum(direction) / max(1, len(direction))),
        "median_branch_flip_scale": float(alpha.median().item()) if alpha.numel() else float("inf"),
        "flippable_branch_alpha_1": float((alpha <= 1).float().mean().item()) if alpha.numel() else 0.0,
        "flippable_branch_alpha_2": float((alpha <= 2).float().mean().item()) if alpha.numel() else 0.0,
        "flippable_branch_alpha_5": float((alpha <= 5).float().mean().item()) if alpha.numel() else 0.0,
    }


def skill_binder_diagnostics(skill_probs: torch.Tensor, oracle_skill_ids: torch.Tensor) -> dict[str, float]:
    pred = skill_probs.argmax(dim=1)
    entropy = -(skill_probs.clamp_min(1e-12) * skill_probs.clamp_min(1e-12).log()).sum(dim=1)
    return {
        "skill_binder_accuracy": float((pred == oracle_skill_ids).float().mean().item()),
        "skill_posterior_entropy": float(entropy.mean().item()),
        "top1_skill_confidence": float(skill_probs.max(dim=1).values.mean().item()),
    }


def any_top1_flip_rate(logits_by_obs: dict[str, torch.Tensor]) -> float:
    ref = logits_by_obs["success"].argmax(dim=1)
    flips = []
    for obs_type in DELTA_OBS_TYPES:
        flips.append(logits_by_obs[obs_type].argmax(dim=1) != ref)
    return float(torch.stack(flips, dim=1).any(dim=1).float().mean().item())


def any_branch_flip_rate(logits_by_obs: dict[str, torch.Tensor], action_names: list[str]) -> float:
    ref_actions = logits_to_actions(logits_by_obs["success"], action_names)
    ref_branches = [branch_for_action(action) for action in ref_actions]
    flipped = torch.zeros(len(ref_branches), dtype=torch.bool)
    for obs_type in DELTA_OBS_TYPES:
        actions = logits_to_actions(logits_by_obs[obs_type], action_names)
        branches = [branch_for_action(action) for action in actions]
        flipped |= torch.tensor([branch != ref for branch, ref in zip(branches, ref_branches)])
    return float(flipped.float().mean().item())


def sf_cosine_distance(space_by_obs: dict[str, torch.Tensor] | None) -> float:
    if not space_by_obs:
        return float("nan")
    a = F.normalize(space_by_obs["success"].float(), dim=1)
    b = F.normalize(space_by_obs["failure"].float(), dim=1)
    return float((1.0 - (a * b).sum(dim=1)).mean().item())


def sf_delta_norm(space_by_obs: dict[str, torch.Tensor] | None) -> float:
    if not space_by_obs:
        return float("nan")
    return float((space_by_obs["failure"].float() - space_by_obs["success"].float()).norm(dim=1).mean().item())


def response_localization_audit(
    examples: dict[str, Any],
    action_names: list[str],
    prefix: str,
) -> dict[str, float]:
    source_h = examples.get("source_hidden")
    source_z = examples.get("source_z")
    source_l = examples.get("source_logits")
    target_h = examples.get("target_hidden")
    target_z = examples.get("target_z")
    target_l = examples.get("target_logits")
    return {
        f"{prefix}_source_delta_h_norm": sf_delta_norm(source_h),
        f"{prefix}_source_delta_z_norm": sf_delta_norm(source_z),
        f"{prefix}_source_delta_logit_norm": sf_delta_norm(source_l),
        f"{prefix}_target_delta_h_norm": sf_delta_norm(target_h),
        f"{prefix}_target_delta_z_norm": sf_delta_norm(target_z),
        f"{prefix}_target_delta_logit_norm": sf_delta_norm(target_l),
        f"{prefix}_source_success_failure_cosine_distance_h": sf_cosine_distance(source_h),
        f"{prefix}_source_success_failure_cosine_distance_z": sf_cosine_distance(source_z),
        f"{prefix}_source_success_failure_cosine_distance_logits": sf_cosine_distance(source_l),
        f"{prefix}_target_success_failure_cosine_distance_h": sf_cosine_distance(target_h),
        f"{prefix}_target_success_failure_cosine_distance_z": sf_cosine_distance(target_z),
        f"{prefix}_target_success_failure_cosine_distance_logits": sf_cosine_distance(target_l),
        f"{prefix}_source_counterfactual_top1_flip_rate": any_top1_flip_rate(source_l),
        f"{prefix}_target_counterfactual_top1_flip_rate": any_top1_flip_rate(target_l),
        f"{prefix}_source_branch_flip_rate": any_branch_flip_rate(source_l, action_names),
        f"{prefix}_target_branch_flip_rate": any_branch_flip_rate(target_l, action_names),
    }


def policy_critical_mask(
    examples: dict[str, Any],
    action_names: list[str],
    impact_threshold: float,
) -> torch.Tensor:
    n = examples["source_logits"]["success"].shape[0]
    mask = torch.zeros(n, dtype=torch.bool)
    ref_actions = logits_to_actions(examples["source_logits"]["success"], action_names)
    ref_branches = [branch_for_action(action) for action in ref_actions]
    max_delta = torch.zeros(n)
    for obs_type in DELTA_OBS_TYPES:
        logits = examples["source_logits"][obs_type]
        actions = logits_to_actions(logits, action_names)
        branches = [branch_for_action(action) for action in actions]
        mask |= torch.tensor([action != ref for action, ref in zip(actions, ref_actions)])
        mask |= torch.tensor([branch != ref for branch, ref in zip(branches, ref_branches)])
        max_delta = torch.maximum(max_delta, (logits - examples["source_logits"]["success"]).norm(dim=1))
    mask |= max_delta > float(impact_threshold)
    return mask


def true_gap_direction_audit(
    rows: list[dict[str, Any]],
    examples: dict[str, Any],
    action_names: list[str],
) -> dict[str, float]:
    direction = []
    alpha_star = []
    branch_margin_lifts = []
    for row_pos, row_idx in enumerate(examples["indices"]):
        family = tool_family(last_tool_name(str(rows[row_idx].get("prompt") or "")))
        for obs_type in OBS_TYPES:
            logits = examples["target_logits"][obs_type][row_pos].float()
            good_branches = expected_branches(obs_type, family)
            good = [idx for idx, action in enumerate(action_names) if branch_for_action(action) in good_branches]
            if not good:
                continue
            bad = [idx for idx in range(len(action_names)) if idx not in good]
            current = int(logits.argmax().item())
            best_good = max(good, key=lambda idx: float(logits[idx].item()))
            single = {
                "source_logits": {obs: value[row_pos : row_pos + 1] for obs, value in examples["source_logits"].items()},
                "target_logits": {obs: value[row_pos : row_pos + 1] for obs, value in examples["target_logits"].items()},
            }
            gap = true_response_gap_mean_ref(single, obs_type)[0].float()
            denom = float((gap[best_good] - gap[current]).item())
            direction.append(denom > 0)
            if denom > 1e-9:
                scale = max(0.0, float((logits[current] - logits[best_good]).item())) / denom
                alpha_star.append(scale)
            good_score = logits[good].max()
            bad_score = logits[bad].max() if bad else logits.new_tensor(0.0)
            good_gap = gap[good].max()
            bad_gap = gap[bad].max() if bad else gap.new_tensor(0.0)
            before = good_score - bad_score
            after = (good_score + good_gap) - (bad_score + bad_gap)
            branch_margin_lifts.append(float((after - before).item()))
    alphas = torch.tensor(alpha_star, dtype=torch.float32) if alpha_star else torch.empty(0)
    return {
        "true_gap_direction_correct_rate": float(sum(direction) / max(1, len(direction))),
        "median_min_flip_scale": float(alphas.median().item()) if alphas.numel() else float("inf"),
        "flippable_alpha_1": float((alphas <= 1).float().mean().item()) if alphas.numel() else 0.0,
        "flippable_alpha_2": float((alphas <= 2).float().mean().item()) if alphas.numel() else 0.0,
        "flippable_alpha_5": float((alphas <= 5).float().mean().item()) if alphas.numel() else 0.0,
        "flippable_alpha_10": float((alphas <= 10).float().mean().item()) if alphas.numel() else 0.0,
        "true_gap_branch_margin_lift": float(sum(branch_margin_lifts) / max(1, len(branch_margin_lifts))),
    }


def rule_constrained_branch_logits(
    rows: list[dict[str, Any]],
    examples: dict[str, Any],
    logits_by_obs: dict[str, torch.Tensor],
    action_names: list[str],
) -> dict[str, torch.Tensor]:
    constrained = {obs: logits.clone() for obs, logits in logits_by_obs.items()}
    for row_pos, row_idx in enumerate(examples["indices"]):
        tool = tool_family(last_tool_name(str(rows[row_idx].get("prompt") or "")))
        for obs_type in OBS_TYPES:
            allowed_branches = expected_branches(obs_type, tool)
            allowed = [idx for idx, action in enumerate(action_names) if branch_for_action(action) in allowed_branches]
            if not allowed:
                continue
            mask = torch.ones(len(action_names), dtype=torch.bool)
            mask[allowed] = False
            constrained[obs_type][row_pos, mask] = -1e9
    return constrained


def source_response_stats(rows: list[dict[str, Any]], examples: dict[str, Any], action_names: list[str]) -> dict[str, Any]:
    metrics = branch_metrics(rows, examples, examples["source_logits"], action_names)
    norms = []
    for obs in DELTA_OBS_TYPES:
        norms.append(response_delta(examples["source_logits"], obs).norm(dim=1))
    metrics["mean_source_response_norm"] = float(torch.cat(norms).mean().item())
    return metrics


def skill_prototypes(
    examples: dict[str, Any],
    centroids: torch.Tensor,
    temperature: float,
    fingerprint: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    source_fingerprint = examples["source_fingerprint"] if fingerprint is None else fingerprint
    assignments = bind_target_skill(source_fingerprint, centroids, temperature).argmax(dim=1)
    prototypes = []
    for skill_id in range(centroids.shape[0]):
        mask = assignments == skill_id
        item: dict[str, Any] = {"skill_id": skill_id, "count": int(mask.sum().item())}
        mean_response = {}
        for obs_type in DELTA_OBS_TYPES:
            deltas = response_delta(examples["source_logits"], obs_type)
            if bool(mask.any()):
                mean_response[obs_type] = deltas[mask].mean(dim=0)
            else:
                mean_response[obs_type] = torch.zeros_like(deltas[0])
        item["mean_source_response"] = mean_response
        prototypes.append(item)
    return prototypes


def save_report(out_dir: Path, metadata: dict[str, Any], results: list[dict[str, Any]]) -> None:
    lines = [
        "# Counterfactual Response Skill Results",
        "",
        "Skills are discovered from same-prefix Source policy response fingerprints, not from action groups.",
        "",
        "## Setup",
        "",
        f"- Train prefixes: {metadata['num_train_prefixes']}",
        f"- Eval prefixes: {metadata['num_eval_prefixes']}",
        f"- Observation bank entries: {metadata['num_observation_bank_entries']}",
        f"- Num skills: {metadata['num_skills']}",
        f"- Split: `{metadata['split_metadata']}`",
        "",
        "## Metrics",
        "",
        "| Method | Branch Acc All | Branch Acc Write | Write Success Branch | Write Failure Branch | Success/Failure Pair | Write S/F Pair | Unique Action Prefix | Write Correct Flip | Write Harmful Flip |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            "| {method} | {branch_all} | {branch_write} | {write_success} | {write_failure} | {pair} | {write_pair} | {unique} | {write_correct_flip} | {write_harmful_flip} |".format(
                method=item["method"],
                branch_all=pct(item["branch_acc_all"]),
                branch_write=pct(item["branch_acc_write"]),
                write_success=pct(item.get("write_success_branch_acc", 0.0)),
                write_failure=pct(item.get("write_failure_branch_acc", 0.0)),
                pair=pct(item["success_failure_pair_acc"]),
                write_pair=pct(item["write_success_failure_pair_acc"]),
                unique=pct(item["unique_action_prefix_rate"]),
                write_correct_flip=pct(item.get("write_correct_flip_rate", 0.0)),
                write_harmful_flip=pct(item.get("write_harmful_flip_rate", 0.0)),
            )
        )
    lines.extend(["", "## Notes", ""])
    lines.append("- `Static Low-Rank` is the existing target-to-source latent alignment plus frozen Source head.")
    lines.append("- `Oracle Source Branch Response` adds held-out Source branch-level response residuals to Static Low-Rank.")
    lines.append("- `Oracle/Predicted Skill + Branch Operator` use raw Target post-pre hidden deltas and predict branch residuals, not full action-logit gaps.")
    lines.append("- `Pair-Contrastive Write Operator` is trained only on strong write-critical prefixes and uses success/failure pair ranking loss.")
    lines.append("- `Calibrated Write-Pair Prior` uses only write-tool family and observation status to enforce a minimum branch margin.")
    lines.append("- `Rule-Constrained Branch Ceiling` is a rule ceiling, not a learned Source oracle.")
    out_dir.joinpath("counterfactual_eval_results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    rows = load_rows(args.decision_jsonl)
    train_idx, val_idx, split_metadata = eval_protocol.load_split(args.split_json, len(rows))
    train_selected = select_valid_indices(rows, [int(i) for i in train_idx.tolist()], args.max_train_prefixes)
    eval_selected = select_valid_indices(rows, [int(i) for i in val_idx.tolist()], args.max_eval_prefixes)
    if not train_selected or not eval_selected:
        raise ValueError("No strict post-tool branch prefixes found in train/eval split")

    bank = build_observation_bank(rows, train_selected)
    source_parts = load_source_parts(args.source_policy, device)
    action_names = source_parts.action_names or list(rows[0]["action_names"])
    alignment = torch.load(args.alignment, map_location="cpu")
    target_projection = RidgeProjection(alignment["coef"]).to(device).eval()
    source_projection = source_parts.adapter.to(device).eval()
    head = source_parts.head.to(device).eval()
    for module in (source_projection, target_projection, head):
        for param in module.parameters():
            param.requires_grad = False

    train_bundles, train_selected = build_bundles(rows, train_selected, bank, args)
    eval_bundles, eval_selected = build_bundles(rows, eval_selected, bank, args)

    source_runner = BackboneRunner(args.source_model_id, args.layer, args.max_input_tokens, args.torch_dtype)
    train_hs, train_ls = forward_bundles(source_runner, train_bundles, source_projection, head, args.batch_size, device)
    eval_hs, eval_ls = forward_bundles(source_runner, eval_bundles, source_projection, head, args.batch_size, device)
    train_zs = project_hidden_by_obs(train_hs, source_projection, device)
    eval_zs = project_hidden_by_obs(eval_hs, source_projection, device)
    del source_runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    target_runner = BackboneRunner(args.target_model_id, args.layer, args.max_input_tokens, args.torch_dtype)
    train_ht, train_lt = forward_bundles(target_runner, train_bundles, target_projection, head, args.batch_size, device)
    eval_ht, eval_lt = forward_bundles(target_runner, eval_bundles, target_projection, head, args.batch_size, device)
    train_zt = project_hidden_by_obs(train_ht, target_projection, device)
    eval_zt = project_hidden_by_obs(eval_ht, target_projection, device)
    train_h_pre, train_z_pre = forward_pre_observation(target_runner, rows, train_selected, target_projection, args.batch_size, device)
    eval_h_pre, eval_z_pre = forward_pre_observation(target_runner, rows, eval_selected, target_projection, args.batch_size, device)
    del target_runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_examples = combine_examples(train_selected, train_bundles, train_hs, train_ls, train_ht, train_lt, train_zs, train_zt)
    train_examples["source_fingerprint"] = build_response_fingerprint(train_ls, args.fingerprint)
    train_examples["target_fingerprint"] = build_response_fingerprint(train_lt, args.fingerprint)
    branch_critical_mask, train_branch_stats = source_validated_branch_critical_mask(
        rows,
        train_examples,
        action_names,
        args.branch_margin_threshold,
        args.critical_top_quantile,
    )
    train_branch_fingerprint = build_branch_response_fingerprint(train_examples["source_logits"], action_names)
    eval_branch_fingerprint = build_branch_response_fingerprint(eval_ls, action_names)
    critical_fingerprints = train_branch_fingerprint[branch_critical_mask]
    if critical_fingerprints.shape[0] >= args.num_skills:
        branch_centroids = fit_kmeans(critical_fingerprints, args.num_skills)
    else:
        branch_centroids = fit_kmeans(train_branch_fingerprint, args.num_skills)
    train_source_skill_ids = source_skill_ids_from_centroids(train_branch_fingerprint, branch_centroids)
    skill_binder = train_functional_skill_binder(train_z_pre, train_source_skill_ids, args.num_skills, args, device)
    branch_library = train_branch_response_operator(
        rows,
        train_examples,
        train_h_pre,
        branch_centroids,
        train_source_skill_ids,
        action_names,
        args,
        device,
    )
    write_pair_mask, write_pair_train_stats = write_pair_critical_mask(
        rows,
        train_examples,
        action_names,
        args.write_pair_top_quantile,
    )
    write_pair_library = train_write_pair_branch_operator(
        rows,
        train_examples,
        train_h_pre,
        branch_centroids,
        train_source_skill_ids,
        write_pair_mask,
        action_names,
        args,
        device,
    )

    eval_examples = combine_examples(eval_selected, eval_bundles, eval_hs, eval_ls, eval_ht, eval_lt, eval_zs, eval_zt)
    eval_examples["source_fingerprint"] = build_response_fingerprint(eval_ls, args.fingerprint)
    eval_examples["target_fingerprint"] = build_response_fingerprint(eval_lt, args.fingerprint)
    eval_source_skill_ids = source_skill_ids_from_centroids(eval_branch_fingerprint, branch_centroids)
    pred_skill_probs = binder_skill_probs(skill_binder, eval_z_pre, device)
    prototype_skill_probs = prototype_binder_skill_probs(
        train_z_pre,
        train_source_skill_ids,
        eval_z_pre,
        args.num_skills,
        args.skill_temperature,
    )
    ridge_skill_probs = ridge_binder_skill_probs(
        train_z_pre,
        train_source_skill_ids,
        eval_z_pre,
        args.num_skills,
    )
    oracle_skill_probs = one_hot_skill_probs(eval_source_skill_ids, args.num_skills)
    source_branch_residuals_eval = source_branch_residual_by_obs(eval_examples, action_names)
    oracle_write_branch_residuals = oracle_correct_write_branch_residual(
        rows,
        eval_examples,
        action_names,
        args.write_branch_strength,
    )
    calibrated_write_residuals = calibrated_write_pair_residual(
        rows,
        eval_examples,
        action_names,
        args.write_pair_calibration_margin,
        args.write_pair_max_calibration,
    )
    source_branch_oracle_broadcast_logits = apply_branch_residual_decoder(
        eval_examples, source_branch_residuals_eval, action_names, args, mode="broadcast"
    )
    source_branch_oracle_hard_logits = apply_branch_residual_decoder(
        eval_examples, source_branch_residuals_eval, action_names, args, mode="hard"
    )
    source_branch_oracle_soft_logits = apply_branch_residual_decoder(
        eval_examples, source_branch_residuals_eval, action_names, args, mode="soft"
    )
    oracle_branch_residuals = branch_response_operator_residual_by_obs(
        eval_examples,
        eval_h_pre,
        oracle_skill_probs,
        branch_library,
        device,
    )
    pred_branch_residuals = branch_response_operator_residual_by_obs(
        eval_examples,
        eval_h_pre,
        pred_skill_probs,
        branch_library,
        device,
    )
    oracle_write_pair_residuals = write_pair_operator_residual_by_obs(
        rows,
        eval_examples,
        eval_h_pre,
        oracle_skill_probs,
        write_pair_library,
        device,
    )
    pred_write_pair_residuals = write_pair_operator_residual_by_obs(
        rows,
        eval_examples,
        eval_h_pre,
        pred_skill_probs,
        write_pair_library,
        device,
    )
    prototype_write_pair_residuals = write_pair_operator_residual_by_obs(
        rows,
        eval_examples,
        eval_h_pre,
        prototype_skill_probs,
        write_pair_library,
        device,
    )
    ridge_write_pair_residuals = write_pair_operator_residual_by_obs(
        rows,
        eval_examples,
        eval_h_pre,
        ridge_skill_probs,
        write_pair_library,
        device,
    )
    oracle_write_branch_soft_logits = apply_branch_residual_decoder(
        eval_examples, oracle_write_branch_residuals, action_names, args, mode="soft"
    )
    calibrated_write_soft_logits = apply_branch_residual_decoder(
        eval_examples, calibrated_write_residuals, action_names, args, mode="soft"
    )
    oracle_branch_operator_broadcast_logits = apply_branch_residual_decoder(
        eval_examples, oracle_branch_residuals, action_names, args, mode="broadcast"
    )
    oracle_branch_operator_hard_logits = apply_branch_residual_decoder(
        eval_examples, oracle_branch_residuals, action_names, args, mode="hard"
    )
    oracle_branch_operator_soft_logits = apply_branch_residual_decoder(
        eval_examples, oracle_branch_residuals, action_names, args, mode="soft"
    )
    pred_branch_operator_broadcast_logits = apply_branch_residual_decoder(
        eval_examples, pred_branch_residuals, action_names, args, mode="broadcast"
    )
    pred_branch_operator_hard_logits = apply_branch_residual_decoder(
        eval_examples, pred_branch_residuals, action_names, args, mode="hard"
    )
    pred_branch_operator_soft_logits = apply_branch_residual_decoder(
        eval_examples, pred_branch_residuals, action_names, args, mode="soft"
    )
    oracle_write_pair_soft_logits = apply_branch_residual_decoder(
        eval_examples, oracle_write_pair_residuals, action_names, args, mode="soft"
    )
    pred_write_pair_soft_logits = apply_branch_residual_decoder(
        eval_examples, pred_write_pair_residuals, action_names, args, mode="soft"
    )
    prototype_write_pair_soft_logits = apply_branch_residual_decoder(
        eval_examples, prototype_write_pair_residuals, action_names, args, mode="soft"
    )
    ridge_write_pair_soft_logits = apply_branch_residual_decoder(
        eval_examples, ridge_write_pair_residuals, action_names, args, mode="soft"
    )
    pred_write_pair_calibrated_soft_logits = apply_branch_residual_decoder(
        eval_examples,
        add_residuals(pred_write_pair_residuals, calibrated_write_residuals),
        action_names,
        args,
        mode="soft",
    )

    static = branch_metrics(rows, eval_examples, eval_examples["target_logits"], action_names)
    static["method"] = "Static Low-Rank"
    static.update(branch_flip_diagnostics(rows, eval_examples, eval_examples["target_logits"], eval_examples["target_logits"], action_names))
    result_specs = [
        ("Oracle Correct Write Branch + Soft Hierarchical", oracle_write_branch_soft_logits),
        ("Calibrated Write-Pair Prior + Soft Hierarchical", calibrated_write_soft_logits),
        ("Oracle Source Branch Response + Broadcast", source_branch_oracle_broadcast_logits),
        ("Oracle Source Branch Response + Hard Branch-First", source_branch_oracle_hard_logits),
        ("Oracle Source Branch Response + Soft Hierarchical", source_branch_oracle_soft_logits),
        ("Oracle Skill + Branch Operator + Broadcast", oracle_branch_operator_broadcast_logits),
        ("Oracle Skill + Branch Operator + Hard Branch-First", oracle_branch_operator_hard_logits),
        ("Oracle Skill + Branch Operator + Soft Hierarchical", oracle_branch_operator_soft_logits),
        ("Predicted Skill + Branch Operator + Broadcast", pred_branch_operator_broadcast_logits),
        ("Predicted Skill + Branch Operator + Hard Branch-First", pred_branch_operator_hard_logits),
        ("Predicted Skill + Branch Operator + Soft Hierarchical", pred_branch_operator_soft_logits),
        ("Oracle Skill + Pair-Contrastive Write Operator", oracle_write_pair_soft_logits),
        ("Predicted Skill MLP + Pair-Contrastive Write Operator", pred_write_pair_soft_logits),
        ("Prototype Binder + Pair-Contrastive Write Operator", prototype_write_pair_soft_logits),
        ("Ridge Binder + Pair-Contrastive Write Operator", ridge_write_pair_soft_logits),
        ("Predicted Pair Operator + Calibrated Write-Pair Prior", pred_write_pair_calibrated_soft_logits),
    ]
    decoded_results = []
    decoder_diagnostics = {}
    for method, logits_by_obs in result_specs:
        metrics = branch_metrics(rows, eval_examples, logits_by_obs, action_names)
        metrics["method"] = method
        metrics.update(branch_flip_diagnostics(rows, eval_examples, logits_by_obs, eval_examples["target_logits"], action_names))
        decoded_results.append(metrics)
        decoder_diagnostics[method] = branch_decoder_diagnostics(
            rows,
            eval_examples,
            logits_by_obs,
            eval_examples["target_logits"],
            action_names,
        )

    source_stats = source_response_stats(rows, eval_examples, action_names)
    source_stats["method"] = "Source Counterfactual Response"
    source_stats.update(branch_flip_diagnostics(rows, eval_examples, eval_examples["source_logits"], eval_examples["target_logits"], action_names))
    ceiling_logits = rule_constrained_branch_logits(rows, eval_examples, eval_examples["target_logits"], action_names)
    ceiling = branch_metrics(rows, eval_examples, ceiling_logits, action_names)
    ceiling["method"] = "Rule-Constrained Branch Ceiling"
    ceiling.update(branch_flip_diagnostics(rows, eval_examples, ceiling_logits, eval_examples["target_logits"], action_names))

    metadata = {
        "num_train_prefixes": len(train_selected),
        "num_eval_prefixes": len(eval_selected),
        "num_observation_bank_entries": sum(len(values) for values in bank.values()),
        "num_skills": args.num_skills,
        "skill_dim": args.skill_dim,
        "response_rank": args.response_rank,
        "binder_rank": args.binder_rank,
        "alpha": args.alpha,
        "critical_impact_threshold": args.critical_impact_threshold,
        "branch_margin_threshold": args.branch_margin_threshold,
        "critical_top_quantile": args.critical_top_quantile,
        "write_pair_top_quantile": args.write_pair_top_quantile,
        "write_branch_strength": args.write_branch_strength,
        "write_pair_calibration_margin": args.write_pair_calibration_margin,
        "write_pair_max_calibration": args.write_pair_max_calibration,
        "branch_aux_weight": args.branch_aux_weight,
        "obs_variants": args.obs_variants,
        "fingerprint": args.fingerprint,
        "split_metadata": split_metadata,
        "source_model_id": args.source_model_id,
        "target_model_id": args.target_model_id,
    }
    localization = response_localization_audit(eval_examples, action_names, "eval")
    source_branch_oracle_diag = gap_diagnostics(eval_examples, source_branch_oracle_broadcast_logits, eval_examples["target_logits"])
    oracle_operator_diag = gap_diagnostics(eval_examples, oracle_branch_operator_broadcast_logits, eval_examples["target_logits"])
    pred_operator_diag = gap_diagnostics(eval_examples, pred_branch_operator_broadcast_logits, eval_examples["target_logits"])
    branch_direction_diag = branch_response_direction_audit(rows, eval_examples, source_branch_residuals_eval, action_names)
    eval_branch_critical_mask, eval_branch_stats = source_validated_branch_critical_mask(
        rows,
        eval_examples,
        action_names,
        args.branch_margin_threshold,
        args.critical_top_quantile,
    )
    eval_write_pair_mask, write_pair_eval_stats = write_pair_critical_mask(
        rows,
        eval_examples,
        action_names,
        args.write_pair_top_quantile,
    )
    train_skill_counts = Counter(train_source_skill_ids.tolist())
    critical_skill_counts = Counter(train_source_skill_ids[branch_critical_mask].tolist())
    binder_diag = skill_binder_diagnostics(pred_skill_probs, eval_source_skill_ids)
    prototype_binder_diag = skill_binder_diagnostics(prototype_skill_probs, eval_source_skill_ids)
    ridge_binder_diag = skill_binder_diagnostics(ridge_skill_probs, eval_source_skill_ids)
    result_by_method = {item["method"]: item for item in decoded_results}
    required_diagnostics = {
        "source_write_pair": source_stats["write_success_failure_pair_acc"],
        "source_unique_action_prefix": source_stats["unique_action_prefix_rate"],
        "source_delta_h_norm": localization["eval_source_delta_h_norm"],
        "source_delta_z_norm": localization["eval_source_delta_z_norm"],
        "source_delta_logit_norm": localization["eval_source_delta_logit_norm"],
        "target_delta_h_norm": localization["eval_target_delta_h_norm"],
        "target_delta_z_norm": localization["eval_target_delta_z_norm"],
        "target_delta_logit_norm": localization["eval_target_delta_logit_norm"],
        "num_total_prefixes": int(eval_branch_stats["num_total_prefixes"]),
        "num_source_validated_critical_prefixes": int(eval_branch_stats["num_source_validated_critical_prefixes"]),
        "critical_prefix_rate": eval_branch_stats["critical_prefix_rate"],
        "source_branch_flip_rate": eval_branch_stats["source_branch_flip_rate"],
        "target_branch_flip_rate": eval_branch_stats["target_branch_flip_rate"],
        "source_branch_margin_delta": eval_branch_stats["source_branch_margin_delta"],
        "target_branch_margin_delta": eval_branch_stats["target_branch_margin_delta"],
        "branch_direction_correct_rate": branch_direction_diag["branch_direction_correct_rate"],
        "median_branch_flip_scale": branch_direction_diag["median_branch_flip_scale"],
        "flippable_branch_alpha_1": branch_direction_diag["flippable_branch_alpha_1"],
        "flippable_branch_alpha_2": branch_direction_diag["flippable_branch_alpha_2"],
        "flippable_branch_alpha_5": branch_direction_diag["flippable_branch_alpha_5"],
        "binder_accuracy": binder_diag["skill_binder_accuracy"],
        "binder_entropy": binder_diag["skill_posterior_entropy"],
        "binder_confidence": binder_diag["top1_skill_confidence"],
        "prototype_binder_accuracy": prototype_binder_diag["skill_binder_accuracy"],
        "ridge_binder_accuracy": ridge_binder_diag["skill_binder_accuracy"],
        "oracle_source_branch_flip_rate": result_by_method["Oracle Source Branch Response + Broadcast"]["unique_action_prefix_rate"],
        "oracle_source_branch_correct_flip_rate": result_by_method["Oracle Source Branch Response + Broadcast"]["correct_flip_rate"],
        "oracle_source_branch_harmful_flip_rate": result_by_method["Oracle Source Branch Response + Broadcast"]["harmful_flip_rate"],
        "train_critical_prefix_count": int(branch_critical_mask.sum().item()),
        "eval_critical_prefix_count": int(eval_branch_critical_mask.sum().item()),
        "train_write_pair_critical_prefix_count": int(write_pair_mask.sum().item()),
        "eval_write_pair_critical_prefix_count": int(eval_write_pair_mask.sum().item()),
        "oracle_correct_write_branch_write_pair": result_by_method["Oracle Correct Write Branch + Soft Hierarchical"][
            "write_success_failure_pair_acc"
        ],
        "calibrated_write_pair_prior_write_pair": result_by_method["Calibrated Write-Pair Prior + Soft Hierarchical"][
            "write_success_failure_pair_acc"
        ],
        "oracle_skill_operator_write_pair": result_by_method["Oracle Skill + Branch Operator + Soft Hierarchical"][
            "write_success_failure_pair_acc"
        ],
        "predicted_skill_operator_write_pair": result_by_method["Predicted Skill + Branch Operator + Soft Hierarchical"][
            "write_success_failure_pair_acc"
        ],
        "oracle_pair_operator_write_pair": result_by_method["Oracle Skill + Pair-Contrastive Write Operator"][
            "write_success_failure_pair_acc"
        ],
        "predicted_pair_operator_write_pair": result_by_method["Predicted Skill MLP + Pair-Contrastive Write Operator"][
            "write_success_failure_pair_acc"
        ],
        "predicted_pair_plus_calibrated_prior_write_pair": result_by_method[
            "Predicted Pair Operator + Calibrated Write-Pair Prior"
        ]["write_success_failure_pair_acc"],
    }
    diagnostics = {
        "pre_observation_skill_binder": binder_diag,
        "prototype_skill_binder": prototype_binder_diag,
        "ridge_skill_binder": ridge_binder_diag,
        "response_signal_localization": localization,
        "source_validated_branch_critical_train": train_branch_stats,
        "source_validated_branch_critical_eval": eval_branch_stats,
        "write_pair_critical_train": write_pair_train_stats,
        "write_pair_critical_eval": write_pair_eval_stats,
        "branch_direction_and_scale": branch_direction_diag,
        "oracle_source_branch_response": source_branch_oracle_diag,
        "oracle_skill_branch_operator": oracle_operator_diag,
        "predicted_skill_branch_operator": pred_operator_diag,
        "branch_first_decoder": decoder_diagnostics,
        "required_fields": required_diagnostics,
        "critical_skill_cluster_stats": {
            "critical_prefix_count": int(branch_critical_mask.sum().item()),
            "critical_fraction": float(branch_critical_mask.float().mean().item()),
            "train_skill_counts": {str(k): int(v) for k, v in sorted(train_skill_counts.items())},
            "critical_skill_counts": {str(k): int(v) for k, v in sorted(critical_skill_counts.items())},
        },
    }
    payload = {
        "metadata": metadata,
        "source_response_stats": source_stats,
        "diagnostics": diagnostics,
        "results": [source_stats, static, *decoded_results, ceiling],
    }
    out_dir.joinpath("counterfactual_eval_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    prototypes = skill_prototypes(train_examples, branch_centroids, args.skill_temperature, train_branch_fingerprint)
    torch.save(
        {"centroids": branch_centroids, "obs_types": OBS_TYPES, "branches": BRANCHES, "prototypes": prototypes},
        out_dir / "branch_response_skill_centroids.pt",
    )
    torch.save(
        {
            "state_dict": skill_binder.state_dict(),
            "num_skills": args.num_skills,
            "binder_rank": args.binder_rank,
            "z_dim": int(train_z_pre.shape[1]),
        },
        out_dir / "pre_observation_skill_binder.pt",
    )
    torch.save(
        {
            "state_dict": branch_library.state_dict(),
            "obs_types": OBS_TYPES,
            "branches": BRANCHES,
            "num_skills": args.num_skills,
            "skill_dim": args.skill_dim,
            "rank": args.rank,
        },
        out_dir / "branch_response_operator.pt",
    )
    torch.save(
        {
            "state_dict": write_pair_library.state_dict(),
            "obs_types": OBS_TYPES,
            "branches": BRANCHES,
            "num_skills": args.num_skills,
            "skill_dim": args.skill_dim,
            "rank": args.rank,
            "write_pair_critical_stats": write_pair_train_stats,
        },
        out_dir / "pair_contrastive_write_operator.pt",
    )
    train_stats = {
        "source_response_stats": source_response_stats(rows, train_examples, action_names),
        "skill_counts": {str(item["skill_id"]): item["count"] for item in prototypes},
        "write_pair_critical_stats": write_pair_train_stats,
    }
    out_dir.joinpath("counterfactual_train_stats.json").write_text(
        json.dumps(train_stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_report(out_dir, metadata, [source_stats, static, *decoded_results, ceiling])

    sample_prompts = []
    rng = random.Random(args.seed)
    for idx in eval_selected[: min(5, len(eval_selected))]:
        sample_prompts.append({"row_index": idx, "bundle": build_cf_bundle(rows[idx], bank, rng, args.obs_variants)})
    out_dir.joinpath("sample_counterfactual_prompts.json").write_text(
        json.dumps(sample_prompts, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
