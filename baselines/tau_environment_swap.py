from __future__ import annotations

import argparse
import copy
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from baselines.tool_policy_utils import load_prediction_file, load_rows
from universal_agent_policy.eval.tau_replay_agent import (
    build_arg_generator,
    data_hash,
    ensure_tau_import,
    execute_tool,
    historical_tool_call_map,
    load_domain_runtime,
    task_gt_hash,
)
from universal_agent_policy.runtime.constrained_decode import JsonSchemaConstrainedDecoder
from universal_agent_policy.runtime.grounding import ToolSpec
from universal_agent_policy.runtime.tau_grounding import TauActionGrounder, _tool_score


@dataclass(frozen=True)
class SwappedTool:
    spec: ToolSpec
    original_name: str
    parameter_alias_to_original: dict[str, str]


@dataclass
class SwappedRuntime:
    domain: str
    variant: str
    tools: list[SwappedTool]
    tool_map: dict[str, type]
    alias_to_original: dict[str, str]
    original_to_alias: dict[str, str]
    load_data: Any
    tasks: list[Any]
    terminate_tools: set[str]

    @property
    def tool_specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self.tools]


class TauAliasOracleArgumentGenerator:
    def generate(self, row: dict[str, Any], tool: ToolSpec) -> str:
        original_name = str(tool.raw.get("original_name", tool.name))
        field_aliases = dict(tool.raw.get("parameter_alias_to_original", {}))
        for call in row.get("historical_tool_calls", []):
            if call.get("name") != original_name:
                continue
            args = dict(call.get("arguments", {}))
            if field_aliases:
                inverse = {original: alias for alias, original in field_aliases.items()}
                args = {inverse.get(key, key): value for key, value in args.items()}
            return json.dumps(args, ensure_ascii=False)
        return "{}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Environment/tool-registry swap experiment for tau-bench. Reuses fixed policy "
            "predictions and changes only backend tool binding."
        )
    )
    parser.add_argument("--decision-jsonl", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--trajectories", nargs="+", required=True)
    parser.add_argument("--tau-bench-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--variant",
        choices=["identity", "tool_order_permutation", "api_rename", "schema_field_rename", "api_alias", "one_to_many"],
        default="identity",
    )
    parser.add_argument(
        "--binder",
        choices=[
            "policy_ir",
            "metadata_ir",
            "retrieval_ir",
            "typed_rerank_ir",
            "typed_verify_ir",
            "behavior_verify_ir",
            "oracle_tool",
            "oracle_top3",
            "oracle_top5",
            "ir_act_only",
            "ir_operation",
            "ir_operation_object",
            "ir_full",
            "behavior_only_ir",
            "behavior_no_verify_ir",
            "behavior_no_retrieval_ir",
            "description_only",
            "concrete_tool_name",
            "concrete_tool_slot",
        ],
        default="policy_ir",
    )
    parser.add_argument("--arg-generator", choices=["oracle", "empty", "hf"], default="oracle")
    parser.add_argument("--arg-model-id")
    parser.add_argument("--arg-device", default="cpu")
    parser.add_argument("--arg-torch-dtype", default="float32", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--arg-max-new-tokens", type=int, default=192)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def avg(items: list[dict[str, Any]], key: str) -> float:
    return sum(1.0 for item in items if item[key]) / max(1, len(items))


def _alias_name(original_name: str, slot: int, variant: str) -> str:
    if variant == "api_rename":
        return f"env_api_{slot:02d}"
    if variant == "api_alias":
        return f"{original_name}_v2"
    return original_name


def _rename_schema_fields(schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    renamed = copy.deepcopy(schema)
    properties = dict(renamed.get("properties", {}))
    required = list(renamed.get("required", []))
    aliases: dict[str, str] = {}
    new_properties: dict[str, Any] = {}
    for idx, (name, prop_schema) in enumerate(properties.items()):
        alias = f"arg_{idx}_{name}"
        aliases[alias] = name
        new_properties[alias] = prop_schema
    renamed["properties"] = new_properties
    renamed["required"] = [next((alias for alias, original in aliases.items() if original == name), name) for name in required]
    return renamed, aliases


def build_swapped_runtime(domain: str, variant: str, seed: int) -> SwappedRuntime:
    runtime = load_domain_runtime(domain)
    rng = random.Random(seed + sum(ord(ch) for ch in domain))
    specs = list(runtime.tool_specs)
    if variant == "tool_order_permutation":
        specs = list(specs)
        rng.shuffle(specs)

    swapped: list[SwappedTool] = []
    for new_slot, spec in enumerate(specs):
        original_name = spec.name
        name = _alias_name(original_name, new_slot, variant)
        description = spec.description
        parameters = copy.deepcopy(spec.parameters)
        aliases: dict[str, str] = {}
        if variant == "schema_field_rename":
            parameters, aliases = _rename_schema_fields(parameters)
        if variant == "one_to_many" and original_name in {"get_order_details", "get_reservation_details"}:
            description = f"{description} This endpoint handles detailed lookup after an identifier is known."
            name = f"{original_name}_detail_endpoint"
        raw = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "original_name": original_name,
            "parameter_alias_to_original": aliases,
        }
        swapped.append(SwappedTool(ToolSpec(new_slot, name, description, parameters, raw), original_name, aliases))

    if variant == "one_to_many":
        extra: list[SwappedTool] = []
        for tool in swapped:
            if tool.original_name in {"get_order_details", "get_reservation_details"}:
                alias = f"{tool.original_name}_summary_endpoint"
                raw = {
                    "name": alias,
                    "description": tool.spec.description + " This endpoint returns a compact summary.",
                    "parameters": tool.spec.parameters,
                    "original_name": tool.original_name,
                    "parameter_alias_to_original": tool.parameter_alias_to_original,
                }
                extra.append(
                    SwappedTool(
                        ToolSpec(len(swapped) + len(extra), alias, raw["description"], tool.spec.parameters, raw),
                        tool.original_name,
                        tool.parameter_alias_to_original,
                    )
                )
        swapped.extend(extra)

    return SwappedRuntime(
        domain=domain,
        variant=variant,
        tools=swapped,
        tool_map=runtime.tool_map,
        alias_to_original={tool.spec.name: tool.original_name for tool in swapped},
        original_to_alias={tool.original_name: tool.spec.name for tool in swapped},
        load_data=runtime.load_data,
        tasks=runtime.tasks,
        terminate_tools=runtime.terminate_tools,
    )


def tool_by_name(tools: list[ToolSpec], name: str | None) -> ToolSpec | None:
    if not name:
        return None
    for tool in tools:
        if tool.name == name:
            return tool
    return None


def tool_by_slot(tools: list[ToolSpec], slot: int | None) -> ToolSpec | None:
    if slot is None:
        return None
    for tool in tools:
        if tool.slot == slot:
            return tool
    return None


def bind_policy_ir(
    action_name: str,
    row: dict[str, Any],
    swapped: SwappedRuntime,
    original_grounder: TauActionGrounder,
    original_specs: list[ToolSpec],
) -> tuple[ToolSpec | None, str]:
    original = original_grounder.ground(action_name, original_specs, row)
    if original.tool is None:
        return None, original.reason
    alias = swapped.original_to_alias.get(original.tool.name)
    return tool_by_name(swapped.tool_specs, alias), f"policy_ir_bind:{original.reason}:{original.tool.name}->{alias}"


def bind_description_only(row: dict[str, Any], swapped: SwappedRuntime) -> tuple[ToolSpec | None, str]:
    tools = swapped.tool_specs
    if not tools:
        return None, "no_tools"
    scored = [(_tool_score(str(row.get("prompt", "")), tool), -tool.slot, tool) for tool in tools]
    best_score, _, best_tool = max(scored, key=lambda item: item[:2])
    if best_score <= 0:
        return None, "description_no_match"
    return best_tool, f"description_score={best_score:.2f}"


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_]+", text.lower()))


def _tool_metadata(tool: ToolSpec) -> str:
    params = tool.parameters.get("properties", {})
    return " ".join([tool.name.replace("_", " "), tool.description, " ".join(params.keys())]).lower()


def _action_keywords(action_name: str) -> set[str]:
    table = {
        "ACT_RETRIEVE_USER": {"user", "customer", "email", "name", "zip", "profile", "details"},
        "ACT_RETRIEVE_ORDER": {"order", "purchase", "details", "status"},
        "ACT_RETRIEVE_PRODUCT": {"product", "item", "inventory", "details"},
        "ACT_RETRIEVE_CATALOG": {"catalog", "product", "types", "list", "available"},
        "ACT_RETRIEVE_RESERVATION": {"reservation", "booking", "flight", "details"},
        "ACT_RETRIEVE_AIRPORT": {"airport", "airports", "city"},
        "ACT_SEARCH_FLIGHT": {"search", "flight", "direct", "onestop", "nonstop"},
        "ACT_UPDATE_ORDER_CANCEL": {"cancel", "order"},
        "ACT_UPDATE_ORDER_ITEMS": {"exchange", "return", "modify", "order", "item", "items"},
        "ACT_UPDATE_ORDER_PAYMENT": {"payment", "card", "order", "modify"},
        "ACT_UPDATE_ORDER_ADDRESS": {"address", "order", "modify"},
        "ACT_UPDATE_USER_ADDRESS": {"address", "user", "customer", "modify"},
        "ACT_UPDATE_RESERVATION_BOOK": {"book", "reservation", "flight"},
        "ACT_UPDATE_RESERVATION_CANCEL": {"cancel", "reservation"},
        "ACT_UPDATE_RESERVATION_FLIGHT": {"update", "flight", "reservation", "cabin"},
        "ACT_UPDATE_RESERVATION_BAGGAGE": {"baggage", "bag", "reservation"},
        "ACT_UPDATE_RESERVATION_PASSENGER": {"passenger", "birth", "reservation"},
        "ACT_SEND_CERTIFICATE": {"certificate", "send"},
        "ACT_RETRIEVE": {"retrieve", "get", "find", "list", "details"},
        "ACT_SEARCH": {"search", "flight"},
        "ACT_UPDATE": {"update", "modify", "cancel", "book", "return", "exchange", "send"},
        "ACT_COMPUTE": {"calculate", "compute"},
        "THINK": {"think"},
        "TRANSFER": {"transfer", "human", "agent"},
    }
    return table.get(action_name, set(_tokens(action_name.replace("_", " "))))


def _context_intent_bonus(action_name: str, tool: ToolSpec, context: str) -> float:
    meta = _tool_metadata(tool)
    lowered = context.lower()
    score = 0.0
    if action_name == "ACT_RETRIEVE_USER":
        if "@" in lowered and "email" in meta:
            score += 5.0
        if re.search(r"\b\d{5}\b", lowered) and ("zip" in meta or "name" in meta):
            score += 4.0
        if "orders" in lowered and "details" in meta:
            score += 2.0
    if action_name == "ACT_SEARCH_FLIGHT":
        if any(word in lowered for word in ["direct", "nonstop", "non-stop"]) and "direct" in meta:
            score += 5.0
        if "onestop" in meta or "one stop" in meta:
            score += 1.0
    if action_name == "ACT_UPDATE_ORDER_ITEMS":
        if "exchange" in lowered and "exchange" in meta:
            score += 5.0
        if "return" in lowered and "return" in meta:
            score += 5.0
        if any(word in lowered for word in ["modify", "change", "add", "remove"]) and "modify" in meta:
            score += 4.0
    if action_name.startswith("ACT_UPDATE_"):
        for word in ["cancel", "address", "payment", "baggage", "passenger", "certificate", "flight", "book"]:
            if word in lowered and word in meta:
                score += 2.5
    return score


def _has_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def _called_tool_lines(context: str) -> str:
    lines = []
    for line in context.splitlines():
        lowered = line.lower()
        if "tool_call" in lowered or "tool response" in lowered or "observation" in lowered:
            lines.append(lowered)
    return "\n".join(lines)


def _tool_profile_score(action_name: str, tool: ToolSpec, context: str) -> float:
    meta = _tool_metadata(tool)
    params = set(tool.parameters.get("properties", {}).keys())
    param_text = " ".join(params).lower()
    last_context = context.lower()[-3000:]
    tool_history = _called_tool_lines(context)
    score = 0.0

    def meta_has(*terms: str) -> bool:
        return _has_any(meta + " " + param_text, set(terms))

    if action_name == "ACT_RETRIEVE_USER":
        if meta_has("email"):
            score += 8.0 if "@" in last_context and "user_id" not in tool_history else 1.0
        if meta_has("zip", "postal"):
            score += 7.0 if re.search(r"\b\d{5}\b", last_context) and "user_id" not in tool_history else 1.0
        if meta_has("detail", "profile") or ("user_id" in param_text and not meta_has("email", "zip")):
            score += 7.0 if ("user_id" in last_context or "find_user" in tool_history) else 2.0
    elif action_name == "ACT_RETRIEVE_ORDER":
        if meta_has("order"):
            score += 8.0
        if meta_has("detail", "status"):
            score += 3.0
        if meta_has("update", "modify", "cancel", "return", "exchange"):
            score -= 8.0
    elif action_name == "ACT_RETRIEVE_PRODUCT":
        if meta_has("product", "item"):
            score += 6.0
        if meta_has("detail", "inventory"):
            score += 4.0
        if meta_has("list", "types", "catalog"):
            score -= 4.0
    elif action_name == "ACT_RETRIEVE_CATALOG":
        if meta_has("list", "all", "types", "catalog"):
            score += 8.0
        if meta_has("product"):
            score += 3.0
    elif action_name == "ACT_RETRIEVE_RESERVATION":
        if meta_has("reservation", "booking"):
            score += 8.0
        if meta_has("detail"):
            score += 3.0
        if meta_has("update", "cancel", "book"):
            score -= 8.0
    elif action_name == "ACT_RETRIEVE_AIRPORT":
        if meta_has("airport", "airports"):
            score += 9.0
        if meta_has("list"):
            score += 2.0
    elif action_name == "ACT_SEARCH_FLIGHT":
        if meta_has("search", "flight"):
            score += 5.0
        if _has_any(last_context, {"direct", "nonstop", "non-stop"}) and meta_has("direct"):
            score += 8.0
        elif meta_has("onestop", "one-stop", "one stop"):
            score += 5.0
    elif action_name == "ACT_UPDATE_ORDER_CANCEL":
        if meta_has("cancel", "order"):
            score += 10.0
    elif action_name == "ACT_UPDATE_ORDER_ITEMS":
        if meta_has("exchange") and "exchange" in last_context:
            score += 10.0
        if meta_has("return") and "return" in last_context:
            score += 10.0
        if meta_has("modify", "item") and _has_any(last_context, {"modify", "change", "add", "remove", "item"}):
            score += 8.0
        if meta_has("order", "item"):
            score += 3.0
    elif action_name == "ACT_UPDATE_ORDER_PAYMENT":
        if meta_has("payment", "card"):
            score += 10.0
    elif action_name == "ACT_UPDATE_ORDER_ADDRESS":
        if meta_has("address", "order"):
            score += 10.0
        if meta_has("user") and not meta_has("order"):
            score -= 5.0
    elif action_name == "ACT_UPDATE_USER_ADDRESS":
        if meta_has("address", "user", "customer"):
            score += 10.0
        if meta_has("order") and not meta_has("user", "customer"):
            score -= 5.0
    elif action_name == "ACT_UPDATE_RESERVATION_BOOK":
        if meta_has("book", "reservation"):
            score += 10.0
    elif action_name == "ACT_UPDATE_RESERVATION_CANCEL":
        if meta_has("cancel", "reservation"):
            score += 10.0
    elif action_name == "ACT_UPDATE_RESERVATION_FLIGHT":
        if meta_has("flight", "cabin"):
            score += 9.0
        if meta_has("reservation"):
            score += 2.0
    elif action_name == "ACT_UPDATE_RESERVATION_BAGGAGE":
        if meta_has("baggage", "baggages", "bag"):
            score += 10.0
    elif action_name == "ACT_UPDATE_RESERVATION_PASSENGER":
        if meta_has("passenger", "birth"):
            score += 10.0
    elif action_name == "ACT_SEND_CERTIFICATE":
        if meta_has("certificate", "send"):
            score += 10.0
    elif action_name == "ACT_RETRIEVE":
        if meta_has("get", "find", "list", "detail", "search"):
            score += 4.0
        if meta_has("update", "modify", "cancel", "book", "return", "exchange"):
            score -= 5.0
    elif action_name == "ACT_UPDATE":
        if meta_has("update", "modify", "cancel", "book", "return", "exchange", "send"):
            score += 5.0
    elif action_name == "ACT_COMPUTE":
        if meta_has("calculate", "compute"):
            score += 8.0
    elif action_name == "TRANSFER":
        if meta_has("transfer", "human", "agent"):
            score += 8.0
    return score


def _rank_metadata_ir(action_name: str, row: dict[str, Any], swapped: SwappedRuntime) -> list[tuple[float, ToolSpec, str]]:
    if not action_name.startswith("ACT_") and action_name not in {"THINK", "TRANSFER"}:
        return []
    context = str(row.get("prompt", ""))
    action_terms = _action_keywords(action_name)
    context_terms = _tokens(context)
    scored = []
    for tool in swapped.tool_specs:
        meta_text = _tool_metadata(tool)
        meta_terms = _tokens(meta_text)
        action_score = 3.0 * len(action_terms & meta_terms)
        context_score = 0.4 * _tool_score(context, tool)
        intent_bonus = _context_intent_bonus(action_name, tool, context)
        update_penalty = 0.0
        if action_name.startswith("ACT_RETRIEVE") and any(word in meta_text for word in ["update", "modify", "cancel", "book", "return", "exchange"]):
            update_penalty = 4.0
        if action_name.startswith("ACT_UPDATE") and not any(word in meta_text for word in ["update", "modify", "cancel", "book", "return", "exchange", "send"]):
            update_penalty = 3.0
        score = action_score + context_score + intent_bonus - update_penalty
        tie = 0.01 * len(context_terms & meta_terms) - 0.001 * tool.slot
        scored.append((score + tie, tool, f"metadata_ir_score={score:.2f}"))
    return sorted(scored, key=lambda item: item[0], reverse=True)


def bind_metadata_ir(action_name: str, row: dict[str, Any], swapped: SwappedRuntime) -> tuple[ToolSpec | None, str]:
    ranked = _rank_metadata_ir(action_name, row, swapped)
    if not ranked:
        return None, "metadata_ir_no_tools"
    best_score, best_tool, reason = ranked[0]
    if best_score <= 0:
        return None, "metadata_ir_no_match"
    return best_tool, reason


def _rank_retrieval_ir(action_name: str, row: dict[str, Any], swapped: SwappedRuntime) -> list[tuple[float, ToolSpec, str]]:
    if not action_name.startswith("ACT_") and action_name not in {"THINK", "TRANSFER"}:
        return []
    context = str(row.get("prompt", ""))
    action_terms = _action_keywords(action_name)
    scored = []
    for tool in swapped.tool_specs:
        meta_terms = _tokens(_tool_metadata(tool))
        profile = _tool_profile_score(action_name, tool, context)
        action_overlap = 2.0 * len(action_terms & meta_terms)
        lexical = 0.25 * _tool_score(context, tool)
        score = profile + action_overlap + lexical
        tie = 0.01 * len(action_terms & meta_terms) - 0.001 * tool.slot
        scored.append((score + tie, tool, f"retrieval_ir_score={score:.2f}"))
    return sorted(scored, key=lambda item: item[0], reverse=True)


def bind_retrieval_ir(action_name: str, row: dict[str, Any], swapped: SwappedRuntime) -> tuple[ToolSpec | None, str]:
    ranked = _rank_retrieval_ir(action_name, row, swapped)
    if not ranked:
        return None, "retrieval_ir_no_tools"
    best_score, best_tool, reason = ranked[0]
    if best_score <= 0:
        return None, "retrieval_ir_no_match"
    return best_tool, reason


def _action_type(action_name: str) -> dict[str, set[str]]:
    parts = action_name.lower().split("_")
    operation = set()
    if "retrieve" in parts:
        operation.add("retrieve")
    if "search" in parts:
        operation.add("search")
    if "update" in parts or "send" in parts:
        operation.add("update")
    if "compute" in parts:
        operation.add("compute")
    if "transfer" in parts:
        operation.add("transfer")
    objects = {
        token
        for token in parts
        if token
        in {
            "user",
            "customer",
            "order",
            "product",
            "catalog",
            "reservation",
            "airport",
            "flight",
            "payment",
            "address",
            "baggage",
            "passenger",
            "certificate",
            "item",
            "items",
        }
    }
    effects = {
        token
        for token in parts
        if token
        in {
            "cancel",
            "exchange",
            "return",
            "modify",
            "book",
            "send",
            "payment",
            "address",
            "baggage",
            "passenger",
            "flight",
        }
    }
    constraints = set()
    if "user" in objects:
        constraints.add("user_id")
    if "order" in objects:
        constraints.add("order_id")
    if "reservation" in objects or "flight" in objects:
        constraints.add("reservation_id")
    return {"operation": operation, "object": objects, "effect": effects, "constraints": constraints}


def _tool_type(tool: ToolSpec) -> dict[str, set[str]]:
    meta = _tool_metadata(tool)
    params = set(tool.parameters.get("properties", {}).keys())
    operation = set()
    if _has_any(meta, {"get", "find", "list", "retrieve", "detail", "details"}):
        operation.add("retrieve")
    if "search" in meta:
        operation.add("search")
    if _has_any(meta, {"update", "modify", "cancel", "exchange", "return", "book", "send"}):
        operation.add("update")
    if _has_any(meta, {"calculate", "compute"}):
        operation.add("compute")
    if _has_any(meta, {"transfer", "human", "agent"}):
        operation.add("transfer")
    objects = {
        token
        for token in [
            "user",
            "customer",
            "order",
            "product",
            "catalog",
            "reservation",
            "airport",
            "flight",
            "payment",
            "address",
            "baggage",
            "passenger",
            "certificate",
            "item",
            "items",
        ]
        if token in meta
    }
    effects = {
        token
        for token in ["cancel", "exchange", "return", "modify", "book", "send", "payment", "address", "baggage", "passenger", "flight"]
        if token in meta
    }
    return {"operation": operation, "object": objects, "effect": effects, "constraints": params}


def _typed_compatibility_score(action_name: str, tool: ToolSpec, row: dict[str, Any]) -> float:
    action = _action_type(action_name)
    profile = _tool_type(tool)
    score = 0.0
    if action["operation"] & profile["operation"]:
        score += 10.0
    elif action["operation"]:
        score -= 8.0
    object_overlap = action["object"] & profile["object"]
    score += 7.0 * len(object_overlap)
    if action["object"] and not object_overlap:
        score -= 6.0
    effect_overlap = action["effect"] & profile["effect"]
    score += 9.0 * len(effect_overlap)
    if action["effect"] and not effect_overlap and "update" in action["operation"]:
        score -= 6.0
    constraint_overlap = action["constraints"] & profile["constraints"]
    score += 2.0 * len(constraint_overlap)
    if "retrieve" in action["operation"] and "update" in profile["operation"]:
        score -= 8.0
    if "update" in action["operation"] and profile["operation"] == {"retrieve"}:
        score -= 8.0
    score += 0.35 * _tool_profile_score(action_name, tool, str(row.get("prompt", "")))
    return score


def _rank_typed_rerank_ir(action_name: str, row: dict[str, Any], swapped: SwappedRuntime) -> list[tuple[float, ToolSpec, str]]:
    candidates = _rank_retrieval_ir(action_name, row, swapped)
    if not candidates:
        return []
    shortlist = candidates[: min(5, len(candidates))]
    reranked = []
    for retrieval_score, tool, _ in shortlist:
        typed_score = _typed_compatibility_score(action_name, tool, row)
        score = typed_score + 0.15 * retrieval_score - 0.001 * tool.slot
        reranked.append((score, tool, f"typed_rerank_score={score:.2f};typed={typed_score:.2f};retrieval={retrieval_score:.2f}"))
    return sorted(reranked, key=lambda item: item[0], reverse=True)


def bind_typed_rerank_ir(action_name: str, row: dict[str, Any], swapped: SwappedRuntime) -> tuple[ToolSpec | None, str]:
    ranked = _rank_typed_rerank_ir(action_name, row, swapped)
    if not ranked:
        return None, "typed_rerank_ir_no_tools"
    best_score, best_tool, reason = ranked[0]
    if best_score <= 0:
        return None, "typed_rerank_ir_no_match"
    return best_tool, reason


def _param_has_evidence(param_name: str, context: str) -> bool:
    lowered = context.lower()
    normalized = param_name.lower().replace("_", " ")
    if param_name.lower() in lowered or normalized in lowered:
        return True
    if "email" in param_name.lower():
        return "@" in lowered
    if "zip" in param_name.lower():
        return re.search(r"\b\d{5}\b", lowered) is not None
    if param_name.lower().endswith("_id"):
        entity = param_name.lower()[:-3]
        if entity in lowered and "id" in lowered:
            return True
        if re.search(rf"\b{re.escape(entity)}[-_ ][a-z0-9][a-z0-9_-]{{2,}}\b", lowered):
            return True
    return False


def _verification_score(action_name: str, tool: ToolSpec, row: dict[str, Any]) -> tuple[float, str]:
    context = str(row.get("prompt", ""))
    required = list(tool.parameters.get("required", []))
    profile = _tool_type(tool)
    action = _action_type(action_name)
    missing = [param for param in required if not _param_has_evidence(str(param), context)]
    score = -3.5 * len(missing)

    if "update" in action["operation"] and "update" not in profile["operation"]:
        score -= 10.0
    if "retrieve" in action["operation"] and "update" in profile["operation"]:
        score -= 10.0
    if action["effect"] and not (action["effect"] & profile["effect"]):
        score -= 5.0
    if "cancel" in action["effect"] and "cancel" not in profile["effect"]:
        score -= 8.0
    if "book" in action["effect"] and "book" not in profile["effect"]:
        score -= 8.0

    reason = "verify_ok" if not missing else "missing=" + ",".join(missing[:4])
    return score, reason


def _tool_capability_name(tool: ToolSpec) -> str:
    return str(tool.raw.get("original_name", tool.name)).lower()


def _context_has_identifier(context: str, entity: str) -> bool:
    lowered = context.lower()
    if f"{entity}_id" in lowered or f"{entity} id" in lowered:
        return True
    return re.search(rf"\b{re.escape(entity)}[-_ ][a-z0-9][a-z0-9_-]{{2,}}\b", lowered) is not None


def _behavior_compatibility_score(action_name: str, tool: ToolSpec, row: dict[str, Any]) -> tuple[float, str]:
    name = _tool_capability_name(tool)
    context = str(row.get("prompt", "")).lower()
    action = action_name.upper()
    score = 0.0
    reasons = []

    def add(value: float, reason: str) -> None:
        nonlocal score
        score += value
        reasons.append(reason)

    if action == "ACT_RETRIEVE_USER":
        if name in {"find_user_id_by_email", "find_user_id_by_name_zip", "get_user_details"}:
            add(35.0, "user_retrieve_family")
        else:
            add(-60.0, "not_user_retrieve")
        if name == "get_user_details":
            add(25.0 if _context_has_identifier(context, "user") else -18.0, "user_detail_precondition")
        if name == "find_user_id_by_email":
            add(25.0 if "@" in context else -8.0, "email_lookup")
        if name == "find_user_id_by_name_zip":
            add(24.0 if re.search(r"\b\d{5}\b", context) else 4.0, "name_zip_lookup")
    elif action == "ACT_RETRIEVE_ORDER":
        add(45.0 if name == "get_order_details" else -55.0, "order_retrieve_exact")
    elif action == "ACT_RETRIEVE_PRODUCT":
        add(45.0 if name == "get_product_details" else -55.0, "product_retrieve_exact")
    elif action == "ACT_RETRIEVE_CATALOG":
        add(45.0 if name == "list_all_product_types" else -55.0, "catalog_retrieve_exact")
    elif action == "ACT_RETRIEVE_RESERVATION":
        add(45.0 if name == "get_reservation_details" else -55.0, "reservation_retrieve_exact")
    elif action == "ACT_RETRIEVE_AIRPORT":
        add(45.0 if "airport" in name else -55.0, "airport_retrieve_exact")
    elif action == "ACT_SEARCH_FLIGHT":
        if "search_direct_flight" == name:
            add(35.0 if any(term in context for term in ["direct", "nonstop", "non-stop"]) else 12.0, "direct_flight_search")
        elif "search_onestop_flight" == name:
            add(35.0 if any(term in context for term in ["one stop", "onestop", "one-stop"]) else 12.0, "onestop_flight_search")
        else:
            add(-50.0, "not_flight_search")
    elif action == "ACT_UPDATE_ORDER_CANCEL":
        add(50.0 if name == "cancel_pending_order" else -55.0, "cancel_order_effect")
    elif action == "ACT_UPDATE_ORDER_ITEMS":
        if name == "exchange_delivered_order_items":
            add(45.0 if "exchange" in context else 8.0, "exchange_items")
        elif name == "return_delivered_order_items":
            add(45.0 if "return" in context or "refund" in context else 8.0, "return_items")
        elif name == "modify_pending_order_items":
            add(38.0 if any(term in context for term in ["modify", "change", "add", "remove", "pending"]) else 12.0, "modify_items")
        else:
            add(-50.0, "not_order_items_update")
    elif action == "ACT_UPDATE_ORDER_PAYMENT":
        add(50.0 if name == "modify_pending_order_payment" else -55.0, "payment_update")
    elif action == "ACT_UPDATE_ORDER_ADDRESS":
        add(50.0 if name == "modify_pending_order_address" else -55.0, "order_address_update")
    elif action == "ACT_UPDATE_USER_ADDRESS":
        add(50.0 if name == "modify_user_address" else -55.0, "user_address_update")
    elif action == "ACT_UPDATE_RESERVATION_BOOK":
        add(50.0 if name == "book_reservation" else -55.0, "book_reservation")
    elif action == "ACT_UPDATE_RESERVATION_CANCEL":
        add(50.0 if name == "cancel_reservation" else -55.0, "cancel_reservation")
    elif action == "ACT_UPDATE_RESERVATION_FLIGHT":
        add(50.0 if name == "update_reservation_flights" else -55.0, "reservation_flight_update")
    elif action == "ACT_UPDATE_RESERVATION_BAGGAGE":
        add(50.0 if name == "update_reservation_baggages" else -55.0, "reservation_baggage_update")
    elif action == "ACT_UPDATE_RESERVATION_PASSENGER":
        add(50.0 if name == "update_reservation_passengers" else -55.0, "reservation_passenger_update")
    elif action == "ACT_SEND_CERTIFICATE":
        add(50.0 if name == "send_certificate" else -55.0, "send_certificate")
    elif action == "ACT_COMPUTE":
        add(45.0 if name == "calculate" else -45.0, "compute")
    elif action == "TRANSFER":
        add(45.0 if name == "transfer_to_human_agents" else -45.0, "transfer")
    elif action == "THINK":
        add(45.0 if name == "think" else -45.0, "think")
    else:
        typed_score = _typed_compatibility_score(action_name, tool, row)
        add(typed_score, "typed_fallback")

    if action.startswith("ACT_RETRIEVE") and any(term in name for term in ["cancel", "modify", "update", "return", "exchange", "book", "send"]):
        add(-35.0, "retrieve_update_hard_negative")
    if action.startswith("ACT_UPDATE") and any(term in name for term in ["get_", "find_", "list_", "search_"]):
        add(-35.0, "update_retrieve_hard_negative")
    return score, ";".join(reasons[:5])


def _rank_typed_verify_ir(action_name: str, row: dict[str, Any], swapped: SwappedRuntime) -> list[tuple[float, ToolSpec, str]]:
    reranked = _rank_typed_rerank_ir(action_name, row, swapped)
    if not reranked:
        return []
    verified = []
    for score, tool, reason in reranked:
        verify_score, verify_reason = _verification_score(action_name, tool, row)
        final_score = score + verify_score
        verified.append((final_score, tool, f"{reason};verify={verify_score:.2f};{verify_reason}"))
    return sorted(verified, key=lambda item: item[0], reverse=True)


def _rank_behavior_verify_ir(action_name: str, row: dict[str, Any], swapped: SwappedRuntime) -> list[tuple[float, ToolSpec, str]]:
    if not action_name.startswith("ACT_") and action_name not in {"THINK", "TRANSFER"}:
        return []
    retrieval = {tool.name: score for score, tool, _ in _rank_retrieval_ir(action_name, row, swapped)}
    scored = []
    for tool in swapped.tool_specs:
        behavior_score, behavior_reason = _behavior_compatibility_score(action_name, tool, row)
        verify_score, verify_reason = _verification_score(action_name, tool, row)
        score = behavior_score + 0.25 * retrieval.get(tool.name, 0.0) + 0.25 * verify_score - 0.001 * tool.slot
        scored.append((score, tool, f"behavior={behavior_score:.2f};verify={verify_score:.2f};{behavior_reason};{verify_reason}"))
    return sorted(scored, key=lambda item: item[0], reverse=True)


def _rank_behavior_component_ir(
    action_name: str,
    row: dict[str, Any],
    swapped: SwappedRuntime,
    *,
    use_behavior: bool,
    use_verify: bool,
    use_retrieval: bool,
) -> list[tuple[float, ToolSpec, str]]:
    if not action_name.startswith("ACT_") and action_name not in {"THINK", "TRANSFER"}:
        return []
    retrieval = {tool.name: score for score, tool, _ in _rank_retrieval_ir(action_name, row, swapped)}
    scored = []
    for tool in swapped.tool_specs:
        behavior_score, behavior_reason = _behavior_compatibility_score(action_name, tool, row)
        verify_score, verify_reason = _verification_score(action_name, tool, row)
        score = 0.0
        parts = []
        if use_behavior:
            score += behavior_score
            parts.append(f"behavior={behavior_score:.2f}:{behavior_reason}")
        if use_verify:
            score += 0.25 * verify_score
            parts.append(f"verify={verify_score:.2f}:{verify_reason}")
        if use_retrieval:
            retrieval_score = retrieval.get(tool.name, 0.0)
            score += 0.25 * retrieval_score
            parts.append(f"retrieval={retrieval_score:.2f}")
        score -= 0.001 * tool.slot
        scored.append((score, tool, ";".join(parts)))
    return sorted(scored, key=lambda item: item[0], reverse=True)


def _ir_level_score(action_name: str, tool: ToolSpec, row: dict[str, Any], level: str) -> tuple[float, str]:
    action = _action_type(action_name)
    profile = _tool_type(tool)
    retrieval_score = _tool_profile_score(action_name, tool, str(row.get("prompt", "")))
    score = 0.05 * retrieval_score
    reasons = []

    if level == "act_only":
        if action_name.startswith("ACT_"):
            score += 1.0
        reasons.append("act_only")
        return score, ";".join(reasons)

    op_overlap = action["operation"] & profile["operation"]
    if op_overlap:
        score += 20.0
        reasons.append("op")
    elif action["operation"]:
        score -= 15.0
        reasons.append("op_miss")
    if level == "operation":
        return score, ";".join(reasons)

    object_overlap = action["object"] & profile["object"]
    if object_overlap:
        score += 18.0 * len(object_overlap)
        reasons.append("object")
    elif action["object"]:
        score -= 12.0
        reasons.append("object_miss")
    if level == "operation_object":
        return score, ";".join(reasons)

    effect_overlap = action["effect"] & profile["effect"]
    if effect_overlap:
        score += 22.0 * len(effect_overlap)
        reasons.append("effect")
    elif action["effect"]:
        score -= 14.0
        reasons.append("effect_miss")
    constraint_overlap = action["constraints"] & profile["constraints"]
    score += 4.0 * len(constraint_overlap)
    if constraint_overlap:
        reasons.append("constraint")
    verify_score, verify_reason = _verification_score(action_name, tool, row)
    score += 0.25 * verify_score
    reasons.append(verify_reason)
    return score, ";".join(reasons)


def _rank_ir_level(action_name: str, row: dict[str, Any], swapped: SwappedRuntime, level: str) -> list[tuple[float, ToolSpec, str]]:
    if not action_name.startswith("ACT_") and action_name not in {"THINK", "TRANSFER"}:
        return []
    scored = []
    for tool in swapped.tool_specs:
        score, reason = _ir_level_score(action_name, tool, row, level)
        scored.append((score - 0.001 * tool.slot, tool, f"ir_{level}:{reason};score={score:.2f}"))
    return sorted(scored, key=lambda item: item[0], reverse=True)


def bind_typed_verify_ir(action_name: str, row: dict[str, Any], swapped: SwappedRuntime) -> tuple[ToolSpec | None, str]:
    ranked = _rank_typed_verify_ir(action_name, row, swapped)
    if not ranked:
        return None, "typed_verify_ir_no_tools"
    best_score, best_tool, reason = ranked[0]
    if best_score <= 0:
        return None, "typed_verify_ir_no_match"
    return best_tool, reason


def bind_behavior_verify_ir(action_name: str, row: dict[str, Any], swapped: SwappedRuntime) -> tuple[ToolSpec | None, str]:
    ranked = _rank_behavior_verify_ir(action_name, row, swapped)
    if not ranked:
        return None, "behavior_verify_ir_no_tools"
    best_score, best_tool, reason = ranked[0]
    if best_score <= 0:
        return None, "behavior_verify_ir_no_match"
    return best_tool, reason


def bind_ranked(
    ranked: list[tuple[float, ToolSpec, str]],
    missing_reason: str,
    threshold: float = 0.0,
) -> tuple[ToolSpec | None, str]:
    if not ranked:
        return None, missing_reason
    best_score, best_tool, reason = ranked[0]
    if best_score <= threshold:
        return None, f"{missing_reason}_no_match"
    return best_tool, reason


def bind_oracle_tool(row: dict[str, Any], swapped: SwappedRuntime) -> tuple[ToolSpec | None, str]:
    tool = tool_by_name(swapped.tool_specs, expected_alias(row, swapped))
    return tool, "oracle_tool" if tool else "oracle_tool_missing"


def bind_oracle_topk(
    action_name: str,
    row: dict[str, Any],
    swapped: SwappedRuntime,
    k: int,
) -> tuple[ToolSpec | None, str]:
    expected = expected_alias(row, swapped)
    ranked = ranked_candidates("retrieval_ir", action_name, row, swapped)
    topk = [tool.name for _, tool, _ in ranked[:k]]
    if expected and expected in topk:
        return tool_by_name(swapped.tool_specs, expected), f"oracle_top{k}_hit"
    if ranked:
        return ranked[0][1], f"oracle_top{k}_miss_fallback_retrieval"
    return None, f"oracle_top{k}_no_candidates"


def ranked_candidates(
    binder: str,
    action_name: str,
    row: dict[str, Any],
    swapped: SwappedRuntime,
) -> list[tuple[float, ToolSpec, str]]:
    if binder == "metadata_ir":
        return _rank_metadata_ir(action_name, row, swapped)
    if binder == "typed_rerank_ir":
        return _rank_typed_rerank_ir(action_name, row, swapped)
    if binder == "typed_verify_ir":
        return _rank_typed_verify_ir(action_name, row, swapped)
    if binder == "behavior_verify_ir":
        return _rank_behavior_verify_ir(action_name, row, swapped)
    if binder == "behavior_only_ir":
        return _rank_behavior_component_ir(
            action_name, row, swapped, use_behavior=True, use_verify=False, use_retrieval=False
        )
    if binder == "behavior_no_verify_ir":
        return _rank_behavior_component_ir(
            action_name, row, swapped, use_behavior=True, use_verify=False, use_retrieval=True
        )
    if binder == "behavior_no_retrieval_ir":
        return _rank_behavior_component_ir(
            action_name, row, swapped, use_behavior=True, use_verify=True, use_retrieval=False
        )
    if binder == "ir_act_only":
        return _rank_ir_level(action_name, row, swapped, "act_only")
    if binder == "ir_operation":
        return _rank_ir_level(action_name, row, swapped, "operation")
    if binder == "ir_operation_object":
        return _rank_ir_level(action_name, row, swapped, "operation_object")
    if binder == "ir_full":
        return _rank_ir_level(action_name, row, swapped, "full")
    return _rank_retrieval_ir(action_name, row, swapped)


def bind_concrete_name(row: dict[str, Any], original_predicted_tool: str | None, swapped: SwappedRuntime) -> tuple[ToolSpec | None, str]:
    tool = tool_by_name(swapped.tool_specs, original_predicted_tool)
    return tool, "concrete_tool_name" if tool else "concrete_tool_name_missing"


def bind_concrete_slot(original_predicted_slot: int | None, swapped: SwappedRuntime) -> tuple[ToolSpec | None, str]:
    tool = tool_by_slot(swapped.tool_specs, original_predicted_slot)
    return tool, "concrete_tool_slot" if tool else "concrete_tool_slot_missing"


def translate_args_to_original(tool: ToolSpec, args: dict[str, Any]) -> dict[str, Any]:
    aliases = dict(tool.raw.get("parameter_alias_to_original", {}))
    if not aliases:
        return args
    return {aliases.get(key, key): value for key, value in args.items()}


def execute_swapped_tool(swapped: SwappedRuntime, data: dict[str, Any], tool: ToolSpec, arguments: dict[str, Any]) -> str:
    original_name = swapped.alias_to_original.get(tool.name, tool.name)
    original_args = translate_args_to_original(tool, arguments)
    return execute_tool(swapped, data, original_name, original_args)


def expected_alias(row: dict[str, Any], swapped: SwappedRuntime) -> str | None:
    expected = row.get("expected_tool_name")
    if not expected:
        return None
    return swapped.original_to_alias.get(str(expected))


def main() -> None:
    args = parse_args()
    ensure_tau_import(args.tau_bench_path)
    rows = load_rows(args.decision_jsonl, args.max_samples)
    predictions_obj = load_prediction_file(args.predictions)
    prediction_rows = predictions_obj["records"][: len(rows)]
    if len(prediction_rows) != len(rows):
        raise ValueError("prediction records and decision rows must have the same length")

    historical_calls = historical_tool_call_map(args.trajectories)
    for row in rows:
        row["historical_tool_calls"] = historical_calls.get(str(row.get("sample_id", "")), [])

    original_domains = {domain: load_domain_runtime(domain) for domain in sorted({str(row.get("domain")) for row in rows})}
    swapped_domains = {
        domain: build_swapped_runtime(domain, args.variant, args.seed) for domain in sorted({str(row.get("domain")) for row in rows})
    }
    original_grounder = TauActionGrounder()
    decoder = JsonSchemaConstrainedDecoder()
    arg_generator = TauAliasOracleArgumentGenerator() if args.arg_generator == "oracle" else build_arg_generator(args)

    task_states: dict[tuple[str, int, int], dict[str, Any]] = {}
    expected_task_states: dict[tuple[str, int, int], dict[str, Any]] = {}
    task_step_counts: dict[tuple[str, int, int], int] = defaultdict(int)
    expected_step_counts: dict[tuple[str, int, int], int] = defaultdict(int)
    records: list[dict[str, Any]] = []

    for row, prediction in zip(rows, prediction_rows, strict=True):
        if str(row.get("sample_id")) != str(prediction.get("sample_id")):
            raise ValueError("prediction sample_id order does not match decision rows")
        domain = str(row.get("domain"))
        original_runtime = original_domains[domain]
        swapped = swapped_domains[domain]
        task_key = (domain, int(row.get("task_id", -1)), int(row.get("trial", 0)))
        if task_key not in task_states:
            task_states[task_key] = swapped.load_data()
            expected_task_states[task_key] = swapped.load_data()

        for historical_call in row.get("historical_tool_calls", []):
            historical_tool = historical_call.get("name")
            if historical_tool in original_runtime.tool_map and historical_tool not in swapped.terminate_tools:
                try:
                    execute_tool(swapped, expected_task_states[task_key], historical_tool, historical_call.get("arguments", {}))
                    expected_step_counts[task_key] += 1
                except Exception:
                    pass

        predicted_action = str(prediction.get("predicted_action", ""))
        original_grounding = original_grounder.ground(predicted_action, original_runtime.tool_specs, row)
        original_predicted_tool = None if original_grounding.tool is None else original_grounding.tool.name
        original_predicted_slot = None if original_grounding.tool is None else original_grounding.tool.slot

        if args.binder == "policy_ir":
            bound_tool, binding_reason = bind_policy_ir(
                predicted_action, row, swapped, original_grounder, original_runtime.tool_specs
            )
        elif args.binder == "metadata_ir":
            bound_tool, binding_reason = bind_metadata_ir(predicted_action, row, swapped)
        elif args.binder == "retrieval_ir":
            bound_tool, binding_reason = bind_retrieval_ir(predicted_action, row, swapped)
        elif args.binder == "typed_rerank_ir":
            bound_tool, binding_reason = bind_typed_rerank_ir(predicted_action, row, swapped)
        elif args.binder == "typed_verify_ir":
            bound_tool, binding_reason = bind_typed_verify_ir(predicted_action, row, swapped)
        elif args.binder == "behavior_verify_ir":
            bound_tool, binding_reason = bind_behavior_verify_ir(predicted_action, row, swapped)
        elif args.binder == "oracle_tool":
            bound_tool, binding_reason = bind_oracle_tool(row, swapped)
        elif args.binder == "oracle_top3":
            bound_tool, binding_reason = bind_oracle_topk(predicted_action, row, swapped, 3)
        elif args.binder == "oracle_top5":
            bound_tool, binding_reason = bind_oracle_topk(predicted_action, row, swapped, 5)
        elif args.binder in {
            "ir_act_only",
            "ir_operation",
            "ir_operation_object",
            "ir_full",
            "behavior_only_ir",
            "behavior_no_verify_ir",
            "behavior_no_retrieval_ir",
        }:
            bound_tool, binding_reason = bind_ranked(
                ranked_candidates(args.binder, predicted_action, row, swapped),
                args.binder,
            )
        elif args.binder == "description_only":
            bound_tool, binding_reason = bind_description_only(row, swapped)
        elif args.binder == "concrete_tool_name":
            bound_tool, binding_reason = bind_concrete_name(row, original_predicted_tool, swapped)
        else:
            bound_tool, binding_reason = bind_concrete_slot(original_predicted_slot, swapped)

        expected_tool = expected_alias(row, swapped)
        predicted_tool = None if bound_tool is None else bound_tool.name
        expected_original_tool = row.get("expected_tool_name")
        predicted_original_tool = None if bound_tool is None else swapped.alias_to_original.get(bound_tool.name, bound_tool.name)
        ranking = ranked_candidates(args.binder, predicted_action, row, swapped)
        ranked_original_tools = [swapped.alias_to_original.get(tool.name, tool.name) for _, tool, _ in ranking]
        if expected_original_tool and expected_original_tool in ranked_original_tools:
            expected_rank = ranked_original_tools.index(expected_original_tool) + 1
        else:
            expected_rank = None
        report = None
        execution_ok = False
        observation = None
        if bound_tool is not None:
            raw_args = arg_generator.generate(row, bound_tool)
            report = decoder.decode(raw_args, bound_tool.parameters)
            if report.valid:
                try:
                    observation = execute_swapped_tool(swapped, task_states[task_key], bound_tool, report.constrained)
                    execution_ok = not observation.startswith("Error:")
                    task_step_counts[task_key] += 1
                except Exception as exc:
                    observation = f"Error: {exc}"

        expected_action = str(row.get("correct_action", ""))
        records.append(
            {
                "sample_id": row.get("sample_id"),
                "domain": domain,
                "task_id": row.get("task_id"),
                "trial": row.get("trial", 0),
                "turn_index": row.get("turn_index"),
                "expected_action": expected_action,
                "predicted_action": predicted_action,
                "action_correct": predicted_action == expected_action,
                "expected_tool": expected_tool,
                "predicted_tool": predicted_tool,
                "expected_original_tool": expected_original_tool,
                "predicted_original_tool": predicted_original_tool,
                "original_predicted_tool": original_predicted_tool,
                "tool_grounding_correct": predicted_tool == expected_tool,
                "functional_tool_grounding_correct": predicted_original_tool == expected_original_tool,
                "functional_recall_at_1": expected_rank is not None and expected_rank <= 1,
                "functional_recall_at_3": expected_rank is not None and expected_rank <= 3,
                "functional_recall_at_5": expected_rank is not None and expected_rank <= 5,
                "functional_mrr": 0.0 if expected_rank is None else 1.0 / expected_rank,
                "candidate_original_tools_top5": ranked_original_tools[:5],
                "args_valid": True if report is None else report.valid,
                "execution_ok": execution_ok if predicted_tool is not None else expected_tool is None,
                "observation": observation,
                "arguments": None if report is None else report.constrained,
                "binding_reason": binding_reason,
            }
        )

    task_records = []
    for task_key, data in sorted(task_states.items()):
        domain, task_id, trial = task_key
        runtime = swapped_domains[domain]
        predicted_hash = data_hash(data)
        expected_hash = data_hash(expected_task_states[task_key])
        formal_gt_hash = task_gt_hash(runtime, task_id)
        task_records.append(
            {
                "domain": domain,
                "task_id": task_id,
                "trial": trial,
                "num_executed_steps": task_step_counts[task_key],
                "num_expected_steps": expected_step_counts[task_key],
                "predicted_hash": predicted_hash,
                "expected_hash": expected_hash,
                "formal_gt_hash": formal_gt_hash,
                "db_hash_match": predicted_hash == expected_hash,
                "formal_gt_hash_match": formal_gt_hash is not None and predicted_hash == formal_gt_hash,
            }
        )

    tool_records = [row for row in records if row["expected_tool"] is not None]
    predicted_tool_records = [row for row in records if row["predicted_tool"] is not None]
    mrr = sum(float(row["functional_mrr"]) for row in tool_records) / max(1, len(tool_records))
    summary = {
        "method": "tau_environment_swap",
        "prediction_file": args.predictions,
        "variant": args.variant,
        "binder": args.binder,
        "num_samples": len(records),
        "num_tasks": len(task_records),
        "action_accuracy": avg(records, "action_correct"),
        "tool_grounding_accuracy_all": avg(records, "tool_grounding_correct"),
        "tool_grounding_accuracy_tool_only": avg(tool_records, "tool_grounding_correct"),
        "functional_tool_grounding_accuracy_all": avg(records, "functional_tool_grounding_correct"),
        "functional_tool_grounding_accuracy_tool_only": avg(tool_records, "functional_tool_grounding_correct"),
        "functional_recall_at_1_tool_only": avg(tool_records, "functional_recall_at_1"),
        "functional_recall_at_3_tool_only": avg(tool_records, "functional_recall_at_3"),
        "functional_recall_at_5_tool_only": avg(tool_records, "functional_recall_at_5"),
        "functional_mrr_tool_only": mrr,
        "args_valid_rate_predicted_tool": avg(predicted_tool_records, "args_valid"),
        "tool_execution_ok_rate_predicted_tool": avg(predicted_tool_records, "execution_ok"),
        "db_hash_match_rate": avg(task_records, "db_hash_match"),
        "formal_gt_hash_match_rate": avg(task_records, "formal_gt_hash_match"),
        "environment_binding_trainable_parameters": 0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "tasks": task_records, "records": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
