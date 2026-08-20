from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from universal_agent_policy.runtime.grounding import ToolSpec, normalize_tool


ANSWER = "ANSWER"
ACT_COMPUTE = "ACT_COMPUTE"
ACT_RETRIEVE = "ACT_RETRIEVE"
ACT_LOOKUP = "ACT_LOOKUP"
ACT_TRANSFORM = "ACT_TRANSFORM"

SEMANTIC_ACTION_NAMES = [ANSWER, ACT_COMPUTE, ACT_RETRIEVE, ACT_LOOKUP, ACT_TRANSFORM]

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "from",
    "given",
    "in",
    "is",
    "me",
    "of",
    "on",
    "or",
    "the",
    "to",
    "user",
    "with",
}

_COMPUTE_KEYWORDS = {
    "area",
    "average",
    "calculate",
    "calculator",
    "circumference",
    "compute",
    "equation",
    "hypotenuse",
    "math",
    "quadratic",
    "root",
    "solve",
    "sum",
    "triangle",
}

_RETRIEVE_KEYWORDS = {
    "current",
    "forecast",
    "latest",
    "live",
    "news",
    "price",
    "retrieve",
    "search",
    "stock",
    "temperature",
    "weather",
    "web",
}

_LOOKUP_KEYWORDS = {
    "account",
    "company",
    "customer",
    "database",
    "employee",
    "find",
    "get",
    "id",
    "lookup",
    "order",
    "profile",
    "query",
    "record",
    "sql",
    "user",
}

_TRANSFORM_KEYWORDS = {
    "convert",
    "extract",
    "format",
    "parse",
    "summarize",
    "translate",
    "transform",
}


@dataclass(frozen=True)
class SemanticGrounding:
    action_name: str
    tool: ToolSpec | None
    reason: str

    @property
    def calls_tool(self) -> bool:
        return self.tool is not None


def tokens(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9_]+", text.lower()) if tok not in _STOPWORDS}


def row_user_text(row: dict[str, Any]) -> str:
    question = row.get("original_question", row.get("question", row.get("messages", "")))
    if isinstance(question, list) and question and isinstance(question[0], list):
        question = question[0]
    if isinstance(question, list):
        return "\n".join(str(msg.get("content", "")) for msg in question if isinstance(msg, dict))
    return str(question)


def tool_text(tool: dict[str, Any]) -> str:
    params = tool.get("parameters", {}).get("properties", {})
    param_text = " ".join(str(key) for key in params)
    return " ".join(
        [
            str(tool.get("name", "")),
            str(tool.get("description", "")),
            param_text,
        ]
    )


def infer_tool_action(tool: dict[str, Any]) -> str:
    text_tokens = tokens(tool_text(tool))
    scores = {
        ACT_COMPUTE: len(text_tokens & _COMPUTE_KEYWORDS),
        ACT_RETRIEVE: len(text_tokens & _RETRIEVE_KEYWORDS),
        ACT_LOOKUP: len(text_tokens & _LOOKUP_KEYWORDS),
        ACT_TRANSFORM: len(text_tokens & _TRANSFORM_KEYWORDS),
    }
    best_action, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score > 0:
        return best_action
    return ACT_LOOKUP


def lexical_tool_score(row: dict[str, Any], tool: dict[str, Any]) -> float:
    query_tokens = tokens(row_user_text(row))
    candidate_tokens = tokens(tool_text(tool))
    overlap = len(query_tokens & candidate_tokens)
    name_tokens = tokens(str(tool.get("name", "")).replace("_", " "))
    name_overlap = len(query_tokens & name_tokens)
    return float(overlap + 2 * name_overlap)


class SemanticActionGrounder:
    def ground(self, action_name: str, row: dict[str, Any]) -> SemanticGrounding:
        if not action_name.startswith("ACT_"):
            return SemanticGrounding(action_name, None, "policy_selected_non_tool_action")

        tools = list(row.get("tools", []))
        if not tools:
            return SemanticGrounding(action_name, None, "no_tools_available")

        scored: list[tuple[float, int, dict[str, Any]]] = []
        for slot, tool in enumerate(tools):
            inferred_action = infer_tool_action(tool)
            intent_score = 10.0 if inferred_action == action_name else 0.0
            scored.append((intent_score + lexical_tool_score(row, tool), slot, tool))

        best_score, slot, tool = max(scored, key=lambda item: (item[0], -item[1]))
        if best_score <= 0:
            return SemanticGrounding(action_name, None, "no_semantic_tool_match")
        return SemanticGrounding(action_name, normalize_tool(tool, slot), f"semantic_score={best_score:.2f}")
