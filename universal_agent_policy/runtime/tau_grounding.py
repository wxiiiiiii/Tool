from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from universal_agent_policy.runtime.grounding import ToolSpec, normalize_tool


TAU_ANSWER = "ANSWER"
TAU_ASK_USER = "ASK_USER"
TAU_VERIFY = "VERIFY"
TAU_ACT_RETRIEVE = "ACT_RETRIEVE"
TAU_ACT_SEARCH = "ACT_SEARCH"
TAU_ACT_COMPUTE = "ACT_COMPUTE"
TAU_ACT_UPDATE = "ACT_UPDATE"
TAU_ACT_RETRIEVE_USER = "ACT_RETRIEVE_USER"
TAU_ACT_RETRIEVE_ORDER = "ACT_RETRIEVE_ORDER"
TAU_ACT_RETRIEVE_PRODUCT = "ACT_RETRIEVE_PRODUCT"
TAU_ACT_RETRIEVE_CATALOG = "ACT_RETRIEVE_CATALOG"
TAU_ACT_RETRIEVE_RESERVATION = "ACT_RETRIEVE_RESERVATION"
TAU_ACT_RETRIEVE_AIRPORT = "ACT_RETRIEVE_AIRPORT"
TAU_ACT_SEARCH_FLIGHT = "ACT_SEARCH_FLIGHT"
TAU_ACT_UPDATE_ORDER_CANCEL = "ACT_UPDATE_ORDER_CANCEL"
TAU_ACT_UPDATE_ORDER_ITEMS = "ACT_UPDATE_ORDER_ITEMS"
TAU_ACT_UPDATE_ORDER_PAYMENT = "ACT_UPDATE_ORDER_PAYMENT"
TAU_ACT_UPDATE_ORDER_ADDRESS = "ACT_UPDATE_ORDER_ADDRESS"
TAU_ACT_UPDATE_USER_ADDRESS = "ACT_UPDATE_USER_ADDRESS"
TAU_ACT_UPDATE_RESERVATION_BOOK = "ACT_UPDATE_RESERVATION_BOOK"
TAU_ACT_UPDATE_RESERVATION_CANCEL = "ACT_UPDATE_RESERVATION_CANCEL"
TAU_ACT_UPDATE_RESERVATION_FLIGHT = "ACT_UPDATE_RESERVATION_FLIGHT"
TAU_ACT_UPDATE_RESERVATION_BAGGAGE = "ACT_UPDATE_RESERVATION_BAGGAGE"
TAU_ACT_UPDATE_RESERVATION_PASSENGER = "ACT_UPDATE_RESERVATION_PASSENGER"
TAU_ACT_SEND_CERTIFICATE = "ACT_SEND_CERTIFICATE"
TAU_THINK = "THINK"
TAU_TRANSFER = "TRANSFER"
TAU_STOP = "STOP"

TAU_TOOL_ACTIONS = {
    TAU_ACT_RETRIEVE,
    TAU_ACT_SEARCH,
    TAU_ACT_COMPUTE,
    TAU_ACT_UPDATE,
    TAU_THINK,
    TAU_TRANSFER,
    TAU_ACT_RETRIEVE_USER,
    TAU_ACT_RETRIEVE_ORDER,
    TAU_ACT_RETRIEVE_PRODUCT,
    TAU_ACT_RETRIEVE_CATALOG,
    TAU_ACT_RETRIEVE_RESERVATION,
    TAU_ACT_RETRIEVE_AIRPORT,
    TAU_ACT_SEARCH_FLIGHT,
    TAU_ACT_UPDATE_ORDER_CANCEL,
    TAU_ACT_UPDATE_ORDER_ITEMS,
    TAU_ACT_UPDATE_ORDER_PAYMENT,
    TAU_ACT_UPDATE_ORDER_ADDRESS,
    TAU_ACT_UPDATE_USER_ADDRESS,
    TAU_ACT_UPDATE_RESERVATION_BOOK,
    TAU_ACT_UPDATE_RESERVATION_CANCEL,
    TAU_ACT_UPDATE_RESERVATION_FLIGHT,
    TAU_ACT_UPDATE_RESERVATION_BAGGAGE,
    TAU_ACT_UPDATE_RESERVATION_PASSENGER,
    TAU_ACT_SEND_CERTIFICATE,
}

RETRIEVE_PREFIXES = ("find_", "get_", "list_")
SEARCH_PREFIXES = ("search_",)
UPDATE_PREFIXES = (
    "book_",
    "cancel_",
    "exchange_",
    "modify_",
    "return_",
    "send_",
    "update_",
)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "from",
    "id",
    "in",
    "is",
    "of",
    "or",
    "the",
    "to",
    "user",
    "with",
}


@dataclass(frozen=True)
class TauGrounding:
    action_name: str
    tool: ToolSpec | None
    reason: str

    @property
    def calls_tool(self) -> bool:
        return self.tool is not None


def flatten_tau_tool(tool_info: dict[str, Any], slot: int) -> dict[str, Any]:
    function = dict(tool_info.get("function", tool_info))
    return {
        "name": str(function.get("name", f"tool_{slot}")),
        "description": str(function.get("description", "")),
        "parameters": dict(function.get("parameters", {})),
        "raw": tool_info,
    }


def tau_tools_to_specs(tools_info: list[dict[str, Any]]) -> list[ToolSpec]:
    return [normalize_tool(flatten_tau_tool(tool, slot), slot) for slot, tool in enumerate(tools_info)]


def tau_tool_action(tool_name: str) -> str:
    if tool_name == "think":
        return TAU_THINK
    if tool_name == "calculate":
        return TAU_ACT_COMPUTE
    if tool_name == "transfer_to_human_agents":
        return TAU_TRANSFER
    if tool_name.startswith(SEARCH_PREFIXES):
        return TAU_ACT_SEARCH
    if tool_name.startswith(UPDATE_PREFIXES):
        return TAU_ACT_UPDATE
    if tool_name.startswith(RETRIEVE_PREFIXES):
        return TAU_ACT_RETRIEVE
    return TAU_ACT_RETRIEVE


def tau_tool_actions(tool_name: str) -> set[str]:
    actions = {tau_tool_action(tool_name)}
    if tool_name in {"find_user_id_by_email", "find_user_id_by_name_zip", "get_user_details"}:
        actions.add(TAU_ACT_RETRIEVE_USER)
    elif tool_name == "get_order_details":
        actions.add(TAU_ACT_RETRIEVE_ORDER)
    elif tool_name == "get_product_details":
        actions.add(TAU_ACT_RETRIEVE_PRODUCT)
    elif tool_name == "list_all_product_types":
        actions.add(TAU_ACT_RETRIEVE_CATALOG)
    elif tool_name == "get_reservation_details":
        actions.add(TAU_ACT_RETRIEVE_RESERVATION)
    elif tool_name == "list_all_airports":
        actions.add(TAU_ACT_RETRIEVE_AIRPORT)
    elif tool_name in {"search_direct_flight", "search_onestop_flight"}:
        actions.add(TAU_ACT_SEARCH_FLIGHT)
    elif tool_name == "cancel_pending_order":
        actions.add(TAU_ACT_UPDATE_ORDER_CANCEL)
    elif tool_name in {"exchange_delivered_order_items", "modify_pending_order_items", "return_delivered_order_items"}:
        actions.add(TAU_ACT_UPDATE_ORDER_ITEMS)
    elif tool_name == "modify_pending_order_payment":
        actions.add(TAU_ACT_UPDATE_ORDER_PAYMENT)
    elif tool_name == "modify_pending_order_address":
        actions.add(TAU_ACT_UPDATE_ORDER_ADDRESS)
    elif tool_name == "modify_user_address":
        actions.add(TAU_ACT_UPDATE_USER_ADDRESS)
    elif tool_name == "book_reservation":
        actions.add(TAU_ACT_UPDATE_RESERVATION_BOOK)
    elif tool_name == "cancel_reservation":
        actions.add(TAU_ACT_UPDATE_RESERVATION_CANCEL)
    elif tool_name == "update_reservation_flights":
        actions.add(TAU_ACT_UPDATE_RESERVATION_FLIGHT)
    elif tool_name == "update_reservation_baggages":
        actions.add(TAU_ACT_UPDATE_RESERVATION_BAGGAGE)
    elif tool_name == "update_reservation_passengers":
        actions.add(TAU_ACT_UPDATE_RESERVATION_PASSENGER)
    elif tool_name == "send_certificate":
        actions.add(TAU_ACT_SEND_CERTIFICATE)
    return actions


def _tokens(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9_]+", text.lower()) if tok not in _STOPWORDS}


def _conversation_state(text: str) -> str:
    marker = "Conversation state before the next assistant decision:"
    if marker in text:
        text = text.split(marker, 1)[1]
    end_marker = "Next policy action:"
    if end_marker in text:
        text = text.split(end_marker, 1)[0]
    return text.strip()


def _called_tools(text: str) -> list[str]:
    return re.findall(r"assistant tool_call:\s*([a-zA-Z0-9_]+)\(", text)


def _called_tool_args(text: str, tool_name: str) -> list[str]:
    pattern = rf"assistant tool_call:\s*{re.escape(tool_name)}\((.*?)\)"
    return re.findall(pattern, text)


def _order_ids_from_user_details(text: str) -> list[str]:
    match = re.search(r'"orders"\s*:\s*\[(.*?)\]', text, flags=re.DOTALL)
    if not match:
        return []
    return re.findall(r"#W\d{7}", match.group(1))


def _called_order_ids(text: str) -> set[str]:
    ids: set[str] = set()
    for args in _called_tool_args(text, "get_order_details"):
        ids.update(re.findall(r"#W\d{7}", args))
    return ids


def _last_user_text(text: str) -> str:
    matches = re.findall(r"(?:^|\n)user:\s*(.*)", text)
    return matches[-1] if matches else ""


def _has_tool(tools: list[ToolSpec], name: str) -> ToolSpec | None:
    for tool in tools:
        if tool.name == name:
            return tool
    return None


def _first_matching_tool(tools: list[ToolSpec], names: set[str]) -> ToolSpec | None:
    for tool in tools:
        if tool.name in names:
            return tool
    return None


def _domain_from_tools(tools: list[ToolSpec]) -> str:
    names = {tool.name for tool in tools}
    if "get_order_details" in names:
        return "retail"
    if "get_reservation_details" in names:
        return "airline"
    return "unknown"


def _tool_score(query: str, tool: ToolSpec) -> float:
    query_tokens = _tokens(query)
    tool_tokens = _tokens(" ".join([tool.name.replace("_", " "), tool.description]))
    parameter_tokens = _tokens(" ".join(tool.parameters.get("properties", {}).keys()))
    name_overlap = len(query_tokens & _tokens(tool.name.replace("_", " ")))
    return float(len(query_tokens & tool_tokens) + len(query_tokens & parameter_tokens) + 2 * name_overlap)


def row_context_text(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if prompt:
        return str(prompt)
    messages = row.get("messages", [])
    if isinstance(messages, list):
        return "\n".join(str(msg.get("content", "")) for msg in messages if isinstance(msg, dict))
    return str(messages)


class TauActionGrounder:
    def ground(self, action_name: str, tools: list[ToolSpec], row: dict[str, Any]) -> TauGrounding:
        if action_name not in TAU_TOOL_ACTIONS:
            return TauGrounding(action_name, None, "policy_selected_non_tool_action")

        context = _conversation_state(row_context_text(row))
        direct_tool = self._direct_hierarchical_ground(action_name, tools, context)
        if direct_tool is not None:
            return TauGrounding(action_name, direct_tool, "hierarchical_direct")

        matching = [tool for tool in tools if action_name in tau_tool_actions(tool.name)]
        if not matching:
            return TauGrounding(action_name, None, "no_tool_matches_action_type")

        if len(matching) == 1:
            return TauGrounding(action_name, matching[0], "single_tool_for_action_type")

        rule_tool = self._rule_ground(action_name, tools, context)
        if rule_tool is not None:
            return TauGrounding(action_name, rule_tool, "domain_rule")

        scored = [(_tool_score(context, tool), -tool.slot, tool) for tool in matching]
        best_score, _, best_tool = max(scored, key=lambda item: item[:2])
        return TauGrounding(action_name, best_tool, f"semantic_score={best_score:.2f}")

    def _direct_hierarchical_ground(self, action_name: str, tools: list[ToolSpec], context: str) -> ToolSpec | None:
        lowered = context.lower()
        last_user = _last_user_text(context).lower()
        called = _called_tools(context)
        if action_name == TAU_ACT_RETRIEVE_USER:
            if "@" in last_user:
                return _has_tool(tools, "find_user_id_by_email")
            if not any(name.startswith("find_user_id_by") for name in called) and re.search(r"\b\d{5}\b", last_user):
                return _has_tool(tools, "find_user_id_by_name_zip")
            if any(name.startswith("find_user_id_by") for name in called) or re.search(r"\b[a-z]+_[a-z]+_\d{3,}\b", context):
                return _has_tool(tools, "get_user_details")
            return _first_matching_tool(tools, {"find_user_id_by_email", "find_user_id_by_name_zip", "get_user_details"})
        if action_name == TAU_ACT_RETRIEVE_ORDER:
            return _has_tool(tools, "get_order_details")
        if action_name == TAU_ACT_RETRIEVE_PRODUCT:
            return _has_tool(tools, "get_product_details")
        if action_name == TAU_ACT_RETRIEVE_CATALOG:
            return _has_tool(tools, "list_all_product_types")
        if action_name == TAU_ACT_RETRIEVE_RESERVATION:
            return _has_tool(tools, "get_reservation_details")
        if action_name == TAU_ACT_RETRIEVE_AIRPORT:
            return _has_tool(tools, "list_all_airports")
        if action_name == TAU_ACT_SEARCH_FLIGHT:
            if any(word in lowered for word in ["direct", "nonstop", "non-stop"]):
                return _has_tool(tools, "search_direct_flight")
            return _has_tool(tools, "search_onestop_flight") or _has_tool(tools, "search_direct_flight")
        if action_name == TAU_ACT_UPDATE_ORDER_CANCEL:
            return _has_tool(tools, "cancel_pending_order")
        if action_name == TAU_ACT_UPDATE_ORDER_PAYMENT:
            return _has_tool(tools, "modify_pending_order_payment")
        if action_name == TAU_ACT_UPDATE_ORDER_ADDRESS:
            return _has_tool(tools, "modify_pending_order_address")
        if action_name == TAU_ACT_UPDATE_USER_ADDRESS:
            return _has_tool(tools, "modify_user_address")
        if action_name == TAU_ACT_UPDATE_ORDER_ITEMS:
            if "exchange" in lowered:
                return _has_tool(tools, "exchange_delivered_order_items")
            if "return" in lowered:
                return _has_tool(tools, "return_delivered_order_items")
            return _has_tool(tools, "modify_pending_order_items") or _has_tool(tools, "exchange_delivered_order_items")
        if action_name == TAU_ACT_UPDATE_RESERVATION_BOOK:
            return _has_tool(tools, "book_reservation")
        if action_name == TAU_ACT_UPDATE_RESERVATION_CANCEL:
            return _has_tool(tools, "cancel_reservation")
        if action_name == TAU_ACT_UPDATE_RESERVATION_FLIGHT:
            return _has_tool(tools, "update_reservation_flights")
        if action_name == TAU_ACT_UPDATE_RESERVATION_BAGGAGE:
            return _has_tool(tools, "update_reservation_baggages")
        if action_name == TAU_ACT_UPDATE_RESERVATION_PASSENGER:
            return _has_tool(tools, "update_reservation_passengers")
        if action_name == TAU_ACT_SEND_CERTIFICATE:
            return _has_tool(tools, "send_certificate")
        return None

    def _rule_ground(self, action_name: str, tools: list[ToolSpec], context: str) -> ToolSpec | None:
        domain = _domain_from_tools(tools)
        lowered = context.lower()
        last_user = _last_user_text(context).lower()
        called = _called_tools(context)

        if action_name == TAU_ACT_SEARCH:
            if any(word in lowered for word in ["direct", "nonstop", "non-stop"]):
                return _has_tool(tools, "search_direct_flight")
            return _has_tool(tools, "search_onestop_flight") or _has_tool(tools, "search_direct_flight")

        if action_name == TAU_ACT_RETRIEVE:
            if domain == "retail":
                return self._rule_retail_retrieve(tools, context, lowered, last_user, called)
            if domain == "airline":
                return self._rule_airline_retrieve(tools, context, lowered, last_user, called)

        if action_name == TAU_ACT_UPDATE:
            if domain == "retail":
                return self._rule_retail_update(tools, lowered, last_user)
            if domain == "airline":
                return self._rule_airline_update(tools, lowered, last_user)

        return None

    def _rule_retail_retrieve(
        self,
        tools: list[ToolSpec],
        context: str,
        lowered: str,
        last_user: str,
        called: list[str],
    ) -> ToolSpec | None:
        if "@" in last_user:
            if not any(name.startswith("find_user_id_by") for name in called):
                return _has_tool(tools, "find_user_id_by_email")
        if (
            not any(name.startswith("find_user_id_by") for name in called)
            and re.search(r"\bzip(?: code)?\b", last_user)
            and re.search(r"\b\d{5}\b", last_user)
        ):
            return _has_tool(tools, "find_user_id_by_name_zip")
        if (
            "list_all_product_types" not in called
            and any(phrase in lowered for phrase in ["product types", "how many", "options are available", "online store"])
        ):
            return _has_tool(tools, "list_all_product_types")
        if "list_all_product_types" in called and "get_product_details" not in called:
            return _has_tool(tools, "get_product_details")
        if "find_user_id_by" in " ".join(called) and "get_user_details" not in called:
            return _has_tool(tools, "get_user_details")
        order_ids = _order_ids_from_user_details(context)
        if order_ids:
            called_order_ids = _called_order_ids(context)
            if any(order_id not in called_order_ids for order_id in order_ids):
                return _has_tool(tools, "get_order_details")
        if (
            re.search(r'"product_id"\s*:', context)
            or "product details" in lowered
            or any(word in lowered for word in ["compatible", "switch", "backlight", "option", "available"])
        ):
            return _has_tool(tools, "get_product_details")
        if '"orders": [' in context or re.search(r"#W\d{7}", context) or "order id" in lowered:
            return _has_tool(tools, "get_order_details")
        if re.search(r"\b[a-z]+_[a-z]+_\d{3,}\b", context):
            return _has_tool(tools, "get_user_details")
        return None

    def _rule_airline_retrieve(
        self,
        tools: list[ToolSpec],
        context: str,
        lowered: str,
        last_user: str,
        called: list[str],
    ) -> ToolSpec | None:
        if "airport" in lowered or re.search(r"\b[A-Z]{3}\b", context):
            if "list_all_airports" not in called and "airport" in lowered:
                return _has_tool(tools, "list_all_airports")
        if re.search(r"\b[A-Z0-9]{6}\b", context) or "reservation" in lowered:
            if "get_user_details" in called or "reservation_id" in lowered:
                return _has_tool(tools, "get_reservation_details")
        if re.search(r"\b[a-z]+_[a-z]+_\d{3,}\b", context) or "user id" in lowered:
            return _has_tool(tools, "get_user_details")
        return None

    def _rule_retail_update(self, tools: list[ToolSpec], lowered: str, last_user: str) -> ToolSpec | None:
        intent = last_user or lowered[-2000:]
        if "exchange" in intent:
            return _has_tool(tools, "exchange_delivered_order_items")
        if "return" in intent:
            return _has_tool(tools, "return_delivered_order_items")
        if "cancel" in intent:
            return _has_tool(tools, "cancel_pending_order")
        if "address" in intent:
            return _has_tool(tools, "modify_pending_order_address") or _has_tool(tools, "modify_user_address")
        if "payment" in intent or "card" in intent:
            return _has_tool(tools, "modify_pending_order_payment")
        if "item" in intent or "quantity" in intent or "change" in intent or "remove" in intent or "add" in intent:
            return _has_tool(tools, "modify_pending_order_items")
        if "item_id" in lowered:
            return _has_tool(tools, "modify_pending_order_items")
        return None

    def _rule_airline_update(self, tools: list[ToolSpec], lowered: str, last_user: str) -> ToolSpec | None:
        intent = last_user or lowered[-2000:]
        if "book" in intent:
            return _has_tool(tools, "book_reservation")
        if "cancel" in intent:
            return _has_tool(tools, "cancel_reservation")
        if "baggage" in intent or "bag" in intent:
            return _has_tool(tools, "update_reservation_baggages")
        if "passenger" in intent or "birth" in intent:
            return _has_tool(tools, "update_reservation_passengers")
        if "certificate" in intent:
            return _has_tool(tools, "send_certificate")
        if "flight" in intent or "cabin" in intent:
            return _has_tool(tools, "update_reservation_flights")
        return None
