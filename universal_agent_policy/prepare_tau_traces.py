from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TAU_ACTION_NAMES = [
    "ANSWER",
    "ASK_USER",
    "VERIFY",
    "ACT_RETRIEVE",
    "ACT_SEARCH",
    "ACT_COMPUTE",
    "ACT_UPDATE",
    "THINK",
    "TRANSFER",
    "STOP",
]

TAU_HIERARCHICAL_ACTION_NAMES = [
    "ANSWER",
    "ASK_USER",
    "VERIFY",
    "THINK",
    "TRANSFER",
    "STOP",
    "ACT_RETRIEVE_USER",
    "ACT_RETRIEVE_ORDER",
    "ACT_RETRIEVE_PRODUCT",
    "ACT_RETRIEVE_CATALOG",
    "ACT_RETRIEVE_RESERVATION",
    "ACT_RETRIEVE_AIRPORT",
    "ACT_SEARCH_FLIGHT",
    "ACT_COMPUTE",
    "ACT_UPDATE_ORDER_CANCEL",
    "ACT_UPDATE_ORDER_ITEMS",
    "ACT_UPDATE_ORDER_PAYMENT",
    "ACT_UPDATE_ORDER_ADDRESS",
    "ACT_UPDATE_USER_ADDRESS",
    "ACT_UPDATE_RESERVATION_BOOK",
    "ACT_UPDATE_RESERVATION_CANCEL",
    "ACT_UPDATE_RESERVATION_FLIGHT",
    "ACT_UPDATE_RESERVATION_BAGGAGE",
    "ACT_UPDATE_RESERVATION_PASSENGER",
    "ACT_SEND_CERTIFICATE",
]

RETRIEVE_TOOLS = {
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_order_details",
    "get_product_details",
    "get_reservation_details",
    "get_user_details",
    "list_all_airports",
    "list_all_product_types",
}

SEARCH_TOOLS = {
    "search_direct_flight",
    "search_onestop_flight",
}

UPDATE_PREFIXES = (
    "book_",
    "cancel_",
    "exchange_",
    "modify_",
    "return_",
    "send_",
    "update_",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--action-vocab", choices=["coarse", "hierarchical"], default="coarse")
    parser.add_argument("--max-per-action", type=int, default=-1)
    parser.add_argument("--max-context-messages", type=int, default=14)
    parser.add_argument("--max-message-chars", type=int, default=900)
    parser.add_argument("--max-system-chars", type=int, default=2500)
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def domain_from_path(path: str | Path) -> str:
    name = Path(path).name.lower()
    if "airline" in name:
        return "airline"
    if "retail" in name:
        return "retail"
    return "unknown"


def shorten(text: Any, limit: int) -> str:
    value = str(text or "")
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= limit:
        return value
    return value[:limit] + " ...[truncated]"


def tool_name_from_call(call: dict[str, Any]) -> str:
    function = call.get("function", {})
    return str(function.get("name") or call.get("name") or "")


def tool_action(tool_name: str) -> str:
    if tool_name == "think":
        return "THINK"
    if tool_name == "calculate":
        return "ACT_COMPUTE"
    if tool_name == "transfer_to_human_agents":
        return "TRANSFER"
    if tool_name in SEARCH_TOOLS:
        return "ACT_SEARCH"
    if tool_name in RETRIEVE_TOOLS or tool_name.startswith(("find_", "get_", "list_")):
        return "ACT_RETRIEVE"
    if tool_name.startswith(UPDATE_PREFIXES):
        return "ACT_UPDATE"
    return "ACT_RETRIEVE"


def hierarchical_tool_action(tool_name: str) -> str:
    if tool_name == "think":
        return "THINK"
    if tool_name == "calculate":
        return "ACT_COMPUTE"
    if tool_name == "transfer_to_human_agents":
        return "TRANSFER"
    if tool_name in {"find_user_id_by_email", "find_user_id_by_name_zip", "get_user_details"}:
        return "ACT_RETRIEVE_USER"
    if tool_name == "get_order_details":
        return "ACT_RETRIEVE_ORDER"
    if tool_name == "get_product_details":
        return "ACT_RETRIEVE_PRODUCT"
    if tool_name == "list_all_product_types":
        return "ACT_RETRIEVE_CATALOG"
    if tool_name == "get_reservation_details":
        return "ACT_RETRIEVE_RESERVATION"
    if tool_name == "list_all_airports":
        return "ACT_RETRIEVE_AIRPORT"
    if tool_name in SEARCH_TOOLS:
        return "ACT_SEARCH_FLIGHT"
    if tool_name == "cancel_pending_order":
        return "ACT_UPDATE_ORDER_CANCEL"
    if tool_name in {"exchange_delivered_order_items", "modify_pending_order_items", "return_delivered_order_items"}:
        return "ACT_UPDATE_ORDER_ITEMS"
    if tool_name == "modify_pending_order_payment":
        return "ACT_UPDATE_ORDER_PAYMENT"
    if tool_name == "modify_pending_order_address":
        return "ACT_UPDATE_ORDER_ADDRESS"
    if tool_name == "modify_user_address":
        return "ACT_UPDATE_USER_ADDRESS"
    if tool_name == "book_reservation":
        return "ACT_UPDATE_RESERVATION_BOOK"
    if tool_name == "cancel_reservation":
        return "ACT_UPDATE_RESERVATION_CANCEL"
    if tool_name == "update_reservation_flights":
        return "ACT_UPDATE_RESERVATION_FLIGHT"
    if tool_name == "update_reservation_baggages":
        return "ACT_UPDATE_RESERVATION_BAGGAGE"
    if tool_name == "update_reservation_passengers":
        return "ACT_UPDATE_RESERVATION_PASSENGER"
    if tool_name == "send_certificate":
        return "ACT_SEND_CERTIFICATE"
    if tool_name.startswith(UPDATE_PREFIXES):
        return "ACT_UPDATE_ORDER_ITEMS"
    return "ACT_RETRIEVE_USER"


def text_action(content: str) -> str:
    text = content.lower()
    if "transfer" in text and "human" in text:
        return "TRANSFER"
    if any(phrase in text for phrase in ["please confirm", "confirm that", "confirm the", "proceed with"]):
        return "VERIFY"
    asks_user = (
        "?" in text
        or "could you" in text
        or "please provide" in text
        or "let me know" in text
        or "would you" in text
        or "which " in text
        or "do you" in text
    )
    if asks_user:
        return "ASK_USER"
    return "ANSWER"


def assistant_action(message: dict[str, Any], action_vocab: str) -> tuple[str, str | None]:
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        tool_name = tool_name_from_call(tool_calls[0])
        if action_vocab == "hierarchical":
            return hierarchical_tool_action(tool_name), tool_name
        return tool_action(tool_name), tool_name
    return text_action(str(message.get("content") or "")), None


def format_message(message: dict[str, Any], max_chars: int) -> str:
    role = str(message.get("role", ""))
    if role == "assistant" and message.get("tool_calls"):
        calls = []
        for call in message.get("tool_calls", []):
            function = call.get("function", {})
            calls.append(f"{function.get('name')}({function.get('arguments', '')})")
        return "assistant tool_call: " + shorten("; ".join(calls), max_chars)
    if role == "tool":
        return f"tool {message.get('name', '')}: " + shorten(message.get("content", ""), max_chars)
    return f"{role}: " + shorten(message.get("content", ""), max_chars)


def build_prompt(
    domain: str,
    trajectory: list[dict[str, Any]],
    idx: int,
    max_context_messages: int,
    max_message_chars: int,
    max_system_chars: int,
    action_names: list[str],
) -> str:
    system = ""
    if trajectory and trajectory[0].get("role") == "system":
        system = shorten(trajectory[0].get("content", ""), max_system_chars)
    start = max(1, idx - max_context_messages)
    context = "\n".join(format_message(message, max_message_chars) for message in trajectory[start:idx])
    return (
        "Predict the next universal agent policy action for a tau-bench customer-service agent.\n"
        f"Domain: {domain}\n"
        f"Action vocabulary: {', '.join(action_names)}\n\n"
        f"Policy summary:\n{system}\n\n"
        f"Conversation state before the next assistant decision:\n{context}\n\n"
        "Next policy action:"
    )


def convert_trajectory(
    item: dict[str, Any],
    domain: str,
    max_context_messages: int,
    max_message_chars: int,
    max_system_chars: int,
    action_vocab: str,
) -> list[dict[str, Any]]:
    rows = []
    trajectory = item.get("traj", [])
    action_names = TAU_HIERARCHICAL_ACTION_NAMES if action_vocab == "hierarchical" else TAU_ACTION_NAMES
    for idx, message in enumerate(trajectory):
        if message.get("role") != "assistant":
            continue
        action_name, tool_name = assistant_action(message, action_vocab)
        action_id = action_names.index(action_name)
        rows.append(
            {
                "sample_id": f"{domain}_{item.get('task_id')}_{item.get('trial', 0)}_{idx}",
                "task_id": item.get("task_id"),
                "trial": item.get("trial", 0),
                "domain": domain,
                "reward": item.get("reward"),
                "turn_index": idx,
                "prompt": build_prompt(
                    domain,
                    trajectory,
                    idx,
                    max_context_messages,
                    max_message_chars,
                    max_system_chars,
                    action_names,
                ),
                "correct_action": action_name,
                "action_id": action_id,
                "action_names": action_names,
                "action_vocab": action_vocab,
                "expected_tool_name": tool_name,
            }
        )
    return rows


def balance_rows(rows: list[dict[str, Any]], max_per_action: int) -> list[dict[str, Any]]:
    if max_per_action <= 0:
        return rows
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[int(row["action_id"])].append(row)
    balanced = []
    for action_id in sorted(buckets):
        balanced.extend(buckets[action_id][:max_per_action])
    return balanced


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for path in args.trajectories:
        domain = domain_from_path(path)
        for item in load_json(path):
            if args.success_only and float(item.get("reward", 0.0)) < 1.0:
                continue
            rows.extend(
                convert_trajectory(
                    item,
                    domain,
                    args.max_context_messages,
                    args.max_message_chars,
                    args.max_system_chars,
                    args.action_vocab,
                )
            )

    rows = balance_rows(rows, args.max_per_action)
    counts = Counter(row["correct_action"] for row in rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "out": str(out),
                "num_rows": len(rows),
                "action_counts": dict(sorted(counts.items())),
                "action_names": TAU_HIERARCHICAL_ACTION_NAMES if args.action_vocab == "hierarchical" else TAU_ACTION_NAMES,
                "action_vocab": args.action_vocab,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
