from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from universal_agent_policy.runtime.constrained_decode import SchemaConstraintError, coerce_value
from universal_agent_policy.runtime.grounding import ToolSpec


class ArgumentGenerator(Protocol):
    def generate(self, row: dict[str, Any], tool: ToolSpec) -> str:
        ...


def _messages_to_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return str(messages)
    if messages and isinstance(messages[0], list):
        messages = messages[0]
    return "\n".join(f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages)


def _extract_conversation_state(text: str, max_lines: int = 16) -> str:
    marker = "Conversation state before the next assistant decision:"
    if marker in text:
        text = text.split(marker, 1)[1]
    end_marker = "Next policy action:"
    if end_marker in text:
        text = text.split(end_marker, 1)[0]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:])


def _row_conversation(row: dict[str, Any]) -> str:
    if "prompt" in row:
        return _extract_conversation_state(str(row["prompt"]))
    return _messages_to_text(row.get("original_question", row.get("question", row.get("messages", ""))))


def _first_ground_truth_args(row: dict[str, Any], tool_name: str) -> dict[str, Any] | None:
    for call in row.get("ground_truth", []):
        if tool_name in call:
            raw_args = call[tool_name]
            return _canonicalize_bfcl_args(raw_args)
    return None


def _first_ground_truth_args_for_schema(
    row: dict[str, Any],
    tool_name: str,
    schema: dict[str, Any],
) -> dict[str, Any] | None:
    for call in row.get("ground_truth", []):
        if tool_name not in call:
            continue
        return _canonicalize_bfcl_args_with_schema(call[tool_name], schema)
    return None


def _canonicalize_bfcl_args(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonicalize_bfcl_args(item) for key, item in value.items()}
    if isinstance(value, list):
        if not value:
            return []
        if len(value) == 1:
            return _canonicalize_bfcl_args(value[0])
        if all(not isinstance(item, (dict, list)) for item in value):
            return value[0]
        return [_canonicalize_bfcl_args(item) for item in value]
    return value


def _canonicalize_bfcl_args_with_schema(value: Any, schema: dict[str, Any]) -> Any:
    schema_type = schema.get("type", "string")
    if schema_type == "dict":
        schema_type = "object"

    if schema_type == "object":
        properties = dict(schema.get("properties", {}))
        if isinstance(value, list):
            for candidate in value:
                if isinstance(candidate, dict):
                    return _canonicalize_bfcl_args_with_schema(candidate, schema)
            return {}
        if not isinstance(value, dict):
            return {}
        return {
            key: _canonicalize_bfcl_args_with_schema(raw_value, dict(properties.get(key, {})))
            for key, raw_value in value.items()
        }

    if schema_type == "array":
        item_schema = dict(schema.get("items", {}))
        if not isinstance(value, list):
            return [_canonicalize_bfcl_args_with_schema(value, item_schema)]
        if value and all(isinstance(item, list) for item in value):
            value = value[0]
        return [_canonicalize_bfcl_args_with_schema(item, item_schema) for item in value]

    if isinstance(value, list):
        for candidate in value:
            try:
                return coerce_value(candidate, schema)
            except (SchemaConstraintError, TypeError, ValueError):
                continue
        return _canonicalize_bfcl_args(value)

    return value


class EmptyArgumentGenerator:
    def generate(self, row: dict[str, Any], tool: ToolSpec) -> str:
        return "{}"


class OracleArgumentGenerator:
    """Uses BFCL possible_answer args.

    This isolates policy/grounding quality from argument generation quality. It
    should be reported as an oracle-args experiment, not as an end-to-end agent.
    """

    def generate(self, row: dict[str, Any], tool: ToolSpec) -> str:
        args = _first_ground_truth_args_for_schema(row, tool.name, tool.parameters)
        return json.dumps(args or {}, ensure_ascii=False)


@dataclass
class HFJsonArgumentGenerator:
    model: Any
    tokenizer: Any
    max_new_tokens: int = 192
    temperature: float = 0.0

    def generate(self, row: dict[str, Any], tool: ToolSpec) -> str:
        encoded = self._encode_prompt(row, tool)
        do_sample = self.temperature > 0
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0, encoded["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _encode_prompt(self, row: dict[str, Any], tool: ToolSpec) -> Any:
        messages = self._build_messages(row, tool)
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
        return self.tokenizer(self._messages_to_prompt(messages), return_tensors="pt").to(self.model.device)

    @staticmethod
    def _build_messages(row: dict[str, Any], tool: ToolSpec) -> list[dict[str, str]]:
        schema = json.dumps(tool.parameters, ensure_ascii=False, sort_keys=True)
        conversation = _row_conversation(row)
        return [
            {
                "role": "system",
                "content": (
                    "You generate arguments for exactly one function call. "
                    "Return only one valid JSON object. The first character must be { and the last character must be }. "
                    "Do not reason, do not use thinking tags, do not include markdown, and do not explain."
                ),
            },
            {
                "role": "user",
                "content": (
                    "/no_think\n"
                    "Extract the function arguments from the conversation state. "
                    "Use exact IDs and values that appear in the state. If a required value is absent, use an empty string.\n\n"
                    f"Conversation state:\n{conversation}\n\n"
                    f"Function name: {tool.name}\n"
                    f"Function description: {tool.description}\n"
                    f"JSON schema:\n{schema}\n\n"
                    "Return only the arguments JSON object. Start now with {"
                ),
            },
        ]

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
        return (
            f"System: {messages[0]['content']}\n\n"
            f"User: {messages[1]['content']}\n\n"
            "Arguments JSON:"
        )

    @staticmethod
    def _build_prompt(row: dict[str, Any], tool: ToolSpec) -> str:
        messages = HFJsonArgumentGenerator._build_messages(row, tool)
        return (
            messages[0]["content"]
            + "\n\n"
            + messages[1]["content"]
            + "\n\nArguments JSON:"
        )
