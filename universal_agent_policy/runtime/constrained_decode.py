from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


class SchemaConstraintError(ValueError):
    pass


_MISSING = object()


@dataclass
class DecodeReport:
    raw_text: str
    parsed: dict[str, Any]
    constrained: dict[str, Any]
    valid: bool
    errors: list[str] = field(default_factory=list)


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text.strip(), flags=re.DOTALL).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        obj = _extract_balanced_json_object(cleaned)
    if not isinstance(obj, dict):
        raise SchemaConstraintError("Generated JSON must be an object")
    return obj


def _extract_balanced_json_object(text: str) -> dict[str, Any]:
    starts = [idx for idx, char in enumerate(text) if char == "{"]
    for start in starts:
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : idx + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
    raise SchemaConstraintError("No JSON object found in generated text")


def _schema_type(schema: dict[str, Any]) -> str:
    schema_type = schema.get("type", "string")
    if schema_type == "dict":
        return "object"
    return str(schema_type)


def _coerce_scalar(value: Any, schema: dict[str, Any]) -> Any:
    schema_type = _schema_type(schema)
    if value is None:
        return None
    if schema_type == "string":
        return str(value)
    if schema_type == "integer":
        if isinstance(value, bool):
            raise SchemaConstraintError("boolean cannot satisfy integer")
        return int(value)
    if schema_type == "number" or schema_type == "float":
        if isinstance(value, bool):
            raise SchemaConstraintError("boolean cannot satisfy number")
        return float(value)
    if schema_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1"}:
                return True
            if lowered in {"false", "no", "0"}:
                return False
        raise SchemaConstraintError(f"Cannot coerce {value!r} to boolean")
    return value


def coerce_value(value: Any, schema: dict[str, Any]) -> Any:
    enum = schema.get("enum")
    schema_type = _schema_type(schema)

    if schema_type == "object":
        if not isinstance(value, dict):
            raise SchemaConstraintError("Expected object")
        return constrain_arguments(value, schema)

    if schema_type == "array":
        items_schema = dict(schema.get("items", {}))
        values = value if isinstance(value, list) else [value]
        return [coerce_value(item, items_schema) for item in values]

    coerced = _coerce_scalar(value, schema)
    if enum is not None and coerced not in enum:
        enum_as_strings = {str(item): item for item in enum}
        if str(coerced) in enum_as_strings:
            coerced = enum_as_strings[str(coerced)]
        else:
            raise SchemaConstraintError(f"{coerced!r} is not in enum {enum!r}")
    return coerced


def constrain_arguments(args: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    properties = dict(schema.get("properties", {}))
    required = set(schema.get("required", []))
    constrained: dict[str, Any] = {}

    for key, prop_schema in properties.items():
        if key in args:
            constrained[key] = coerce_value(args[key], dict(prop_schema))
        elif "default" in prop_schema:
            constrained[key] = prop_schema["default"]
        else:
            inferred_default = _default_from_description(dict(prop_schema))
            if inferred_default is not _MISSING:
                constrained[key] = coerce_value(inferred_default, dict(prop_schema))
            elif key in required:
                raise SchemaConstraintError(f"Missing required argument: {key}")

    missing = required.difference(constrained)
    if missing:
        raise SchemaConstraintError(f"Missing required arguments: {sorted(missing)}")
    return constrained


def _default_from_description(schema: dict[str, Any]) -> Any:
    description = str(schema.get("description", ""))
    if not description:
        return _MISSING
    patterns = [
        r"default(?:s)?\s+to\s+['\"]?([^'\".,;)]+)",
        r"default\s+(?:value\s+)?is\s+['\"]?([^'\".,;)]+)",
        r"default\s+value\s*[:=]\s*['\"]?([^'\".,;)]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, description, flags=re.IGNORECASE)
        if match:
            return _parse_default_literal(match.group(1).strip())
    return _MISSING


def _parse_default_literal(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("'\"")


class JsonSchemaConstrainedDecoder:
    def decode(self, raw_text: str, schema: dict[str, Any]) -> DecodeReport:
        errors: list[str] = []
        try:
            parsed = extract_json_object(raw_text)
        except Exception as exc:
            return DecodeReport(raw_text, {}, {}, False, [str(exc)])

        try:
            constrained = constrain_arguments(parsed, schema)
        except Exception as exc:
            errors.append(str(exc))
            for candidate in _wrapped_argument_candidates(parsed, schema):
                try:
                    constrained = constrain_arguments(candidate, schema)
                    return DecodeReport(raw_text, parsed, constrained, True, errors)
                except Exception as candidate_exc:
                    errors.append(str(candidate_exc))
            return DecodeReport(raw_text, parsed, {}, False, errors)

        return DecodeReport(raw_text, parsed, constrained, True, errors)


def _wrapped_argument_candidates(parsed: dict[str, Any], schema: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    properties = set(dict(schema.get("properties", {})))
    for key in ("arguments", "args", "parameters"):
        value = parsed.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    if len(parsed) == 1:
        value = next(iter(parsed.values()))
        if isinstance(value, dict):
            candidates.append(value)
    for key, value in parsed.items():
        if key not in properties and isinstance(value, dict):
            candidates.append(value)
    return candidates
