from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


TOOL_ACTIONS = {"THINK", "TRANSFER"}


def load_rows(path: str | Path, max_samples: int = -1) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def action_names_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    for row in rows:
        names = row.get("action_names")
        if names:
            return list(names)
    raise ValueError("No action_names found in decision rows")


def is_tool_action(action_name: str) -> bool:
    return action_name.startswith("ACT_") or action_name.startswith("SELECT_TOOL_") or action_name in TOOL_ACTIONS


def parse_action(text: str, action_names: list[str]) -> str:
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?think>", " ", text, flags=re.IGNORECASE)
    upper = text.upper()
    for name in sorted(action_names, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name.upper())}\b", upper):
            return name
    match = re.search(r"[A-Z][A-Z0-9_]+", upper)
    return match.group(0) if match else ""


def prediction_records(
    rows: list[dict[str, Any]],
    predicted_actions: list[str],
    method: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(rows) != len(predicted_actions):
        raise ValueError("rows and predicted_actions must have the same length")
    action_names = action_names_from_rows(rows)
    action_to_id = {name: idx for idx, name in enumerate(action_names)}
    records: list[dict[str, Any]] = []
    by_expected: dict[str, list[bool]] = defaultdict(list)
    by_domain: dict[str, list[bool]] = defaultdict(list)
    tool_results: list[bool] = []
    for row, predicted_action in zip(rows, predicted_actions, strict=True):
        expected_action = str(row["correct_action"])
        predicted_id = action_to_id.get(predicted_action, -1)
        expected_id = int(row.get("action_id", action_to_id.get(expected_action, -1)))
        correct = predicted_action == expected_action if predicted_id < 0 else predicted_id == expected_id
        by_expected[expected_action].append(correct)
        by_domain[str(row.get("domain", "unknown"))].append(correct)
        if is_tool_action(expected_action):
            tool_results.append(correct)
        records.append(
            {
                "sample_id": row.get("sample_id"),
                "domain": row.get("domain"),
                "task_id": row.get("task_id"),
                "trial": row.get("trial", 0),
                "turn_index": row.get("turn_index"),
                "expected_action": expected_action,
                "predicted_action": predicted_action,
                "predicted_action_id": predicted_id,
                "expected_action_id": expected_id,
                "correct": correct,
            }
        )
    total = max(1, len(records))
    per_action = {key: sum(vals) / max(1, len(vals)) for key, vals in sorted(by_expected.items())}
    per_domain = {key: sum(vals) / max(1, len(vals)) for key, vals in sorted(by_domain.items())}
    summary = {
        "method": method,
        "num_samples": len(records),
        "action_accuracy": sum(1 for row in records if row["correct"]) / total,
        "macro_action_accuracy": sum(per_action.values()) / max(1, len(per_action)),
        "tool_action_accuracy": sum(tool_results) / max(1, len(tool_results)),
        "per_action_accuracy": per_action,
        "per_domain_accuracy": per_domain,
    }
    if metadata:
        summary.update(metadata)
    return {"summary": summary, "records": records}


def load_prediction_file(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if "records" not in obj:
        raise ValueError("prediction file must contain records")
    return obj


def predictions_by_sample_id(path: str | Path) -> dict[str, str]:
    obj = load_prediction_file(path)
    return {str(row.get("sample_id")): str(row.get("predicted_action", "")) for row in obj["records"]}


def build_transition_pairs(rows: list[dict[str, Any]]) -> list[tuple[int, int]]:
    groups: dict[tuple[str, int, int], list[tuple[int, int]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        key = (
            str(row.get("domain", "unknown")),
            int(row.get("task_id", -1)),
            int(row.get("trial", 0)),
        )
        groups[key].append((int(row.get("turn_index", idx)), idx))
    pairs: list[tuple[int, int]] = []
    for items in groups.values():
        ordered = [idx for _, idx in sorted(items)]
        pairs.extend((left, right) for left, right in zip(ordered, ordered[1:]))
    return pairs


def transition_agreement(rows: list[dict[str, Any]], predictions: list[str], reference: list[str]) -> float:
    pairs = build_transition_pairs(rows)
    if not pairs:
        return 0.0
    matched = 0
    for left, right in pairs:
        matched += int(predictions[left] == reference[left] and predictions[right] == reference[right])
    return matched / len(pairs)


def add_reference_agreement(
    artifact: dict[str, Any],
    rows: list[dict[str, Any]],
    reference_predictions: dict[str, str],
    prefix: str = "reference",
) -> None:
    preds = [str(row.get("predicted_action", "")) for row in artifact["records"]]
    refs = [reference_predictions.get(str(row.get("sample_id")), "") for row in rows]
    valid = [bool(ref) for ref in refs]
    if not any(valid):
        return
    agreement = sum(int(pred == ref) for pred, ref, ok in zip(preds, refs, valid, strict=True) if ok) / sum(valid)
    artifact["summary"][f"{prefix}_policy_agreement"] = agreement
    artifact["summary"][f"{prefix}_transition_agreement"] = transition_agreement(rows, preds, refs)


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def torch_dtype_from_name(name: str) -> str | torch.dtype:
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]
