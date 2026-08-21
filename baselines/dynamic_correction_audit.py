from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from baselines import eval_protocol
from baselines.tool_policy_utils import load_prediction_file, load_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether a dynamic policy correction fixes or damages a static "
            "policy-latent baseline on held-out trajectories."
        )
    )
    parser.add_argument("--decision-jsonl", required=True)
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--dynamic-predictions", required=True)
    parser.add_argument("--base-replay-json")
    parser.add_argument("--dynamic-replay-json")
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown-out")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{100 * value:.2f}%"


def avg(values: list[bool]) -> float:
    return sum(1.0 for value in values if value) / max(1, len(values))


def load_records_by_sample(path: str | Path) -> tuple[str, dict[str, dict[str, Any]]]:
    obj = load_prediction_file(path)
    method = str(obj.get("summary", {}).get("method", Path(path).stem))
    return method, {str(row.get("sample_id")): row for row in obj["records"]}


def load_replay_by_sample(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(row.get("sample_id")): row for row in obj.get("records", [])}


def load_replay_tasks(path: str | Path | None) -> dict[tuple[str, int, int], dict[str, Any]]:
    if not path:
        return {}
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = {}
    for row in obj.get("tasks", []):
        key = (str(row.get("domain")), int(row.get("task_id", -1)), int(row.get("trial", 0)))
        tasks[key] = row
    return tasks


def trajectory_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row.get("domain")), int(row.get("task_id", -1)), int(row.get("trial", 0)))


def action_family(action: str) -> str:
    if action.startswith("ACT_UPDATE"):
        return "ACT_UPDATE"
    if action.startswith("ACT_RETRIEVE") or action.startswith("ACT_SEARCH"):
        return "ACT_RETRIEVE"
    if action.startswith("ACT_"):
        return "ACT_OTHER"
    return action


def turn_phase(row: dict[str, Any], trajectory_len: int) -> str:
    turn = int(row.get("turn_index") or 0)
    if trajectory_len <= 1:
        return "single"
    rank = min(1.0, max(0.0, turn / max(1, 2 * trajectory_len)))
    if rank < 0.33:
        return "early"
    if rank < 0.67:
        return "middle"
    return "late"


def extract_last_tool_observation(prompt: str) -> str:
    marker = "Conversation state before the next assistant decision:"
    text = prompt.split(marker, 1)[-1] if marker in prompt else prompt
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    tool_lines = [line for line in lines if line.lower().startswith("tool ")]
    return tool_lines[-1] if tool_lines else ""


def observation_type(row: dict[str, Any]) -> str:
    obs = extract_last_tool_observation(str(row.get("prompt", ""))).lower()
    if not obs:
        return "no_previous_tool_observation"
    if any(token in obs for token in ("error", "invalid", "not found", "failed", "exception")):
        return "tool_failure_or_error"
    if any(token in obs for token in ("[]", "{}", "none", "null", "empty")):
        return "empty_or_null_result"
    if any(token in obs for token in ("cancelled", "canceled", "success", "updated", "booked", "created")):
        return "state_changing_success"
    if re.search(r"\b(true|false|\d+|[a-z0-9_]{6,})\b", obs):
        return "informative_result"
    return "other_tool_observation"


def previous_expected_actions(rows: list[dict[str, Any]]) -> dict[str, str]:
    by_traj: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_traj[trajectory_key(row)].append(row)
    prev: dict[str, str] = {}
    for items in by_traj.values():
        items.sort(key=lambda row: int(row.get("turn_index") or 0))
        last = "START"
        for row in items:
            prev[str(row.get("sample_id"))] = last
            last = str(row.get("correct_action"))
    return prev


def selected_rows(rows: list[dict[str, Any]], split_json: str) -> list[dict[str, Any]]:
    _, val_idx, _ = eval_protocol.load_split(split_json, len(rows))
    return [rows[int(idx)] for idx in val_idx.tolist()]


def summarize_bool(rows: list[dict[str, Any]], field: str) -> float:
    return avg([bool(row[field]) for row in rows])


def top_counter(counter: Counter[str], n: int = 20) -> list[dict[str, Any]]:
    return [{"key": key, "count": count} for key, count in counter.most_common(n)]


def bootstrap_by_trajectory(rows: list[dict[str, Any]], iters: int, seed: int) -> dict[str, Any]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[trajectory_key(row)].append(row)
    groups = list(grouped.values())
    rng = random.Random(seed)
    base_scores: list[float] = []
    dynamic_scores: list[float] = []
    deltas: list[float] = []
    for _ in range(iters):
        sample = [rng.choice(groups) for _ in groups]
        flat = [row for group in sample for row in group]
        base = summarize_bool(flat, "base_correct")
        dyn = summarize_bool(flat, "dynamic_correct")
        base_scores.append(base)
        dynamic_scores.append(dyn)
        deltas.append(dyn - base)

    def ci(values: list[float]) -> dict[str, float]:
        values = sorted(values)
        lo = values[int(0.025 * (len(values) - 1))]
        hi = values[int(0.975 * (len(values) - 1))]
        return {"mean": sum(values) / len(values), "ci95_low": lo, "ci95_high": hi}

    return {
        "base_action_accuracy": ci(base_scores),
        "dynamic_action_accuracy": ci(dynamic_scores),
        "dynamic_minus_base": ci(deltas),
    }


def main() -> None:
    args = parse_args()
    all_rows = load_rows(args.decision_jsonl)
    rows = selected_rows(all_rows, args.split_json)
    base_method, base_records = load_records_by_sample(args.base_predictions)
    dyn_method, dyn_records = load_records_by_sample(args.dynamic_predictions)
    base_replay = load_replay_by_sample(args.base_replay_json)
    dyn_replay = load_replay_by_sample(args.dynamic_replay_json)
    base_tasks = load_replay_tasks(args.base_replay_json)
    dyn_tasks = load_replay_tasks(args.dynamic_replay_json)
    prev_expected = previous_expected_actions(all_rows)

    traj_lengths = Counter(trajectory_key(row) for row in rows)
    audit_rows: list[dict[str, Any]] = []
    for row in rows:
        sid = str(row.get("sample_id"))
        if sid not in base_records or sid not in dyn_records:
            raise ValueError(f"Missing prediction for sample_id={sid}")
        base = base_records[sid]
        dyn = dyn_records[sid]
        base_correct = str(base.get("predicted_action")) == str(row.get("correct_action"))
        dyn_correct = str(dyn.get("predicted_action")) == str(row.get("correct_action"))
        changed = str(base.get("predicted_action")) != str(dyn.get("predicted_action"))
        if base_correct and dyn_correct:
            cell = "C_to_C"
        elif (not base_correct) and dyn_correct:
            cell = "B_to_C"
        elif base_correct and (not dyn_correct):
            cell = "C_to_B"
        else:
            cell = "B_to_B"
        replay_base = base_replay.get(sid, {})
        replay_dyn = dyn_replay.get(sid, {})
        audit_rows.append(
            {
                "sample_id": sid,
                "domain": str(row.get("domain")),
                "task_id": int(row.get("task_id", -1)),
                "trial": int(row.get("trial", 0)),
                "turn_index": int(row.get("turn_index") or 0),
                "turn_phase": turn_phase(row, traj_lengths[trajectory_key(row)]),
                "expected_action": str(row.get("correct_action")),
                "expected_family": action_family(str(row.get("correct_action"))),
                "previous_expected_action": prev_expected.get(sid, "START"),
                "base_predicted_action": str(base.get("predicted_action")),
                "dynamic_predicted_action": str(dyn.get("predicted_action")),
                "base_family": action_family(str(base.get("predicted_action"))),
                "dynamic_family": action_family(str(dyn.get("predicted_action"))),
                "base_correct": base_correct,
                "dynamic_correct": dyn_correct,
                "changed": changed,
                "cell": cell,
                "transition": f"{prev_expected.get(sid, 'START')}->{row.get('correct_action')}",
                "predicted_transition": f"{base.get('predicted_action')}->{dyn.get('predicted_action')}",
                "observation_type": observation_type(row),
                "base_execution_ok": replay_base.get("execution_ok"),
                "dynamic_execution_ok": replay_dyn.get("execution_ok"),
                "base_tool": replay_base.get("predicted_tool"),
                "dynamic_tool": replay_dyn.get("predicted_tool"),
            }
        )

    total = max(1, len(audit_rows))
    cell_counts = Counter(row["cell"] for row in audit_rows)
    changed_rows = [row for row in audit_rows if row["changed"]]
    improved = [row for row in audit_rows if row["cell"] == "B_to_C"]
    damaged = [row for row in audit_rows if row["cell"] == "C_to_B"]
    unchanged_wrong = [row for row in audit_rows if row["cell"] == "B_to_B"]

    def stratify(field: str, subset: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in subset:
            by_key[str(row.get(field))].append(row)
        out = []
        for key, items in by_key.items():
            out.append(
                {
                    "key": key,
                    "count": len(items),
                    "changed_rate": avg([row["changed"] for row in items]),
                    "base_accuracy": summarize_bool(items, "base_correct"),
                    "dynamic_accuracy": summarize_bool(items, "dynamic_correct"),
                    "delta": summarize_bool(items, "dynamic_correct") - summarize_bool(items, "base_correct"),
                    "B_to_C": sum(1 for row in items if row["cell"] == "B_to_C"),
                    "C_to_B": sum(1 for row in items if row["cell"] == "C_to_B"),
                }
            )
        out.sort(key=lambda row: (abs(row["delta"]), row["count"]), reverse=True)
        return out

    transition_flip_counter = Counter(
        f"{row['base_predicted_action']}->{row['dynamic_predicted_action']}" for row in changed_rows
    )
    damage_counter = Counter(
        f"{row['expected_action']} | {row['base_predicted_action']}->{row['dynamic_predicted_action']}"
        for row in damaged
    )
    improve_counter = Counter(
        f"{row['expected_action']} | {row['base_predicted_action']}->{row['dynamic_predicted_action']}"
        for row in improved
    )

    task_changes = []
    for key in sorted(set(base_tasks) | set(dyn_tasks)):
        base_ok = bool(base_tasks.get(key, {}).get("db_hash_match"))
        dyn_ok = bool(dyn_tasks.get(key, {}).get("db_hash_match"))
        if base_ok != dyn_ok:
            task_changes.append(
                {
                    "domain": key[0],
                    "task_id": key[1],
                    "trial": key[2],
                    "base_db_hash_match": base_ok,
                    "dynamic_db_hash_match": dyn_ok,
                    "change": "DB_fixed" if dyn_ok else "DB_broken",
                }
            )

    result = {
        "metadata": {
            "base_method": base_method,
            "dynamic_method": dyn_method,
            "num_heldout_samples": len(audit_rows),
            "num_changed_decisions": len(changed_rows),
            "changed_rate": len(changed_rows) / total,
        },
        "summary": {
            "base_action_accuracy": summarize_bool(audit_rows, "base_correct"),
            "dynamic_action_accuracy": summarize_bool(audit_rows, "dynamic_correct"),
            "dynamic_minus_base": summarize_bool(audit_rows, "dynamic_correct") - summarize_bool(audit_rows, "base_correct"),
            "B_to_C_rate": len(improved) / total,
            "C_to_B_rate": len(damaged) / total,
            "net_fixed_minus_broken": (len(improved) - len(damaged)) / total,
            "cell_counts": dict(cell_counts),
        },
        "bootstrap": bootstrap_by_trajectory(audit_rows, args.bootstrap_iters, args.seed),
        "stratified": {
            "domain": stratify("domain", audit_rows),
            "turn_phase": stratify("turn_phase", audit_rows),
            "observation_type": stratify("observation_type", audit_rows),
            "expected_family": stratify("expected_family", audit_rows),
            "previous_expected_action": stratify("previous_expected_action", audit_rows),
            "transition": stratify("transition", audit_rows),
        },
        "changed_action_flips": top_counter(transition_flip_counter),
        "damaging_flips": top_counter(damage_counter),
        "improving_flips": top_counter(improve_counter),
        "db_task_changes": task_changes,
        "examples": {
            "B_to_C": improved[:30],
            "C_to_B": damaged[:30],
            "B_to_B": unchanged_wrong[:30],
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.markdown_out:
        lines = [
            "# Dynamic Correction Audit",
            "",
            f"- Base: `{base_method}`",
            f"- Dynamic: `{dyn_method}`",
            f"- held-out samples: {len(audit_rows)}",
            f"- changed decisions: {len(changed_rows)} ({pct(len(changed_rows) / total)})",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Base action accuracy | {pct(result['summary']['base_action_accuracy'])} |",
            f"| Dynamic action accuracy | {pct(result['summary']['dynamic_action_accuracy'])} |",
            f"| Dynamic - Base | {pct(result['summary']['dynamic_minus_base'])} |",
            f"| B->C fixed | {len(improved)} ({pct(result['summary']['B_to_C_rate'])}) |",
            f"| C->B broken | {len(damaged)} ({pct(result['summary']['C_to_B_rate'])}) |",
            f"| Net fixed - broken | {pct(result['summary']['net_fixed_minus_broken'])} |",
            "",
            "## Trajectory Bootstrap 95% CI",
            "",
            "| Metric | Mean | 95% CI |",
            "|---|---:|---:|",
        ]
        for key, label in (
            ("base_action_accuracy", "Base action accuracy"),
            ("dynamic_action_accuracy", "Dynamic action accuracy"),
            ("dynamic_minus_base", "Dynamic - Base"),
        ):
            ci = result["bootstrap"][key]
            lines.append(f"| {label} | {pct(ci['mean'])} | [{pct(ci['ci95_low'])}, {pct(ci['ci95_high'])}] |")
        lines.extend(["", "## Stratified Delta", ""])
        for field in ("observation_type", "expected_family", "turn_phase", "previous_expected_action"):
            lines.extend([f"### {field}", "", "| Key | Count | Changed | Base Acc | Dynamic Acc | Delta | B->C | C->B |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
            for row in result["stratified"][field][:15]:
                lines.append(
                    f"| {row['key']} | {row['count']} | {pct(row['changed_rate'])} | {pct(row['base_accuracy'])} | "
                    f"{pct(row['dynamic_accuracy'])} | {pct(row['delta'])} | {row['B_to_C']} | {row['C_to_B']} |"
                )
            lines.append("")
        lines.extend(["## Most Common Changed Action Flips", "", "| Base -> Dynamic | Count |", "|---|---:|"])
        lines.extend(f"| {row['key']} | {row['count']} |" for row in result["changed_action_flips"][:15])
        lines.extend(["", "## Damaging Flips", "", "| Expected / Base -> Dynamic | Count |", "|---|---:|"])
        lines.extend(f"| {row['key']} | {row['count']} |" for row in result["damaging_flips"][:15])
        lines.extend(["", "## Improving Flips", "", "| Expected / Base -> Dynamic | Count |", "|---|---:|"])
        lines.extend(f"| {row['key']} | {row['count']} |" for row in result["improving_flips"][:15])
        if task_changes:
            lines.extend(["", "## DB Hash Task Changes", "", "| Task | Change |", "|---|---:|"])
            for row in task_changes[:30]:
                lines.append(f"| {row['domain']}:{row['task_id']}:{row['trial']} | {row['change']} |")
        Path(args.markdown_out).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
