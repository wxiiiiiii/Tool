from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from baselines import eval_protocol
from baselines.skill_policy_validation import action_to_skill, load_rows
from baselines.tool_policy_utils import load_prediction_file


SKILL_TO_OUTCOME_TYPE = {
    "COLLECT_INFO": "Information / Entity Resolution",
    "CONFIRM": "Verify / Confirm",
    "EXECUTE": "Commit / Execute",
    "RECOVER_OR_ELICIT": "Recovery",
    "FINISH": "Finish",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit which skill-level corrections affect final tau-bench task outcomes. "
            "This script consumes precomputed skill predictions and replay JSONs."
        )
    )
    parser.add_argument("--decision-jsonl", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--replay-prefix", default="smollm3_v10")
    parser.add_argument("--static-method", default="static_action")
    parser.add_argument("--corrected-method", default="source_discovered_cluster_binder")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def task_key_from_row(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row.get("domain")), int(row.get("task_id", -1)), int(row.get("trial", 0)))


def task_key_from_task(task: dict[str, Any]) -> tuple[str, int, int]:
    return (str(task.get("domain")), int(task.get("task_id", -1)), int(task.get("trial", 0)))


def replay_path(replay_dir: Path, prefix: str, method: str) -> Path:
    return replay_dir / f"{prefix}_{method}_replay_oracle_args.json"


def load_replay(replay_dir: Path, prefix: str, method: str) -> dict[str, Any]:
    path = replay_path(replay_dir, prefix, method)
    if not path.exists():
        raise FileNotFoundError(f"Missing replay file for {method}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_actions(predictions_dir: Path, method: str) -> list[str]:
    path = predictions_dir / f"{method}_predictions.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file for {method}: {path}")
    records = load_prediction_file(path)["records"]
    return [str(record.get("predicted_action")) for record in records]


def task_outcomes(replay: dict[str, Any]) -> dict[tuple[str, int, int], bool]:
    return {task_key_from_task(task): bool(task.get("db_hash_match")) for task in replay.get("tasks", [])}


def task_records_by_key(rows: list[dict[str, Any]], selected: list[int]) -> dict[tuple[str, int, int], list[int]]:
    selected_set = set(selected)
    grouped: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        if idx in selected_set:
            grouped[task_key_from_row(row)].append(idx)
    for key in grouped:
        grouped[key].sort(key=lambda idx: int(rows[idx].get("turn_index") or 0))
    return dict(grouped)


def first_db_mutation_index(rows: list[dict[str, Any]], indices: list[int]) -> int | None:
    for pos, idx in enumerate(indices):
        action = str(rows[idx].get("correct_action"))
        if action.startswith(("ACT_UPDATE", "ACT_SEND")):
            return pos
    return None


def first_difference(
    rows: list[dict[str, Any]],
    indices: list[int],
    static_actions: list[str],
    corrected_actions: list[str],
) -> dict[str, Any] | None:
    mutation_pos = first_db_mutation_index(rows, indices)
    for pos, idx in enumerate(indices):
        static_action = static_actions[idx]
        corrected_action = corrected_actions[idx]
        static_skill = action_to_skill(static_action)
        corrected_skill = action_to_skill(corrected_action)
        if static_action == corrected_action and static_skill == corrected_skill:
            continue
        row = rows[idx]
        return {
            "sample_id": row.get("sample_id"),
            "domain": row.get("domain"),
            "task_id": row.get("task_id"),
            "trial": row.get("trial", 0),
            "turn_index": row.get("turn_index"),
            "before_first_db_mutation": mutation_pos is None or pos < mutation_pos,
            "expected_action": row.get("correct_action"),
            "expected_skill": action_to_skill(str(row.get("correct_action"))),
            "static_action": static_action,
            "static_skill": static_skill,
            "corrected_action": corrected_action,
            "corrected_skill": corrected_skill,
            "outcome_type": SKILL_TO_OUTCOME_TYPE.get(corrected_skill, "Other"),
            "observation_type": infer_observation_type(str(row.get("prompt") or "")),
        }
    return None


def infer_observation_type(prompt: str) -> str:
    lower = prompt.lower()
    if any(token in lower for token in ["permission denied", "not authorized", "unauthorized"]):
        return "permission_denied"
    if any(token in lower for token in ["error:", "failed", "invalid", "not found", "exception"]):
        return "failure"
    if any(token in lower for token in ["empty", "no results", "[]", "null"]):
        return "empty"
    if any(token in lower for token in ["success", "updated", "cancelled", "booked", "created"]):
        return "success"
    return "none"


def replay_summary_row(method: str, replay: dict[str, Any]) -> dict[str, Any]:
    summary = replay.get("summary", {})
    return {
        "method": method,
        "action_accuracy": float(summary.get("action_accuracy", 0.0)),
        "tool_execution_ok": float(summary.get("tool_execution_ok_rate_predicted_tool", 0.0)),
        "db_hash": float(summary.get("db_hash_match_rate", 0.0)),
        "num_tasks": int(summary.get("num_tasks", 0)),
    }


def task_level_audit(
    rows: list[dict[str, Any]],
    selected: list[int],
    static_actions: list[str],
    corrected_actions: list[str],
    static_outcome: dict[tuple[str, int, int], bool],
    corrected_outcome: dict[tuple[str, int, int], bool],
) -> dict[str, Any]:
    grouped = task_records_by_key(rows, selected)
    quadrants: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    first_diffs: list[dict[str, Any]] = []
    for key, indices in sorted(grouped.items()):
        static_ok = static_outcome.get(key, False)
        corrected_ok = corrected_outcome.get(key, False)
        quadrant = ("S" if static_ok else "F") + "->" + ("S" if corrected_ok else "F")
        quadrants[quadrant].append(key)
        diff = first_difference(rows, indices, static_actions, corrected_actions)
        if diff:
            diff["quadrant"] = quadrant
            first_diffs.append(diff)

    repair_counts = Counter()
    repair_pre_mutation = Counter()
    regression_counts = Counter()
    for diff in first_diffs:
        if diff["quadrant"] == "F->S":
            repair_counts[diff["outcome_type"]] += 1
            if diff["before_first_db_mutation"]:
                repair_pre_mutation[diff["outcome_type"]] += 1
        if diff["quadrant"] == "S->F":
            regression_counts[diff["outcome_type"]] += 1

    return {
        "quadrants": {name: len(keys) for name, keys in sorted(quadrants.items())},
        "repair_type_counts": dict(sorted(repair_counts.items())),
        "repair_type_pre_mutation_counts": dict(sorted(repair_pre_mutation.items())),
        "regression_type_counts": dict(sorted(regression_counts.items())),
        "first_differences": first_diffs,
    }


def deployability_gap(
    rows: list[dict[str, Any]],
    selected: list[int],
    source_skill_actions: list[str],
    deployable_actions: list[str],
    static_replay: dict[str, Any],
    source_replay: dict[str, Any],
    deployable_replay: dict[str, Any],
) -> dict[str, Any]:
    expected = [str(row.get("correct_action")) for row in rows]
    source_skills = [action_to_skill(action) for action in source_skill_actions]
    deployable_skills = [action_to_skill(action) for action in deployable_actions]
    correct_skill = [deployable_skills[idx] == source_skills[idx] for idx in selected]
    correct_branch_given_skill = [
        deployable_actions[idx] == source_skill_actions[idx]
        for idx in selected
        if deployable_skills[idx] == source_skills[idx]
    ]
    correct_action_given_skill = [
        deployable_actions[idx] == expected[idx]
        for idx in selected
        if deployable_skills[idx] == source_skills[idx]
    ]
    return {
        "p_target_skill_equals_source_skill": mean(correct_skill),
        "p_source_action_given_correct_skill": mean(correct_branch_given_skill),
        "p_gt_action_given_correct_skill": mean(correct_action_given_skill),
        "static_db_hash": float(static_replay["summary"].get("db_hash_match_rate", 0.0)),
        "source_skill_db_hash": float(source_replay["summary"].get("db_hash_match_rate", 0.0)),
        "deployable_db_hash": float(deployable_replay["summary"].get("db_hash_match_rate", 0.0)),
        "db_gap_source_minus_deployable": (
            float(source_replay["summary"].get("db_hash_match_rate", 0.0))
            - float(deployable_replay["summary"].get("db_hash_match_rate", 0.0))
        ),
    }


def mean(values: list[bool]) -> float:
    return sum(1.0 for value in values if value) / max(1, len(values))


def oracle_decomposition(
    replay_dir: Path,
    prefix: str,
    static_replay: dict[str, Any],
) -> dict[str, Any]:
    methods = [
        "static_action",
        "oracle_only_collect_info",
        "oracle_only_confirm",
        "oracle_only_execute",
        "oracle_only_recover_or_elicit",
        "oracle_only_finish",
        "oracle_skill_rerank",
    ]
    aliases = {
        "oracle_only_collect_info": "Oracle Entity / Information",
        "oracle_only_confirm": "Oracle Verify / Confirm",
        "oracle_only_execute": "Oracle Commit / Execute",
        "oracle_only_recover_or_elicit": "Oracle Recovery",
        "oracle_only_finish": "Oracle Finish",
        "oracle_skill_rerank": "Full Oracle Skill",
        "static_action": "Static",
    }
    rows = []
    static_db = float(static_replay["summary"].get("db_hash_match_rate", 0.0))
    for method in methods:
        try:
            replay = load_replay(replay_dir, prefix, method)
        except FileNotFoundError:
            continue
        row = replay_summary_row(aliases[method], replay)
        row["method_id"] = method
        row["db_delta_vs_static"] = row["db_hash"] - static_db
        rows.append(row)
    return {"rows": rows}


def write_report(payload: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "outcome_critical_skill_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    def pct(value: float) -> str:
        return f"{100 * value:.2f}%"

    lines = [
        "# Outcome-Critical Skill Audit",
        "",
        "## Replay Summary",
        "",
        "| Method | Action Acc | Tool Exec OK | DB Hash | Tasks |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["replay_summary"]:
        lines.append(
            f"| {row['method']} | {pct(row['action_accuracy'])} | "
            f"{pct(row['tool_execution_ok'])} | {pct(row['db_hash'])} | {row['num_tasks']} |"
        )
    lines.extend(["", "## Static vs Corrected Task-Level Audit", ""])
    lines.append("| Quadrant | Tasks |")
    lines.append("|---|---:|")
    for name, count in payload["task_audit"]["quadrants"].items():
        lines.append(f"| {name} | {count} |")
    lines.extend(["", "### F -> S Repair Types", ""])
    lines.append("| Outcome Type | Repaired Tasks | Before First DB Mutation |")
    lines.append("|---|---:|---:|")
    repair_counts = payload["task_audit"]["repair_type_counts"]
    pre_counts = payload["task_audit"]["repair_type_pre_mutation_counts"]
    for name, count in repair_counts.items():
        lines.append(f"| {name} | {count} | {pre_counts.get(name, 0)} |")
    lines.extend(["", "### S -> F Regression Types", ""])
    lines.append("| Outcome Type | Regressed Tasks |")
    lines.append("|---|---:|")
    for name, count in payload["task_audit"]["regression_type_counts"].items():
        lines.append(f"| {name} | {count} |")
    lines.extend(["", "## Outcome-Critical Oracle Decomposition", ""])
    lines.append("| Oracle Setting | Action Acc | Tool Exec OK | DB Hash | DB Delta vs Static |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in payload["oracle_decomposition"]["rows"]:
        lines.append(
            f"| {row['method']} | {pct(row['action_accuracy'])} | {pct(row['tool_execution_ok'])} | "
            f"{pct(row['db_hash'])} | {pct(row['db_delta_vs_static'])} |"
        )
    lines.extend(["", "## Deployability Gap", ""])
    gap = payload["deployability_gap"]
    lines.append("| Layer | Value |")
    lines.append("|---|---:|")
    lines.append(f"| P(Target Skill = Source Skill) | {pct(gap['p_target_skill_equals_source_skill'])} |")
    lines.append(f"| P(Source Action | Correct Skill) | {pct(gap['p_source_action_given_correct_skill'])} |")
    lines.append(f"| P(GT Action | Correct Skill) | {pct(gap['p_gt_action_given_correct_skill'])} |")
    lines.append(f"| Static DB Hash | {pct(gap['static_db_hash'])} |")
    lines.append(f"| Source Skill DB Hash | {pct(gap['source_skill_db_hash'])} |")
    lines.append(f"| Deployable DB Hash | {pct(gap['deployable_db_hash'])} |")
    lines.append(f"| DB Gap Source Skill - Deployable | {pct(gap['db_gap_source_minus_deployable'])} |")
    lines.extend(["", "## Go / No-Go", ""])
    lines.append(payload["go_no_go"])
    lines.append("")
    (out_dir / "outcome_critical_skill_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.decision_jsonl)
    _, val_idx, split_metadata = eval_protocol.load_split(args.split_json, len(rows))
    selected = [int(idx) for idx in val_idx.tolist()]
    predictions_dir = Path(args.predictions_dir)
    replay_dir = Path(args.replay_dir)

    static_replay = load_replay(replay_dir, args.replay_prefix, args.static_method)
    corrected_replay = load_replay(replay_dir, args.replay_prefix, args.corrected_method)
    source_skill_replay = load_replay(replay_dir, args.replay_prefix, "source_skill_rerank")
    oracle_skill_replay = load_replay(replay_dir, args.replay_prefix, "oracle_skill_rerank")

    static_actions = load_actions(predictions_dir, args.static_method)
    corrected_actions = load_actions(predictions_dir, args.corrected_method)
    source_skill_actions = load_actions(predictions_dir, "source_skill_rerank")

    task_audit = task_level_audit(
        rows,
        selected,
        static_actions,
        corrected_actions,
        task_outcomes(static_replay),
        task_outcomes(corrected_replay),
    )
    oracle = oracle_decomposition(replay_dir, args.replay_prefix, static_replay)
    gap = deployability_gap(
        rows,
        selected,
        source_skill_actions,
        corrected_actions,
        static_replay,
        source_skill_replay,
        corrected_replay,
    )
    replay_summary = [
        replay_summary_row(args.static_method, static_replay),
        replay_summary_row(args.corrected_method, corrected_replay),
        replay_summary_row("source_skill_rerank", source_skill_replay),
        replay_summary_row("oracle_skill_rerank", oracle_skill_replay),
    ]
    best_delta = max((row["db_delta_vs_static"] for row in oracle["rows"]), default=0.0)
    go_no_go = (
        "GO: at least one oracle skill class improves DB Hash; continue with outcome-weighted skill binding."
        if best_delta >= 0.02
        else "NO-GO for more binder tuning: single-class oracle skills do not clearly improve DB Hash."
    )
    payload = {
        "metadata": {
            "decision_jsonl": args.decision_jsonl,
            "split_json": args.split_json,
            "split": split_metadata,
            "predictions_dir": str(predictions_dir),
            "replay_dir": str(replay_dir),
            "replay_prefix": args.replay_prefix,
            "static_method": args.static_method,
            "corrected_method": args.corrected_method,
            "selected_samples": len(selected),
        },
        "replay_summary": replay_summary,
        "task_audit": task_audit,
        "oracle_decomposition": oracle,
        "deployability_gap": gap,
        "go_no_go": go_no_go,
    }
    write_report(payload, Path(args.out_dir))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
