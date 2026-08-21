from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines import eval_protocol
from baselines.tool_policy_utils import load_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a reusable decision/trajectory/task-disjoint evaluation split.")
    parser.add_argument("--decision-jsonl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split-unit", choices=["decision", "trajectory", "task"], default="trajectory")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.decision_jsonl)
    train_idx, val_idx, metadata = eval_protocol.split_indices_for_protocol(
        len(rows), args.val_frac, args.seed, rows, args.split_unit
    )
    metadata["decision_jsonl"] = args.decision_jsonl
    eval_protocol.save_split(args.out, train_idx, val_idx, metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
