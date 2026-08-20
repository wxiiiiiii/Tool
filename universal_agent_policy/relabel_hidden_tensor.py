from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from universal_agent_policy.data import load_hidden_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--decision-jsonl", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    obj = load_hidden_tensor(args.tensor)
    rows = load_rows(args.decision_jsonl)
    if len(rows) != obj["h"].shape[0]:
        raise ValueError("decision rows and hidden tensor must have the same length")

    tensor_ids = obj.get("sample_id")
    if tensor_ids:
        row_ids = [row.get("sample_id", "") for row in rows]
        if list(tensor_ids)[: len(row_ids)] != row_ids:
            raise ValueError("decision rows and hidden tensor sample_id order differ")

    action_names = rows[0].get("action_names") if rows else None
    labels = torch.tensor([int(row["action_id"]) for row in rows], dtype=torch.long)
    relabeled = dict(obj)
    relabeled["y"] = labels
    relabeled["action_names"] = action_names

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(relabeled, out)
    print(
        json.dumps(
            {
                "out": str(out),
                "num_rows": len(rows),
                "action_names": action_names,
                "label_counts": {
                    str(idx): int((labels == idx).sum().item()) for idx in sorted(labels.unique().tolist())
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
