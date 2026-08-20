from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-samples", type=int, default=-1)
    return parser.parse_args()


def dtype_from_name(name: str) -> str | torch.dtype:
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_rows(path: str | Path, max_samples: int) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def row_to_text(tokenizer: Any, row: dict[str, Any]) -> str:
    if "messages" in row:
        return tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
    return str(row["prompt"])


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_jsonl, args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype_from_name(args.torch_dtype),
        device_map=args.device if args.device != "cpu" else None,
        trust_remote_code=args.trust_remote_code,
    )
    if args.device == "cpu":
        model.to("cpu")
    model.eval()

    hidden_states = []
    labels = []
    sample_ids = []
    categories = []
    action_names = rows[0].get("action_names") if rows else None
    with torch.no_grad():
        for row in tqdm(rows):
            text = row_to_text(tokenizer, row)
            encoded = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
            state = outputs.hidden_states[args.layer][0, -1].detach().float().cpu()
            hidden_states.append(state)
            labels.append(int(row["action_id"]))
            sample_ids.append(row.get("sample_id", ""))
            categories.append(row.get("category", ""))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "h": torch.stack(hidden_states),
            "y": torch.tensor(labels, dtype=torch.long),
            "sample_id": sample_ids,
            "category": categories,
            "action_names": action_names,
        },
        out,
    )


if __name__ == "__main__":
    main()
