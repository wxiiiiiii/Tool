from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-jsonl", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=60)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=16)
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


def parse_action(text: str, action_names: list[str]) -> str:
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?think>", " ", text, flags=re.IGNORECASE)
    upper = text.upper()
    for name in action_names:
        if re.search(rf"\b{re.escape(name.upper())}\b", upper):
            return name
    match = re.search(r"[A-Z][A-Z0-9_]+", upper)
    return match.group(0) if match else ""


def prompt_for(row: dict[str, Any], action_names: list[str]) -> str:
    return (
        f"{row['prompt']}\n\n"
        "You are a strict classifier. Do not reason. Do not output <think> tags.\n"
        "Return exactly one action name from this list and no other text:\n"
        f"{', '.join(action_names)}"
    )


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_rows(args.decision_jsonl, args.max_samples)
    if not rows:
        raise ValueError("No rows loaded")
    action_names = list(rows[0]["action_names"])
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype_from_name(args.torch_dtype),
        device_map=args.device if args.device != "cpu" else None,
    )
    if args.device == "cpu":
        model.to("cpu")
    model.eval()

    records = []
    for row in rows:
        messages = [
            {"role": "system", "content": "Return only the requested label. No reasoning, no explanation."},
            {"role": "user", "content": prompt_for(row, action_names)},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        predicted = parse_action(text, action_names)
        expected = str(row["correct_action"])
        records.append(
            {
                "sample_id": row.get("sample_id"),
                "expected_action": expected,
                "predicted_action": predicted,
                "raw_output": text,
                "correct": predicted == expected,
            }
        )

    summary = {
        "method": "native_backbone_prompting",
        "model_id": args.model_id,
        "num_samples": len(records),
        "action_accuracy": sum(1 for row in records if row["correct"]) / max(1, len(records)),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
