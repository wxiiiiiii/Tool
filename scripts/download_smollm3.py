from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM3-3B")
    parser.add_argument("--cache-dir", default="/root/data/huggingface")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"model_id={args.model_id}")
    print(f"cache_dir={cache_dir}")
    print(f"cuda_available={torch.cuda.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=str(cache_dir))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        cache_dir=str(cache_dir),
        device_map="auto",
        torch_dtype="auto",
    )
    print(f"model_device={model.device}")

    messages = [{"role": "user", "content": "Who are you?"}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    print("generation:")
    print(text)


if __name__ == "__main__":
    main()
