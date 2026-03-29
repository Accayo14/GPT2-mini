from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sample text from Hugging Face GPT-2 and save JSON output.")
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="huggingface/outputs")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    enc = tokenizer(args.prompt, return_tensors="pt").to(device)
    prompt_len = int(enc["input_ids"].shape[1])

    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            do_sample=True,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    full_text = tokenizer.decode(out_ids[0], skip_special_tokens=False)
    continuation_ids = out_ids[0][prompt_len:]
    continuation_text = tokenizer.decode(continuation_ids, skip_special_tokens=False)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hf_sample_{ts}_seed{args.seed}.json"

    payload = {
        "timestamp_utc": ts,
        "model_name": args.model_name,
        "input": args.prompt,
        "output": full_text,
        "continuation_only": continuation_text,
        "sampling_config": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "seed": args.seed,
        },
        "device": device,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(full_text)
    print(f"Saved JSON to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
