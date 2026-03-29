from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_prompts(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list) or not all(isinstance(x, str) for x in payload):
        raise ValueError("Prompts file must be a JSON array of strings.")
    if not payload:
        raise ValueError("Prompts list is empty.")
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Batch sample Hugging Face GPT-2 and save one JSON per prompt.")
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--prompts-file", type=str, default="huggingface/prompts.json")
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

    prompts = _load_prompts(Path(args.prompts_file))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for idx, prompt in enumerate(prompts):
        enc = tokenizer(prompt, return_tensors="pt").to(device)
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

        payload = {
            "timestamp_utc": run_ts,
            "model_name": args.model_name,
            "input": prompt,
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
            "index": idx,
        }

        out_path = out_dir / f"hf_sample_{run_ts}_{idx:03d}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[{idx}] saved {out_path}")


if __name__ == "__main__":
    main()
