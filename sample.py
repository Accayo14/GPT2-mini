from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from .model import GPT2, GPT2Config
    from .tokenizer import GPT2Tokenizer
except ImportError:
    from model import GPT2, GPT2Config
    from tokenizer import GPT2Tokenizer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained GPT-2 checkpoint.")
    parser.add_argument("--out-dir", type=str, default="out-gpt2-small")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.out_dir) / "ckpt.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    model = GPT2(GPT2Config(**checkpoint["model_args"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    tokenizer = GPT2Tokenizer()
    x = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    print(tokenizer.decode(y[0].tolist()))


if __name__ == "__main__":
    main()
