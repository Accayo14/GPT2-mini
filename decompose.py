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
    parser = argparse.ArgumentParser(description="Decompose logits into residual components and attention heads.")
    parser.add_argument("--out-dir", type=str, default="out-gpt2-small")
    parser.add_argument("--prompt", type=str, default="The meaning of life is")
    parser.add_argument("--position", type=int, default=-1, help="Token position to inspect. -1 means final position.")
    parser.add_argument("--target-token-id", type=int, default=-1, help="Token id to inspect. -1 means predicted token.")
    args = parser.parse_args(argv)

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
    token_ids = tokenizer.encode(args.prompt)
    if not token_ids:
        raise ValueError("Prompt must not be empty.")

    x = torch.tensor([token_ids], dtype=torch.long, device=device)
    if x.shape[1] > model.config.block_size:
        x = x[:, -model.config.block_size :]

    with torch.no_grad():
        logits, component_logits, max_error = model.decompose_logits(x)

    position = args.position if args.position >= 0 else x.shape[1] - 1
    if position >= x.shape[1]:
        raise ValueError(f"position={position} out of bounds for sequence length {x.shape[1]}.")

    if args.target_token_id >= 0:
        target_token_id = args.target_token_id
    else:
        target_token_id = int(logits[0, position].argmax().item())

    total_logit = float(logits[0, position, target_token_id].item())
    target_text = tokenizer.decode([target_token_id])

    print(f"Prompt: {args.prompt!r}")
    print(f"Position: {position}")
    print(f"Target token id: {target_token_id}")
    print(f"Target token text: {target_text!r}")
    print(f"Total logit: {total_logit:+.6f}")
    print()
    print("Attention head contributions:")

    head_sum = 0.0
    for name, component_logit in component_logits:
        if ".head_" not in name:
            continue
        value = float(component_logit[0, position, target_token_id].item())
        head_sum += value
        print(f"{name:20s} {value:+.6f}")

    print()
    print("Other component contributions:")
    other_sum = 0.0
    for name, component_logit in component_logits:
        if ".head_" in name:
            continue
        value = float(component_logit[0, position, target_token_id].item())
        other_sum += value
        print(f"{name:20s} {value:+.6f}")

    reconstructed = head_sum + other_sum
    print()
    print(f"Sum of heads:         {head_sum:+.6f}")
    print(f"Sum of other terms:   {other_sum:+.6f}")
    print(f"Reconstructed total:  {reconstructed:+.6f}")
    print(f"Actual total logit:   {total_logit:+.6f}")
    print(f"Max reconstruction error over all logits: {max_error:.8f}")


if __name__ == "__main__":
    main()
