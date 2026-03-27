from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

try:
    from .model import GPT2, GPT2Config
    from .tokenizer import CharTokenizer
except ImportError:
    from model import GPT2, GPT2Config
    from tokenizer import CharTokenizer


TRAINING_TEXT = """
In the beginning the model knows nothing.
We train a full GPT-2 small architecture on a tiny toy corpus.
Because the dataset is tiny, the model will overfit quickly.
That is fine for a demo: we only want to see training, generation, and head attribution.
Transformers build predictions through the residual stream.
Each attention head writes a vector into the residual stream.
Those vectors combine with embeddings and MLP outputs before the final logits are formed.
""".strip()


def build_dataset(text: str, tokenizer: CharTokenizer, block_size: int) -> torch.Tensor:
    token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    if token_ids.numel() <= block_size:
        raise ValueError("Training text must be longer than block_size")
    return token_ids


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, data.size(0) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts]).to(device)
    return x, y


def train_demo() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CharTokenizer(TRAINING_TEXT)
    config = GPT2Config(
        vocab_size=50257,
        block_size=64,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=True,
    )
    model = GPT2(config).to(device)

    total_params = sum(param.numel() for param in model.parameters())
    print("Model config:", asdict(config))
    print(f"Trainable parameters: {total_params:,}")
    print(f"Active tokenizer vocab used in this demo: {tokenizer.vocab_size}")

    data = build_dataset(TRAINING_TEXT, tokenizer, config.block_size)
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=3e-4,
        device_type=device.type,
    )

    model.train()
    steps = 40
    batch_size = 4
    for step in range(steps):
        xb, yb = get_batch(data, batch_size, config.block_size, device)
        _, loss = model(xb, yb)
        if loss is None:
            raise RuntimeError("Expected training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == steps - 1:
            print(f"step={step:03d} loss={loss.item():.4f}")

    model.eval()
    prompt = "Transformers build "
    prompt_tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    generated = model.generate(prompt_tokens, max_new_tokens=60, temperature=0.8, top_k=20)
    generated_text = tokenizer.decode(generated[0].tolist())

    print()
    print("Prompt:")
    print(prompt)
    print()
    print("Generated text:")
    print(generated_text)

    print()
    print("Residual decomposition at the final prompt position:")
    logits, component_logits, max_error = model.decompose_logits(prompt_tokens)
    final_position = prompt_tokens.shape[1] - 1
    predicted_token = int(logits[0, final_position].argmax().item())
    predicted_char = tokenizer.decode([predicted_token])
    total_logit = float(logits[0, final_position, predicted_token].item())

    print(f"Predicted next token id: {predicted_token}")
    print(f"Predicted next token char: {predicted_char!r}")
    print(f"Total logit: {total_logit:+.6f}")
    print()
    print("Attention heads:")
    head_sum = 0.0
    for name, component_logit in component_logits:
        if ".head_" not in name:
            continue
        value = float(component_logit[0, final_position, predicted_token].item())
        head_sum += value
        print(f"{name:20s} {value:+.6f}")

    print()
    print("Other terms:")
    other_sum = 0.0
    for name, component_logit in component_logits:
        if ".head_" in name:
            continue
        value = float(component_logit[0, final_position, predicted_token].item())
        other_sum += value
        print(f"{name:20s} {value:+.6f}")

    print()
    print(f"Sum of heads:         {head_sum:+.6f}")
    print(f"Sum of other terms:   {other_sum:+.6f}")
    print(f"Reconstructed total:  {head_sum + other_sum:+.6f}")
    print(f"Actual total logit:   {total_logit:+.6f}")
    print(f"Max reconstruction error: {max_error:.8f}")

    checkpoint_path = Path("GPT2") / "gpt2_small_demo.pt"
    torch.save(
        {
            "config": asdict(config),
            "model_state_dict": model.state_dict(),
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
        },
        checkpoint_path,
    )
    print()
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    train_demo()
