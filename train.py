from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

try:
    from .model import GPT2, GPT2Config
except ImportError:
    from model import GPT2, GPT2Config


def _load_memmaps(data_dir: Path) -> tuple[np.memmap, np.memmap]:
    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Expected {train_path} and {val_path}. Run prepare_dataset.py first."
        )
    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")
    return train_data, val_data


def _get_batch(
    split: str,
    train_data: np.memmap,
    val_data: np.memmap,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else val_data
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in starts]
    )
    y = torch.stack(
        [torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in starts]
    )
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


@torch.no_grad()
def _estimate_loss(
    model: GPT2,
    train_data: np.memmap,
    val_data: np.memmap,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: torch.device,
    ctx: nullcontext,
) -> dict[str, float]:
    out: dict[str, float] = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb, yb = _get_batch(split, train_data, val_data, batch_size, block_size, device)
            with ctx:
                _, loss = model(xb, yb)
            if loss is None:
                raise RuntimeError("Loss should not be None during evaluation.")
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def _get_lr(iter_num: int, warmup_iters: int, lr_decay_iters: int, learning_rate: float, min_lr: float) -> float:
    if iter_num < warmup_iters:
        return learning_rate * (iter_num + 1) / (warmup_iters + 1)
    if iter_num > lr_decay_iters:
        return min_lr
    decay_ratio = (iter_num - warmup_iters) / max(1, (lr_decay_iters - warmup_iters))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def _save_checkpoint(
    out_dir: Path,
    model: GPT2,
    optimizer: torch.optim.Optimizer,
    iter_num: int,
    best_val_loss: float,
    model_args: dict[str, int | float | bool],
    train_args: dict[str, int | float | bool | str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ckpt.pt"
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "model_args": model_args,
        "train_args": train_args,
    }
    torch.save(checkpoint, ckpt_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train GPT-2 small on tokenized memmap data.")
    parser.add_argument("--data-dir", type=str, default="data/openwebtext")
    parser.add_argument("--out-dir", type=str, default="out-gpt2-small")
    parser.add_argument("--init-from", type=str, default="scratch", choices=["scratch", "resume", "gpt2"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--min-lr", type=float, default=6e-5)
    parser.add_argument("--max-iters", type=int, default=20000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-iters", type=int, default=1000)
    parser.add_argument("--lr-decay-iters", type=int, default=20000)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = device.type
    if device_type == "cpu":
        args.dtype = "float32"

    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16" and device_type == "cuda"))

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    train_data, val_data = _load_memmaps(data_dir)

    iter_num = 0
    best_val_loss = float("inf")

    model_args: dict[str, int | float | bool] = {
        "vocab_size": 50257,
        "block_size": args.block_size,
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "dropout": args.dropout,
        "bias": True,
    }

    if args.init_from == "scratch":
        model = GPT2(GPT2Config(**model_args))
    elif args.init_from == "gpt2":
        model = GPT2.from_pretrained("gpt2", override_args={"dropout": args.dropout})
        model.crop_block_size(args.block_size)
    else:
        ckpt_path = out_dir / "ckpt.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Cannot resume without checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model_args = checkpoint["model_args"]
        model = GPT2(GPT2Config(**model_args))
        model.load_state_dict(checkpoint["model"])
        iter_num = int(checkpoint["iter_num"])
        best_val_loss = float(checkpoint["best_val_loss"])

    model = model.to(device)
    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(args.beta1, args.beta2),
        device_type=device_type,
    )

    if args.init_from == "resume":
        checkpoint = torch.load(out_dir / "ckpt.pt", map_location="cpu")
        optimizer.load_state_dict(checkpoint["optimizer"])

    if args.compile:
        model = torch.compile(model)

    train_args = vars(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(train_args, f, indent=2)

    print(f"device: {device}")
    print(f"parameters: {model.get_num_params():,}")
    print(f"tokens/iter: {args.batch_size * args.block_size * args.gradient_accumulation_steps:,}")

    t0 = time.time()
    while iter_num < args.max_iters:
        lr = _get_lr(
            iter_num=iter_num,
            warmup_iters=args.warmup_iters,
            lr_decay_iters=args.lr_decay_iters,
            learning_rate=args.learning_rate,
            min_lr=args.min_lr,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        if iter_num % args.eval_interval == 0 or iter_num == args.max_iters - 1:
            losses = _estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                batch_size=args.batch_size,
                block_size=args.block_size,
                eval_iters=args.eval_iters,
                device=device,
                ctx=ctx,
            )
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.3e}")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                _save_checkpoint(
                    out_dir=out_dir,
                    model=model,
                    optimizer=optimizer,
                    iter_num=iter_num,
                    best_val_loss=best_val_loss,
                    model_args=model_args,
                    train_args=train_args,
                )

        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.gradient_accumulation_steps):
            xb, yb = _get_batch("train", train_data, val_data, args.batch_size, args.block_size, device)
            with ctx:
                _, loss = model(xb, yb)
            if loss is None:
                raise RuntimeError("Loss should not be None in training.")
            loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if iter_num % args.log_interval == 0:
            dt = time.time() - t0
            print(f"iter {iter_num:6d} | loss {loss.item() * args.gradient_accumulation_steps:.4f} | dt {dt * 1000:.2f} ms")
            t0 = time.time()

        iter_num += 1

    _save_checkpoint(
        out_dir=out_dir,
        model=model,
        optimizer=optimizer,
        iter_num=iter_num,
        best_val_loss=best_val_loss,
        model_args=model_args,
        train_args=train_args,
    )
    print(f"Training complete. Checkpoint saved to {out_dir / 'ckpt.pt'}")


if __name__ == "__main__":
    main()
