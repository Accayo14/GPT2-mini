from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset

try:
    from .tokenizer import GPT2Tokenizer
except ImportError:
    from tokenizer import GPT2Tokenizer


def _choose_text_column(dataset: Dataset) -> str:
    for candidate in ("text", "content", "document"):
        if candidate in dataset.column_names:
            return candidate
    raise ValueError(f"No known text column found. Available columns: {dataset.column_names}")


def _tokenize_split(dataset: Dataset, text_column: str, num_proc: int) -> Dataset:
    tokenizer = GPT2Tokenizer()

    def process(example: dict[str, str]) -> dict[str, list[int] | int]:
        ids = tokenizer.encode(example[text_column], add_eot=True)
        return {"ids": ids, "len": len(ids)}

    tokenized = dataset.map(
        process,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc=f"Tokenizing {text_column}",
    )
    return tokenized


def _write_memmap(dataset: Dataset, out_path: Path, dtype: type[np.uint16]) -> int:
    arr_len = int(np.sum(dataset["len"], dtype=np.uint64))
    mmap = np.memmap(out_path, dtype=dtype, mode="w+", shape=(arr_len,))
    shard_count = min(1024, max(1, len(dataset)))
    write_index = 0
    for shard_idx in range(shard_count):
        shard = dataset.shard(num_shards=shard_count, index=shard_idx, contiguous=True).with_format("numpy")
        ids = np.concatenate(shard["ids"]) if len(shard) > 0 else np.array([], dtype=dtype)
        mmap[write_index : write_index + len(ids)] = ids
        write_index += len(ids)
    mmap.flush()
    return arr_len


def build_and_save_dataset(
    dataset_name: str,
    dataset_config: str | None,
    out_dir: Path,
    val_fraction: float,
    seed: int,
    num_proc: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_dataset(dataset_name, dataset_config)

    if "train" in raw and "validation" in raw:
        train_raw = raw["train"]
        val_raw = raw["validation"]
    elif "train" in raw:
        split = raw["train"].train_test_split(test_size=val_fraction, seed=seed, shuffle=True)
        train_raw = split["train"]
        val_raw = split["test"]
    else:
        raise ValueError("Dataset must include a train split.")

    text_column = _choose_text_column(train_raw)
    train_tok = _tokenize_split(train_raw, text_column=text_column, num_proc=num_proc)
    val_tok = _tokenize_split(val_raw, text_column=text_column, num_proc=num_proc)

    dtype = np.uint16
    train_tokens = _write_memmap(train_tok, out_dir / "train.bin", dtype=dtype)
    val_tokens = _write_memmap(val_tok, out_dir / "val.bin", dtype=dtype)

    metadata = {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "text_column": text_column,
        "tokenizer": "gpt2",
        "dtype": "uint16",
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "vocab_size": 50257,
        "seed": seed,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    resolved = out_dir.resolve()
    print(f"Saved dataset to: {resolved}")
    print(f"train.bin: {resolved / 'train.bin'}")
    print(f"val.bin:   {resolved / 'val.bin'}")
    print(f"meta.json: {resolved / 'meta.json'}")
    print(f"train tokens: {train_tokens:,}")
    print(f"val tokens:   {val_tokens:,}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare GPT-2 tokenized memmap dataset.")
    parser.add_argument("--dataset", type=str, default="openwebtext")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="data/openwebtext")
    parser.add_argument("--val-fraction", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-proc", type=int, default=8)
    args = parser.parse_args(argv)

    build_and_save_dataset(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        out_dir=Path(args.out_dir),
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_proc=args.num_proc,
    )


if __name__ == "__main__":
    main()
