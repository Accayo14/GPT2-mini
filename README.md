# GPT-2 Small (124M) Training + Interpretability Toolkit

This repository now contains a full GPT-2 small implementation with a practical training pipeline, GPT-2 BPE tokenization, real dataset preparation, generation, checkpointing, and residual-stream decomposition down to individual attention heads.

## What Is Implemented

- Full GPT-2 small architecture (`12` layers, `12` heads, `768` hidden, `1024` context, `50257` vocab)
- Correct GPT-2 style initialization, tied embeddings/unembedding
- Causal attention with PyTorch SDPA fast path and manual path for decomposition
- Trainable language modeling forward pass with next-token cross-entropy
- AdamW parameter grouping (`decay`/`no_decay`)
- Warmup + cosine LR scheduling
- Gradient accumulation, gradient clipping, AMP support (`bf16/fp16`)
- Checkpoint save and resume
- Optional initialization from Hugging Face pretrained `gpt2` weights
- GPT-2 BPE tokenizer using `tiktoken`
- Real dataset preparation into memory-mapped `train.bin` / `val.bin`
- Generation script for prompt completion
- Logit decomposition script printing per-head contributions and reconstruction error

## Repository Layout

- `model.py`: GPT-2 model and decomposition utilities
- `tokenizer.py`: GPT-2 BPE tokenizer wrapper (`tiktoken`)
- `prepare_dataset.py`: builds tokenized memmap dataset files
- `train.py`: full training loop
- `sample.py`: inference/generation from checkpoint
- `decompose.py`: residual-stream and per-head logit attribution
- `GPT2.py`: simple command router for `prepare/train/sample/decompose`
- `requirements.txt`: Python dependencies

## Install

```bash
pip install -r requirements.txt
```

## Proper Dataset Preparation

The recommended default dataset is `openwebtext`. This script:

1. Downloads the dataset via Hugging Face `datasets`
2. Tokenizes with GPT-2 BPE (`tiktoken`, EOT-delimited documents)
3. Writes `train.bin`, `val.bin`, and `meta.json`

Run from repo root (`GPT2`):

```bash
python prepare_dataset.py --dataset openwebtext --out-dir data/openwebtext --num-proc 8
```

Alternative example (if you want a smaller benchmark dataset first):

```bash
python prepare_dataset.py --dataset wikitext --dataset-config wikitext-103-raw-v1 --out-dir data/wikitext103 --num-proc 8
```

## Train GPT-2 Small

Scratch training:

```bash
python train.py \
  --data-dir data/openwebtext \
  --out-dir out-gpt2-small \
  --init-from scratch \
  --batch-size 8 \
  --gradient-accumulation-steps 8 \
  --block-size 1024 \
  --max-iters 20000 \
  --eval-interval 200 \
  --compile
```

Resume from checkpoint:

```bash
python train.py --data-dir data/openwebtext --out-dir out-gpt2-small --init-from resume
```

Start from pretrained OpenAI/HF GPT-2 small weights and continue training:

```bash
python train.py --data-dir data/openwebtext --out-dir out-gpt2-small --init-from gpt2
```

## Generate Text

```bash
python sample.py --out-dir out-gpt2-small --prompt "Once upon a time" --max-new-tokens 200 --temperature 0.8 --top-k 200
```

## Decompose Attention Head Contributions

This prints each head's contribution to the selected logit at a specific position and verifies reconstruction:

```bash
python decompose.py --out-dir out-gpt2-small --prompt "The meaning of life is"
```

You get:

- per-head logit contributions (`layer_i.head_j`)
- other residual terms (embeddings, MLPs, biases)
- summed reconstruction vs actual logit
- max reconstruction error across logits

## One-Entry Command Router

You can also use:

```bash
python GPT2.py prepare ...
python GPT2.py train ...
python GPT2.py sample ...
python GPT2.py decompose ...
```

## Notes

- `decompose.py` should be run in eval mode; decomposition is deterministic there.
- For real quality training, use a GPU with enough memory and tune `batch_size` + `gradient_accumulation_steps`.
- `openwebtext` download/tokenization can take significant time and disk.
