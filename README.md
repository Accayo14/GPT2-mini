# GPT-2 Small From Scratch in PyTorch

This repository contains a full GPT-2 small implementation written manually in PyTorch, plus a runnable demo for training, generation, and residual-stream attribution. The project is designed to be both trainable and inspectable, especially at the level of attention heads and their contribution to final logits.

## What You Built

### Full GPT-2 small architecture

The core model matches GPT-2 small:

- `12` transformer blocks
- `12` attention heads
- `768` hidden size
- `1024` context length
- tied token embedding and output projection
- causal self-attention
- MLP hidden size `4 * n_embd`
- final layer normalization

With `vocab_size=50257`, the model has `124,439,808` trainable parameters.

### Trainable forward pass

The forward pass supports standard next-token training:

- token and positional embeddings are added
- the sequence is processed through all transformer blocks
- logits are produced for each time step
- if targets are provided, cross-entropy loss is computed

### Text generation

The model includes autoregressive generation with:

- temperature scaling
- optional top-k sampling
- rolling context cropped to `block_size`

### Residual-stream and head-level attribution

The model can decompose logits back into residual-stream components:

- token embeddings
- positional embeddings
- each attention head’s projected output (post-`W_O`)
- attention output biases
- each MLP output
- final layer norm bias

Each component is transformed through the final layer norm and unembedding, and the pieces are summed to reconstruct the total logits. The demo prints per-head contributions and the max reconstruction error to verify correctness.

## Repo Structure

- `model.py`: GPT-2 small model implementation
- `train_demo.py`: demo training loop, generation, decomposition, and checkpoint save
- `tokenizer.py`: lightweight character tokenizer used for the demo
- `GPT2.py`: entrypoint wrapper
- `__init__.py`: package exports
- `gpt2_small_demo.pt`: demo checkpoint produced by the training script

## How the Model Works

High-level flow:

1. Token IDs are embedded and combined with positional embeddings.
2. The sequence is passed through 12 transformer blocks.
3. Each block applies layer norm, attention, residual add, layer norm, MLP, and residual add.
4. A final layer norm is applied.
5. The output is projected to vocabulary logits using tied weights.

## How the Decomposition Works

During an analysis run, the model captures every residual-stream write:

- embedding contributions
- per-head attention contributions after projection
- attention output bias terms
- MLP outputs

Each contribution is mapped through the final layer norm in decomposed form, then projected into logit space. The sum of all component logits is compared to the actual logits to confirm the attribution matches the model’s output up to floating-point error.

## Run the Demo

From the workspace root:

```bash
conda run --no-capture-output -n pdiff python GPT2/train_demo.py
```

The demo:

- instantiates GPT-2 small with the full parameter count
- trains for a few steps on a tiny toy corpus
- generates text from a prompt
- prints per-head logit contributions
- checks the reconstruction error
- saves a checkpoint to `gpt2_small_demo.pt`

## Example Usage

```python
from GPT2 import GPT2, GPT2Config

model = GPT2(GPT2Config(vocab_size=50257))
logits, loss = model(input_ids, target_ids)
sample = model.generate(input_ids, max_new_tokens=50, temperature=0.8, top_k=20)
logits, components, max_error = model.decompose_logits(input_ids)
```

## Notes and Current Limitations

- The demo uses a character-level tokenizer for simplicity.
- The training corpus is intentionally tiny, so the model will overfit quickly.
- The saved checkpoint is a demo checkpoint, not a pretrained GPT-2 release.

If you want, we can add a GPT-2 BPE tokenizer and a real dataset loader next.
