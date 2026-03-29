# Hugging Face GPT-2 Sampling

This folder is for testing Hugging Face GPT-2 directly (no manual training).

## Files

- `sample_hf.py`: run one prompt and save one JSON output
- `batch_sample_hf.py`: run many prompts from a JSON file and save one JSON per prompt
- `prompts.json`: example prompt list for batch sampling

## Install (in your own conda env)

```bash
pip install transformers torch
```

## Single Prompt

```bash
python huggingface/sample_hf.py \
  --prompt "Indian Institute of Technology Bombay is" \
  --out-dir huggingface/outputs \
  --temperature 0.7 \
  --top-k 40 \
  --max-new-tokens 120
```

## Batch Prompts

```bash
python huggingface/batch_sample_hf.py \
  --prompts-file huggingface/prompts.json \
  --out-dir huggingface/outputs \
  --temperature 0.7 \
  --top-k 40 \
  --max-new-tokens 120
```

Each output JSON contains:

- `input`
- `output`
- `continuation_only`
- `sampling_config`
- `model_name`
- timestamp metadata
