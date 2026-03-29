from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ContributionRow:
    component: str
    layer: int
    head: int
    contribution: float


@dataclass
class TopTokenRow:
    component: str
    layer: int
    head: int
    token_id: int
    token_text: str
    logit_boost: float


def _apply_final_ln_to_component(
    component: torch.Tensor, full_residual: torch.Tensor, ln_f: torch.nn.LayerNorm
) -> torch.Tensor:
    mean = full_residual.mean(dim=-1, keepdim=True)
    var = ((full_residual - mean) ** 2).mean(dim=-1, keepdim=True)
    scale = ln_f.weight.view(1, 1, -1) / torch.sqrt(var + ln_f.eps)
    centered = component - component.mean(dim=-1, keepdim=True)
    return centered * scale


def _decompose_gpt2_logits(
    model: AutoModelForCausalLM, input_ids: torch.Tensor
) -> tuple[torch.Tensor, list[tuple[str, torch.Tensor]], float]:
    transformer = model.transformer
    batch_size, seq_len = input_ids.shape
    n_embd = transformer.config.n_embd
    n_head = transformer.config.n_head
    head_dim = n_embd // n_head
    device = input_ids.device

    position_ids = torch.arange(0, seq_len, device=device).unsqueeze(0)
    tok_embed = transformer.wte(input_ids)
    pos_embed = transformer.wpe(position_ids)

    x = tok_embed + pos_embed
    residual_components: list[tuple[str, torch.Tensor]] = [
        ("tok_embed", tok_embed),
        ("pos_embed", pos_embed),
    ]

    for layer_idx, block in enumerate(transformer.h):
        ln1 = block.ln_1(x)
        qkv = block.attn.c_attn(ln1)
        q, k, v = qkv.split(n_embd, dim=2)

        q = q.view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)

        att_scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        ).view(1, 1, seq_len, seq_len)
        att_scores = att_scores.masked_fill(~causal_mask, torch.finfo(att_scores.dtype).min)
        att_probs = F.softmax(att_scores, dim=-1)

        head_outputs = att_probs @ v
        head_outputs_bt = head_outputs.transpose(1, 2)

        w_out = block.attn.c_proj.weight.view(n_embd, n_head, head_dim)
        per_head_projected = torch.einsum("bthd,ohd->btho", head_outputs_bt, w_out)

        for head_idx in range(n_head):
            residual_components.append(
                (f"layer_{layer_idx}.head_{head_idx}", per_head_projected[:, :, head_idx, :])
            )

        if block.attn.c_proj.bias is not None:
            residual_components.append(
                (
                    f"layer_{layer_idx}.attn_bias",
                    block.attn.c_proj.bias.view(1, 1, -1).expand(batch_size, seq_len, -1),
                )
            )

        attn_out = per_head_projected.sum(dim=2)
        if block.attn.c_proj.bias is not None:
            attn_out = attn_out + block.attn.c_proj.bias.view(1, 1, -1)
        x = x + attn_out

        mlp_out = block.mlp(block.ln_2(x))
        residual_components.append((f"layer_{layer_idx}.mlp", mlp_out))
        x = x + mlp_out

    full_residual = x
    ln_f = transformer.ln_f
    normalized = ln_f(full_residual)
    logits = model.lm_head(normalized)

    component_logits: list[tuple[str, torch.Tensor]] = []
    for name, component in residual_components:
        after_ln = _apply_final_ln_to_component(component, full_residual, ln_f)
        component_logits.append((name, model.lm_head(after_ln)))

    if ln_f.bias is not None:
        ln_bias = ln_f.bias.view(1, 1, -1).expand(batch_size, seq_len, -1)
        component_logits.append(("ln_f.bias", model.lm_head(ln_bias)))

    reconstructed = sum(piece for _, piece in component_logits)
    max_error = (reconstructed - logits).abs().max().item()
    return logits, component_logits, max_error


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute and log per-attention-head contribution to GPT-2 logits."
    )
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--target-token-id", type=int, default=-1)
    parser.add_argument("--top-n", type=int, default=5, help="Top boosted tokens to log for each head/MLP.")
    parser.add_argument("--out-dir", type=str, default="huggingface/head_logs")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        logits, component_logits, max_error = _decompose_gpt2_logits(model, input_ids)

    seq_len = input_ids.shape[1]
    position = args.position if args.position >= 0 else seq_len - 1
    if not (0 <= position < seq_len):
        raise ValueError(f"position={position} out of bounds for sequence length {seq_len}")

    if args.target_token_id >= 0:
        target_token_id = args.target_token_id
    else:
        target_token_id = int(logits[0, position].argmax().item())

    total_logit = float(logits[0, position, target_token_id].item())
    target_token_text = tokenizer.decode([target_token_id])

    rows: list[ContributionRow] = []
    top_rows: list[TopTokenRow] = []
    head_sum = 0.0
    other_sum = 0.0

    for name, piece in component_logits:
        value = float(piece[0, position, target_token_id].item())
        if ".head_" in name:
            layer_str, head_str = name.split(".")
            layer = int(layer_str.split("_")[1])
            head = int(head_str.split("_")[1])
            head_sum += value
            rows.append(
                ContributionRow(
                    component=name,
                    layer=layer,
                    head=head,
                    contribution=value,
                )
            )
        else:
            other_sum += value
            rows.append(
                ContributionRow(
                    component=name,
                    layer=-1,
                    head=-1,
                    contribution=value,
                )
            )

        if ".head_" in name or ".mlp" in name:
            if ".head_" in name:
                layer_str, head_str = name.split(".")
                layer = int(layer_str.split("_")[1])
                head = int(head_str.split("_")[1])
            else:
                layer_str, _ = name.split(".")
                layer = int(layer_str.split("_")[1])
                head = -1

            values, token_ids = torch.topk(piece[0, position], k=args.top_n)
            for token_id, logit_boost in zip(token_ids.tolist(), values.tolist()):
                top_rows.append(
                    TopTokenRow(
                        component=name,
                        layer=layer,
                        head=head,
                        token_id=int(token_id),
                        token_text=tokenizer.decode([int(token_id)]),
                        logit_boost=float(logit_boost),
                    )
                )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"head_contrib_{ts}"
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    top_tokens_csv_path = out_dir / f"{stem}_top_tokens.csv"

    payload = {
        "timestamp_utc": ts,
        "model_name": args.model_name,
        "prompt": args.prompt,
        "position": position,
        "target_token_id": target_token_id,
        "target_token_text": target_token_text,
        "total_logit": total_logit,
        "sum_of_heads": head_sum,
        "sum_of_other_terms": other_sum,
        "reconstructed_total": head_sum + other_sum,
        "max_reconstruction_error": max_error,
        "contributions": [asdict(r) for r in rows],
        "top_boosted_tokens_per_head_mlp": [asdict(r) for r in top_rows],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["component", "layer", "head", "contribution"])
        for r in rows:
            writer.writerow([r.component, r.layer, r.head, r.contribution])

    with open(top_tokens_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["component", "layer", "head", "token_id", "token_text", "logit_boost"])
        for r in top_rows:
            writer.writerow([r.component, r.layer, r.head, r.token_id, r.token_text, r.logit_boost])

    print(f"Prompt: {args.prompt!r}")
    print(f"Position: {position}")
    print(f"Target token id: {target_token_id}")
    print(f"Target token text: {target_token_text!r}")
    print(f"Total logit: {total_logit:+.6f}")
    print(f"Sum of heads: {head_sum:+.6f}")
    print(f"Sum of other terms: {other_sum:+.6f}")
    print(f"Reconstructed total: {head_sum + other_sum:+.6f}")
    print(f"Max reconstruction error: {max_error:.8f}")
    print()
    print(f"Top {args.top_n} boosted tokens per head/MLP:")
    grouped: dict[str, list[TopTokenRow]] = {}
    for row in top_rows:
        grouped.setdefault(row.component, []).append(row)
    for component in sorted(grouped.keys()):
        print(component)
        for row in grouped[component]:
            safe_text = row.token_text.replace("\n", "\\n")
            print(f"  token_id={row.token_id:5d} boost={row.logit_boost:+.6f} text={safe_text!r}")
    print(f"Saved JSON log: {json_path.resolve()}")
    print(f"Saved CSV log:  {csv_path.resolve()}")
    print(f"Saved top-token CSV log: {top_tokens_csv_path.resolve()}")


if __name__ == "__main__":
    main()
