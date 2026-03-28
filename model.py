from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.flash = hasattr(F, "scaled_dot_product_attention")

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer(
            "bias_mask",
            mask.view(1, 1, config.block_size, config.block_size),
            persistent=False,
        )

    def forward(
        self, x: torch.Tensor, capture_per_head: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        batch_size, seq_len, _ = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        if capture_per_head or not self.flash:
            scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            scores = scores.masked_fill(
                self.bias_mask[:, :, :seq_len, :seq_len] == 0,
                torch.finfo(scores.dtype).min,
            )
            attn = F.softmax(scores, dim=-1)
            attn = self.attn_dropout(attn)
            head_outputs = attn @ v
        else:
            head_outputs = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
            attn = None

        merged = head_outputs.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_embd)
        out = self.resid_dropout(self.c_proj(merged))

        if not capture_per_head:
            return out, None

        w_out = self.c_proj.weight.view(self.n_embd, self.n_head, self.head_dim)
        per_head_projected = torch.einsum("bthd,ohd->btho", head_outputs.transpose(1, 2), w_out)
        cache = {
            "attn_probs": attn,
            "per_head_projected": per_head_projected,
            "out_proj_bias": self.c_proj.bias.view(1, 1, -1) if self.c_proj.bias is not None else None,
        }
        return out, cache


class MLP(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        hidden = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, elementwise_affine=True)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=True)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.ln_1(x), capture_per_head=False)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd, elementwise_affine=True),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer["wte"].weight

        self.apply(self._init_weights)
        self._init_residual_projections()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_residual_projections(self) -> None:
        scale = 0.02 / (2 * self.config.n_layer) ** 0.5
        for block in self.transformer["h"]:
            nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=scale)
            nn.init.normal_(block.mlp.c_proj.weight, mean=0.0, std=scale)

    def get_num_params(self) -> int:
        return sum(param.numel() for param in self.parameters())

    def crop_block_size(self, block_size: int) -> None:
        if block_size > self.config.block_size:
            raise ValueError("block_size must be <= current config.block_size")
        self.config.block_size = block_size
        self.transformer["wpe"].weight = nn.Parameter(self.transformer["wpe"].weight[:block_size])
        for block in self.transformer["h"]:
            block.attn.bias_mask = block.attn.bias_mask[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(
        cls, model_type: str = "gpt2", override_args: dict[str, float | int | bool] | None = None
    ) -> GPT2:
        try:
            from transformers import GPT2LMHeadModel
        except ImportError as exc:
            raise ImportError("transformers is required for from_pretrained()") from exc

        override_args = override_args or {}
        if model_type != "gpt2":
            raise ValueError("Only 'gpt2' (small, 124M) is supported in this repository.")

        config = GPT2Config(
            vocab_size=50257,
            block_size=1024,
            n_layer=12,
            n_head=12,
            n_embd=768,
            dropout=float(override_args.get("dropout", 0.0)),
            bias=True,
        )
        model = cls(config)

        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        hf_state = hf_model.state_dict()
        model_state = model.state_dict()

        ignore = {"attn.masked_bias", "attn.bias"}
        transposed = {
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        }

        for key in model_state.keys():
            if any(key.endswith(token) for token in ignore):
                continue
            if key not in hf_state:
                continue
            source = hf_state[key]
            target = model_state[key]
            if any(key.endswith(name) for name in transposed):
                if source.t().shape != target.shape:
                    raise ValueError(f"Shape mismatch for {key}: {source.t().shape} vs {target.shape}")
                with torch.no_grad():
                    target.copy_(source.t())
            else:
                if source.shape != target.shape:
                    raise ValueError(f"Shape mismatch for {key}: {source.shape} vs {target.shape}")
                with torch.no_grad():
                    target.copy_(source)
        return model

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError("Sequence length exceeds block size")

        positions = torch.arange(0, seq_len, device=idx.device)
        tok_emb = self.transformer["wte"](idx)
        pos_emb = self.transformer["wpe"](positions).unsqueeze(0)
        x = self.transformer["drop"](tok_emb + pos_emb)

        for block in self.transformer["h"]:
            x = block(x)

        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(batch_size * seq_len, self.config.vocab_size),
                targets.view(batch_size * seq_len),
                ignore_index=-1,
            )
        return logits, loss

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < top_values[:, [-1]], float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float] = (0.9, 0.95),
        device_type: str = "cpu",
    ) -> torch.optim.Optimizer:
        param_dict = {name: param for name, param in self.named_parameters() if param.requires_grad}
        decay_params = [param for _, param in param_dict.items() if param.dim() >= 2]
        nodecay_params = [param for _, param in param_dict.items() if param.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        use_fused = fused_available and device_type == "cuda"
        return torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            fused=use_fused,
        )

    def forward_with_residuals(
        self, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[str, torch.Tensor]]]:
        if self.training and self.config.dropout > 0:
            raise RuntimeError("Run decomposition in eval mode when dropout > 0 for deterministic attribution.")

        batch_size, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError("Sequence length exceeds block size")

        positions = torch.arange(0, seq_len, device=idx.device)
        tok_emb = self.transformer["wte"](idx)
        pos_emb = self.transformer["wpe"](positions).unsqueeze(0).expand(batch_size, -1, -1)
        x = self.transformer["drop"](tok_emb + pos_emb)

        residual_components: list[tuple[str, torch.Tensor]] = [
            ("tok_embed", tok_emb),
            ("pos_embed", pos_emb),
        ]

        for layer_idx, block in enumerate(self.transformer["h"]):
            attn_out, attn_cache = block.attn(block.ln_1(x), capture_per_head=True)
            if attn_cache is None:
                raise RuntimeError("Expected attention cache when capture_per_head=True")

            per_head = attn_cache["per_head_projected"]
            for head_idx in range(self.config.n_head):
                residual_components.append(
                    (f"layer_{layer_idx}.head_{head_idx}", per_head[:, :, head_idx, :])
                )

            if attn_cache["out_proj_bias"] is not None:
                residual_components.append(
                    (
                        f"layer_{layer_idx}.attn_bias",
                        attn_cache["out_proj_bias"].expand(batch_size, seq_len, -1),
                    )
                )

            x = x + attn_out
            mlp_out = block.mlp(block.ln_2(x))
            residual_components.append((f"layer_{layer_idx}.mlp", mlp_out))
            x = x + mlp_out

        final_residual = x
        normalized = self.transformer["ln_f"](final_residual)
        logits = self.lm_head(normalized)
        return logits, final_residual, residual_components

    def decompose_logits(
        self, idx: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[str, torch.Tensor]], float]:
        logits, final_residual, residual_components = self.forward_with_residuals(idx)
        ln_f = self.transformer["ln_f"]

        component_logits: list[tuple[str, torch.Tensor]] = []
        for name, component in residual_components:
            component_after_ln = apply_final_ln_to_component(component, final_residual, ln_f)
            component_logits.append((name, self.lm_head(component_after_ln)))

        if ln_f.bias is not None:
            ln_bias = ln_f.bias.view(1, 1, -1).expand(idx.shape[0], idx.shape[1], -1)
            component_logits.append(("ln_f.bias", self.lm_head(ln_bias)))

        reconstructed = sum(piece for _, piece in component_logits)
        max_error = (reconstructed - logits).abs().max().item()
        return logits, component_logits, max_error


def apply_final_ln_to_component(
    component: torch.Tensor, full_residual: torch.Tensor, ln: nn.LayerNorm
) -> torch.Tensor:
    mean = full_residual.mean(dim=-1, keepdim=True)
    var = ((full_residual - mean) ** 2).mean(dim=-1, keepdim=True)
    scale = ln.weight.view(1, 1, -1) / torch.sqrt(var + ln.eps)
    centered = component - component.mean(dim=-1, keepdim=True)
    return centered * scale
