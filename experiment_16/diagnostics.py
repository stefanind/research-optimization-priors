#!/usr/bin/env python3
from __future__ import annotations

"""
Self-contained teacher block norm diagnostics.

Requires only:
  - this script
  - final_model.pt
  - PyTorch

It does NOT import your training script.
It does NOT require dataset shards or tokenizer files.
By default, it creates random token IDs and runs the trained teacher on them.

Example:
  TEACHER_PATH=./final_model.pt \
  DIAG_LAYERS=6,7,8 \
  python teacher_block_norm_diagnostics_selfcontained.py

For your big teacher defaults:
  layers=12, dim=768, heads=12, kv_heads=4, mlp_mult=2, vocab=1024
"""

import csv
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# -----------------------------
# Config
# -----------------------------

class Config:
    teacher_path = os.environ.get("TEACHER_PATH", "./large_teacher.pt")
    out_csv = os.environ.get("OUT_CSV", "experiment_16/teacher_block_norm_diagnostics.csv")

    # Teacher architecture. These defaults match your 12-layer/768 teacher.
    vocab_size = int(os.environ.get("VOCAB_SIZE", "1024"))
    num_layers = int(os.environ.get("TEACHER_NUM_LAYERS", "12"))
    model_dim = int(os.environ.get("TEACHER_MODEL_DIM", "768"))
    num_heads = int(os.environ.get("TEACHER_NUM_HEADS", "12"))
    num_kv_heads = int(os.environ.get("TEACHER_NUM_KV_HEADS", "4"))
    mlp_mult = int(os.environ.get("TEACHER_MLP_MULT", "2"))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", "0.005"))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", "30.0"))
    rope_base = float(os.environ.get("ROPE_BASE", "10000.0"))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", "1.5"))

    # Diagnostic input. Random tokens are enough to inspect norm flow.
    seq_len = int(os.environ.get("SEQ_LEN", "1024"))
    batch_seqs = int(os.environ.get("DIAG_BATCH_SEQS", "2"))
    seed = int(os.environ.get("SEED", "1337"))

    # Which block-output layer indices to print. Layer index matches your training script:
    # layer 0 = after block 0, layer 11 = after block 11 for a 12-layer teacher.
    diag_layers = os.environ.get("DIAG_LAYERS", "0,1,2,3,4,5,6,7,8,9,10,11")

    # Device.
    device = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Model definition copied/self-contained
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)


# -----------------------------
# Diagnostic helpers
# -----------------------------

def mean_token_norm(x: Tensor) -> Tensor:
    return x.float().norm(dim=-1).mean()


def rms_state(x: Tensor) -> Tensor:
    return F.rms_norm(x.float(), (x.size(-1),))


def tensor_item(x: Tensor) -> float:
    return float(x.detach().float().cpu().item())


def mean_token_cos(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    """Mean cosine similarity between matching token vectors in a and b."""
    return F.cosine_similarity(a.float(), b.float(), dim=-1, eps=eps).mean()


def min_token_cos(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    """Minimum token-wise cosine similarity between matching token vectors."""
    return F.cosine_similarity(a.float(), b.float(), dim=-1, eps=eps).min()


def std_token_cos(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    """Std of token-wise cosine similarity between matching token vectors."""
    return F.cosine_similarity(a.float(), b.float(), dim=-1, eps=eps).std()


def frac_token_cos_below(a: Tensor, b: Tensor, threshold: float, eps: float = 1e-8) -> Tensor:
    """Fraction of tokens whose matching-vector cosine is below threshold."""
    cos = F.cosine_similarity(a.float(), b.float(), dim=-1, eps=eps)
    return (cos < threshold).float().mean()


def add_token_norm_distribution_stats(diag: dict[str, Tensor], prefix: str, x: Tensor) -> None:
    """
    Adds distribution stats for per-token vector magnitudes.

    x shape: [B, T, D]
    token_norms shape after norm: [B, T]

    These answer: are all tokens increasing together, or are a few tokens dominating?
    """
    norms = x.float().norm(dim=-1).flatten()
    total = norms.sum().clamp_min(1e-12)
    sorted_norms = torch.sort(norms, descending=True).values
    k1 = max(1, int(0.01 * sorted_norms.numel()))
    k5 = max(1, int(0.05 * sorted_norms.numel()))

    diag[f"{prefix}_toknorm_mean"] = norms.mean()
    diag[f"{prefix}_toknorm_std"] = norms.std()
    diag[f"{prefix}_toknorm_min"] = norms.min()
    diag[f"{prefix}_toknorm_p50"] = torch.quantile(norms, 0.50)
    diag[f"{prefix}_toknorm_p90"] = torch.quantile(norms, 0.90)
    diag[f"{prefix}_toknorm_p95"] = torch.quantile(norms, 0.95)
    diag[f"{prefix}_toknorm_p99"] = torch.quantile(norms, 0.99)
    diag[f"{prefix}_toknorm_max"] = norms.max()
    diag[f"{prefix}_toknorm_max_over_mean"] = norms.max() / norms.mean().clamp_min(1e-12)
    diag[f"{prefix}_toknorm_cv"] = norms.std() / norms.mean().clamp_min(1e-12)
    diag[f"{prefix}_toknorm_top1pct_share"] = sorted_norms[:k1].sum() / total
    diag[f"{prefix}_toknorm_top5pct_share"] = sorted_norms[:k5].sum() / total


def add_dim_energy_distribution_stats(diag: dict[str, Tensor], prefix: str, x: Tensor) -> None:
    """
    Adds distribution stats across hidden dimensions/features.

    x shape: [B, T, D]
    dim_energy shape: [D]

    These answer: is the magnitude spread across many hidden dimensions,
    or concentrated in a small number of dimensions/features?
    """
    x32 = x.float()
    dim_energy = x32.pow(2).mean(dim=(0, 1))  # [D]
    dim_abs = x32.abs().mean(dim=(0, 1))      # [D]

    total_energy = dim_energy.sum().clamp_min(1e-12)
    sorted_energy = torch.sort(dim_energy, descending=True).values
    k1 = max(1, int(0.01 * sorted_energy.numel()))
    k5 = max(1, int(0.05 * sorted_energy.numel()))
    k10 = max(1, int(0.10 * sorted_energy.numel()))

    diag[f"{prefix}_dim_energy_mean"] = dim_energy.mean()
    diag[f"{prefix}_dim_energy_std"] = dim_energy.std()
    diag[f"{prefix}_dim_energy_p50"] = torch.quantile(dim_energy, 0.50)
    diag[f"{prefix}_dim_energy_p90"] = torch.quantile(dim_energy, 0.90)
    diag[f"{prefix}_dim_energy_p95"] = torch.quantile(dim_energy, 0.95)
    diag[f"{prefix}_dim_energy_p99"] = torch.quantile(dim_energy, 0.99)
    diag[f"{prefix}_dim_energy_max"] = dim_energy.max()
    diag[f"{prefix}_dim_energy_max_over_mean"] = dim_energy.max() / dim_energy.mean().clamp_min(1e-12)
    diag[f"{prefix}_dim_energy_cv"] = dim_energy.std() / dim_energy.mean().clamp_min(1e-12)
    diag[f"{prefix}_dim_energy_top1pct_share"] = sorted_energy[:k1].sum() / total_energy
    diag[f"{prefix}_dim_energy_top5pct_share"] = sorted_energy[:k5].sum() / total_energy
    diag[f"{prefix}_dim_energy_top10pct_share"] = sorted_energy[:k10].sum() / total_energy

    diag[f"{prefix}_dim_abs_mean"] = dim_abs.mean()
    diag[f"{prefix}_dim_abs_std"] = dim_abs.std()
    diag[f"{prefix}_dim_abs_max"] = dim_abs.max()
    diag[f"{prefix}_dim_abs_max_over_mean"] = dim_abs.max() / dim_abs.mean().clamp_min(1e-12)


def add_before_after_dim_change_stats(diag: dict[str, Tensor], prefix: str, before: Tensor, after: Tensor) -> None:
    """
    Compares hidden-dimension energy before vs after a component/block.

    These answer: did the magnitude increase broadly across dimensions,
    or did a few dimensions/features become much larger?
    """
    b = before.float()
    a = after.float()
    b_energy = b.pow(2).mean(dim=(0, 1))
    a_energy = a.pow(2).mean(dim=(0, 1))
    delta_energy = a_energy - b_energy
    abs_delta_energy = delta_energy.abs()
    pos_delta_energy = delta_energy.clamp_min(0.0)

    ratio = a_energy / b_energy.clamp_min(1e-12)
    total_abs_delta = abs_delta_energy.sum().clamp_min(1e-12)
    total_pos_delta = pos_delta_energy.sum().clamp_min(1e-12)
    sorted_abs_delta = torch.sort(abs_delta_energy, descending=True).values
    sorted_pos_delta = torch.sort(pos_delta_energy, descending=True).values
    k1 = max(1, int(0.01 * sorted_abs_delta.numel()))
    k5 = max(1, int(0.05 * sorted_abs_delta.numel()))
    k10 = max(1, int(0.10 * sorted_abs_delta.numel()))

    diag[f"{prefix}_dim_energy_ratio_mean"] = ratio.mean()
    diag[f"{prefix}_dim_energy_ratio_p50"] = torch.quantile(ratio, 0.50)
    diag[f"{prefix}_dim_energy_ratio_p90"] = torch.quantile(ratio, 0.90)
    diag[f"{prefix}_dim_energy_ratio_p95"] = torch.quantile(ratio, 0.95)
    diag[f"{prefix}_dim_energy_ratio_p99"] = torch.quantile(ratio, 0.99)
    diag[f"{prefix}_dim_energy_ratio_max"] = ratio.max()
    diag[f"{prefix}_dim_energy_ratio_max_over_mean"] = ratio.max() / ratio.mean().clamp_min(1e-12)

    diag[f"{prefix}_dim_abs_delta_energy_mean"] = abs_delta_energy.mean()
    diag[f"{prefix}_dim_abs_delta_energy_p50"] = torch.quantile(abs_delta_energy, 0.50)
    diag[f"{prefix}_dim_abs_delta_energy_p90"] = torch.quantile(abs_delta_energy, 0.90)
    diag[f"{prefix}_dim_abs_delta_energy_p99"] = torch.quantile(abs_delta_energy, 0.99)
    diag[f"{prefix}_dim_abs_delta_energy_max"] = abs_delta_energy.max()
    diag[f"{prefix}_dim_abs_delta_energy_top1pct_share"] = sorted_abs_delta[:k1].sum() / total_abs_delta
    diag[f"{prefix}_dim_abs_delta_energy_top5pct_share"] = sorted_abs_delta[:k5].sum() / total_abs_delta
    diag[f"{prefix}_dim_abs_delta_energy_top10pct_share"] = sorted_abs_delta[:k10].sum() / total_abs_delta

    diag[f"{prefix}_dim_pos_delta_energy_top1pct_share"] = sorted_pos_delta[:k1].sum() / total_pos_delta
    diag[f"{prefix}_dim_pos_delta_energy_top5pct_share"] = sorted_pos_delta[:k5].sum() / total_pos_delta
    diag[f"{prefix}_dim_pos_delta_energy_top10pct_share"] = sorted_pos_delta[:k10].sum() / total_pos_delta

    diag[f"{prefix}_frac_dims_energy_increased"] = (delta_energy > 0).float().mean()


@torch.no_grad()
def block_forward_with_diag(block: Block, x: Tensor, x0: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    diag: dict[str, Tensor] = {}
    x_in = x

    mix = block.resid_mix.to(dtype=x.dtype)
    x_mixed = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

    attn_in = block.attn_norm(x_mixed)
    attn_out = block.attn(attn_in)
    attn_update = block.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
    x_attn = x_mixed + attn_update

    mlp_in = block.mlp_norm(x_attn)
    mlp_out = block.mlp(mlp_in)
    mlp_update = block.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
    x_out = x_attn + mlp_update

    # Raw residual-stream norms.
    diag["x_in_norm"] = mean_token_norm(x_in)
    diag["x_mixed_norm"] = mean_token_norm(x_mixed)
    diag["mix_delta_norm"] = mean_token_norm(x_mixed - x_in)
    diag["attn_in_norm"] = mean_token_norm(attn_in)
    diag["attn_out_norm"] = mean_token_norm(attn_out)
    diag["attn_update_norm"] = mean_token_norm(attn_update)
    diag["x_attn_norm"] = mean_token_norm(x_attn)
    diag["attn_delta_norm"] = mean_token_norm(x_attn - x_mixed)
    diag["mlp_in_norm"] = mean_token_norm(mlp_in)
    diag["mlp_out_norm"] = mean_token_norm(mlp_out)
    diag["mlp_update_norm"] = mean_token_norm(mlp_update)
    diag["x_out_norm"] = mean_token_norm(x_out)
    diag["mlp_delta_norm"] = mean_token_norm(x_out - x_attn)
    diag["block_delta_norm"] = mean_token_norm(x_out - x_in)

    # Per-token magnitude distribution diagnostics.
    # These tell you whether magnitude growth is broad across most tokens
    # or concentrated in a small number of token positions.
    add_token_norm_distribution_stats(diag, "x_in", x_in)
    add_token_norm_distribution_stats(diag, "x_mixed", x_mixed)
    add_token_norm_distribution_stats(diag, "attn_update", attn_update)
    add_token_norm_distribution_stats(diag, "x_attn", x_attn)
    add_token_norm_distribution_stats(diag, "mlp_update", mlp_update)
    add_token_norm_distribution_stats(diag, "x_out", x_out)
    add_token_norm_distribution_stats(diag, "block_delta", x_out - x_in)

    # Per-hidden-dimension / feature energy distribution diagnostics.
    # These tell you whether magnitude is spread across many dimensions
    # or concentrated in a small number of hidden dimensions/features.
    add_dim_energy_distribution_stats(diag, "x_in", x_in)
    add_dim_energy_distribution_stats(diag, "x_mixed", x_mixed)
    add_dim_energy_distribution_stats(diag, "attn_update", attn_update)
    add_dim_energy_distribution_stats(diag, "x_attn", x_attn)
    add_dim_energy_distribution_stats(diag, "mlp_update", mlp_update)
    add_dim_energy_distribution_stats(diag, "x_out", x_out)
    add_dim_energy_distribution_stats(diag, "block_delta", x_out - x_in)

    # Before/after per-dimension change diagnostics.
    # These answer whether each component broadly increases energy across dimensions
    # or sharply increases only a small subset of dimensions.
    add_before_after_dim_change_stats(diag, "mix", x_in, x_mixed)
    add_before_after_dim_change_stats(diag, "attn", x_mixed, x_attn)
    add_before_after_dim_change_stats(diag, "mlp", x_attn, x_out)
    add_before_after_dim_change_stats(diag, "block", x_in, x_out)

    # RMS-normalized state movement diagnostics.
    diag["normed_block_delta_norm"] = mean_token_norm(rms_state(x_out) - rms_state(x_in))
    diag["normed_attn_delta_norm"] = mean_token_norm(rms_state(x_attn) - rms_state(x_mixed))
    diag["normed_mlp_delta_norm"] = mean_token_norm(rms_state(x_out) - rms_state(x_attn))

    # Direction-change diagnostics.
    # These answer: did the token vector mostly keep pointing the same way,
    # or did the block/component rotate it into a different direction?
    diag["cos_x_in_to_x_mixed_mean"] = mean_token_cos(x_in, x_mixed)
    diag["cos_x_mixed_to_x_attn_mean"] = mean_token_cos(x_mixed, x_attn)
    diag["cos_x_attn_to_x_out_mean"] = mean_token_cos(x_attn, x_out)
    diag["cos_x_in_to_x_out_mean"] = mean_token_cos(x_in, x_out)

    diag["cos_x_in_to_x_out_std"] = std_token_cos(x_in, x_out)
    diag["cos_x_in_to_x_out_min"] = min_token_cos(x_in, x_out)
    diag["frac_tokens_block_cos_lt_0_5"] = frac_token_cos_below(x_in, x_out, 0.5)
    diag["frac_tokens_block_cos_lt_0"] = frac_token_cos_below(x_in, x_out, 0.0)

    # Update alignment diagnostics.
    # Positive means the update points along the existing residual direction.
    # Near zero means it is mostly orthogonal.
    # Negative means it pushes against the current residual direction.
    diag["cos_attn_update_to_x_mixed_mean"] = mean_token_cos(attn_update, x_mixed)
    diag["cos_mlp_update_to_x_attn_mean"] = mean_token_cos(mlp_update, x_attn)
    diag["cos_block_delta_to_x_in_mean"] = mean_token_cos(x_out - x_in, x_in)

    # Scale/control parameters.
    diag["attn_scale_abs_mean"] = block.attn_scale.float().abs().mean()
    diag["attn_scale_abs_max"] = block.attn_scale.float().abs().max()
    diag["mlp_scale_abs_mean"] = block.mlp_scale.float().abs().mean()
    diag["mlp_scale_abs_max"] = block.mlp_scale.float().abs().max()
    diag["resid_mix0_abs_mean"] = block.resid_mix[0].float().abs().mean()
    diag["resid_mix0_abs_max"] = block.resid_mix[0].float().abs().max()
    diag["resid_mix1_abs_mean"] = block.resid_mix[1].float().abs().mean()
    diag["resid_mix1_abs_max"] = block.resid_mix[1].float().abs().max()

    return x_out, diag


@torch.no_grad()
def forward_block_diagnostics(model: GPT, input_ids: Tensor, wanted_layers: set[int]) -> dict[int, dict[str, Tensor]]:
    out: dict[int, dict[str, Tensor]] = {}

    x = model.tok_emb(input_ids)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    skips: list[Tensor] = []
    layer_idx = 0

    for i in range(model.num_encoder_layers):
        x, diag = block_forward_with_diag(model.blocks[i], x, x0)
        if layer_idx in wanted_layers:
            out[layer_idx] = diag
        skips.append(x)
        layer_idx += 1

    for i in range(model.num_decoder_layers):
        skip_diag: dict[str, Tensor] = {}
        if skips:
            x_before_skip = x
            skip = skips.pop()
            skip_update = model.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skip
            x = x + skip_update
            skip_diag = {
                "pre_skip_norm": mean_token_norm(x_before_skip),
                "skip_source_norm": mean_token_norm(skip),
                "skip_update_norm": mean_token_norm(skip_update),
                "post_skip_norm": mean_token_norm(x),
                "skip_delta_norm": mean_token_norm(x - x_before_skip),
                "normed_skip_delta_norm": mean_token_norm(rms_state(x) - rms_state(x_before_skip)),
                "skip_weight_abs_mean": model.skip_weights[i].float().abs().mean(),
                "skip_weight_abs_max": model.skip_weights[i].float().abs().max(),
            }

        block_id = model.num_encoder_layers + i
        x, diag = block_forward_with_diag(model.blocks[block_id], x, x0)
        if layer_idx in wanted_layers:
            out[layer_idx] = {**skip_diag, **diag}
        layer_idx += 1

    return out


def make_random_tokens(cfg: Config, device: torch.device) -> Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(cfg.seed)
    x = torch.randint(
        low=0,
        high=cfg.vocab_size,
        size=(cfg.batch_seqs, cfg.seq_len),
        generator=g,
        dtype=torch.int64,
    )
    return x.to(device)


def main() -> None:
    cfg = Config()
    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    wanted_layers = {int(x) for x in cfg.diag_layers.split(",") if x.strip()}

    model = GPT(
        vocab_size=cfg.vocab_size,
        num_layers=cfg.num_layers,
        model_dim=cfg.model_dim,
        num_heads=cfg.num_heads,
        num_kv_heads=cfg.num_kv_heads,
        mlp_mult=cfg.mlp_mult,
        tie_embeddings=cfg.tie_embeddings,
        tied_embed_init_std=cfg.tied_embed_init_std,
        logit_softcap=cfg.logit_softcap,
        rope_base=cfg.rope_base,
        qk_gain_init=cfg.qk_gain_init,
    ).to(device).bfloat16()

    teacher_path = Path(cfg.teacher_path)
    if not teacher_path.exists():
        raise FileNotFoundError(f"Could not find TEACHER_PATH={teacher_path}")

    state = torch.load(str(teacher_path), map_location=device)
    model.load_state_dict(state, strict=True)

    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)
    model.eval()

    x = make_random_tokens(cfg, device)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            diags = forward_block_diagnostics(model, x, wanted_layers)

    if not diags:
        raise RuntimeError(f"No diagnostics produced. Check DIAG_LAYERS={wanted_layers} for num_layers={cfg.num_layers}")

    metric_names = sorted({k for d in diags.values() for k in d.keys()})
    rows: list[dict[str, float | int | str]] = []
    for layer in sorted(diags):
        row: dict[str, float | int | str] = {"layer": layer}
        for name in metric_names:
            val = diags[layer].get(name)
            row[name] = "" if val is None else tensor_item(val)
        rows.append(row)

    with open(cfg.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer"] + metric_names)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote:{cfg.out_csv}")
    print(f"teacher_path:{teacher_path}")
    print(f"input: random tokens batch_seqs={cfg.batch_seqs} seq_len={cfg.seq_len} vocab_size={cfg.vocab_size} seed={cfg.seed}")
    print("")
    print("Quick table:")
    compact_cols = [
        "pre_skip_norm",
        "skip_update_norm",
        "post_skip_norm",
        "x_in_norm",
        "mix_delta_norm",
        "attn_update_norm",
        "mlp_update_norm",
        "x_out_norm",
        "block_delta_norm",
        "normed_block_delta_norm",
        "cos_x_in_to_x_mixed_mean",
        "cos_x_mixed_to_x_attn_mean",
        "cos_x_attn_to_x_out_mean",
        "cos_x_in_to_x_out_mean",
        "cos_block_delta_to_x_in_mean",
    ]
    print("\t".join(["layer"] + compact_cols))
    for row in rows:
        vals = [str(row["layer"])]
        for col in compact_cols:
            v = row.get(col, "")
            vals.append("" if v == "" else f"{float(v):.4e}")
        print("\t".join(vals))

    print("")
    print("Interpretation hints:")
    print("  skip_update_norm huge     -> decoder skip path is a major source")
    print("  mix_delta_norm huge       -> resid_mix is a major source")
    print("  attn_update_norm huge     -> attention update is a major source")
    print("  mlp_update_norm huge      -> MLP update is a major source")
    print("  x_in_norm already huge    -> norm grew in earlier layers")
    print("  normed_block_delta small while block_delta huge -> mostly raw scale movement")


if __name__ == "__main__":
    main()