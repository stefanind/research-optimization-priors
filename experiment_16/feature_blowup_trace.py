#!/usr/bin/env python3
from __future__ import annotations

"""
Trace which hidden features blow up inside each token.

This is intentionally a small companion to diagnostics.py. It reuses the same
self-contained teacher definition, runs the teacher on random token IDs, and
writes the actual hidden-dimension indices with the largest positive energy
increase for each token.

Examples:
  python experiment_16/feature_blowup_trace.py

  BLOWUP_COMPONENTS=block,attn,mlp \
  BLOWUP_TOPK=7 \
  BLOWUP_MAX_TOKENS=128 \
  python experiment_16/feature_blowup_trace.py

Outputs by default:
  experiment_16/teacher_feature_blowup_tokens.csv
  experiment_16/teacher_feature_blowup_summary.csv
  experiment_16/teacher_feature_blowup_effective_tokens.csv
  experiment_16/teacher_feature_blowup_effective_summary.csv
"""

import csv
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from diagnostics import (
    CastedLinear,
    Config,
    GPT,
    restore_low_dim_params_to_fp32,
    make_random_tokens,
)


VALID_COMPONENTS = {"skip", "mix", "attn", "mlp", "block"}


class BlowupConfig(Config):
    token_csv = os.environ.get("BLOWUP_TOKEN_CSV", "experiment_16/teacher_feature_blowup_tokens.csv")
    summary_csv = os.environ.get("BLOWUP_SUMMARY_CSV", "experiment_16/teacher_feature_blowup_summary.csv")
    effective_token_csv = os.environ.get(
        "BLOWUP_EFFECTIVE_TOKEN_CSV",
        "experiment_16/teacher_feature_blowup_effective_tokens.csv",
    )
    effective_summary_csv = os.environ.get(
        "BLOWUP_EFFECTIVE_SUMMARY_CSV",
        "experiment_16/teacher_feature_blowup_effective_summary.csv",
    )
    components = os.environ.get("BLOWUP_COMPONENTS", "block")
    topk = int(os.environ.get("BLOWUP_TOPK", "7"))
    summary_topn = int(os.environ.get("BLOWUP_SUMMARY_TOPN", "32"))
    effective_thresholds = os.environ.get("BLOWUP_EFFECTIVE_THRESHOLDS", "0.5,0.8,0.9,0.95,0.99")
    token_stride = int(os.environ.get("BLOWUP_TOKEN_STRIDE", "1"))
    max_tokens = int(os.environ.get("BLOWUP_MAX_TOKENS", "0"))
    min_before_energy = float(os.environ.get("BLOWUP_MIN_BEFORE_ENERGY", "1e-12"))


def parse_layers(text: str) -> set[int]:
    return {int(x) for x in text.split(",") if x.strip()}


def parse_components(text: str) -> list[str]:
    components = [x.strip().lower() for x in text.split(",") if x.strip()]
    bad = [x for x in components if x not in VALID_COMPONENTS]
    if bad:
        raise ValueError(f"unknown BLOWUP_COMPONENTS={bad}; valid={sorted(VALID_COMPONENTS)}")
    return components


def parse_thresholds(text: str) -> list[float]:
    thresholds = [float(x) for x in text.split(",") if x.strip()]
    bad = [x for x in thresholds if x <= 0.0 or x > 1.0]
    if bad:
        raise ValueError(f"BLOWUP_EFFECTIVE_THRESHOLDS must be in (0, 1], got {bad}")
    return sorted(set(thresholds))


def threshold_label(threshold: float) -> str:
    pct = threshold * 100.0
    if abs(pct - round(pct)) < 1e-6:
        return f"{int(round(pct))}pct"
    return f"{pct:.2f}".rstrip("0").rstrip(".").replace(".", "p") + "pct"


def load_teacher(cfg: Config, device: torch.device) -> GPT:
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
    return model


@torch.no_grad()
def block_forward_with_traces(block, x: Tensor, x0: Tensor) -> tuple[Tensor, dict[str, tuple[Tensor, Tensor]]]:
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

    traces = {
        "mix": (x_in, x_mixed),
        "attn": (x_mixed, x_attn),
        "mlp": (x_attn, x_out),
        "block": (x_in, x_out),
    }
    return x_out, traces


def selected_flat_token_indices(num_tokens: int, cfg: BlowupConfig, device: torch.device) -> Tensor:
    indices = torch.arange(num_tokens, device=device)
    if cfg.token_stride > 1:
        indices = indices[:: cfg.token_stride]
    if cfg.max_tokens > 0:
        indices = indices[: cfg.max_tokens]
    return indices


def gather_rows_for_component(
    *,
    layer: int,
    component: str,
    before: Tensor,
    after: Tensor,
    flat_indices: Tensor,
    topk: int,
    summary_topn: int,
    effective_thresholds: list[float],
    min_before_energy: float,
) -> tuple[
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
]:
    bsz, seqlen, dim = before.shape
    k = min(topk, dim)

    before_flat = before.float().reshape(-1, dim).index_select(0, flat_indices)
    after_flat = after.float().reshape(-1, dim).index_select(0, flat_indices)
    before_energy = before_flat.pow(2)
    after_energy = after_flat.pow(2)
    pos_delta_energy = (after_energy - before_energy).clamp_min(0.0)

    raw_token_total = pos_delta_energy.sum(dim=-1)
    token_total = raw_token_total.clamp_min(1e-12)
    top_values, top_features = torch.topk(pos_delta_energy, k=k, dim=-1)
    top_before = before_flat.gather(1, top_features)
    top_after = after_flat.gather(1, top_features)
    top_before_energy = before_energy.gather(1, top_features)
    top_after_energy = after_energy.gather(1, top_features)
    top_ratio = top_after_energy / top_before_energy.clamp_min(min_before_energy)
    top_share = top_values / token_total[:, None]

    flat_indices_cpu = flat_indices.detach().cpu()
    top_values_cpu = top_values.detach().cpu()
    top_features_cpu = top_features.detach().cpu()
    top_before_cpu = top_before.detach().cpu()
    top_after_cpu = top_after.detach().cpu()
    top_before_energy_cpu = top_before_energy.detach().cpu()
    top_after_energy_cpu = top_after_energy.detach().cpu()
    top_ratio_cpu = top_ratio.detach().cpu()
    top_share_cpu = top_share.detach().cpu()
    token_total_cpu = token_total.detach().cpu()

    token_rows: list[dict[str, float | int | str]] = []
    for i, flat_idx in enumerate(flat_indices_cpu.tolist()):
        batch_index = flat_idx // seqlen
        token_index = flat_idx % seqlen
        for rank in range(k):
            token_rows.append(
                {
                    "layer": layer,
                    "component": component,
                    "batch_index": batch_index,
                    "token_index": token_index,
                    "flat_token_index": flat_idx,
                    "rank": rank + 1,
                    "feature_index": int(top_features_cpu[i, rank].item()),
                    "before_value": float(top_before_cpu[i, rank].item()),
                    "after_value": float(top_after_cpu[i, rank].item()),
                    "before_energy": float(top_before_energy_cpu[i, rank].item()),
                    "after_energy": float(top_after_energy_cpu[i, rank].item()),
                    "pos_delta_energy": float(top_values_cpu[i, rank].item()),
                    "energy_ratio": float(top_ratio_cpu[i, rank].item()),
                    "share_of_token_pos_delta_energy": float(top_share_cpu[i, rank].item()),
                    "token_total_pos_delta_energy": float(token_total_cpu[i].item()),
                }
            )

    sorted_token_energy = torch.sort(pos_delta_energy, dim=-1, descending=True).values
    cumulative_share = sorted_token_energy.cumsum(dim=-1) / token_total[:, None]
    threshold_counts: dict[str, Tensor] = {}
    for threshold in effective_thresholds:
        label = threshold_label(threshold)
        reached = cumulative_share >= threshold
        counts = reached.to(torch.int64).argmax(dim=-1) + 1
        threshold_counts[label] = torch.where(raw_token_total > 0.0, counts, torch.zeros_like(counts))

    top1_share = sorted_token_energy[:, 0] / token_total
    topk_share = sorted_token_energy[:, :k].sum(dim=-1) / token_total
    raw_token_total_cpu = raw_token_total.detach().cpu()
    top1_share_cpu = top1_share.detach().cpu()
    topk_share_cpu = topk_share.detach().cpu()
    threshold_counts_cpu = {label: counts.detach().cpu() for label, counts in threshold_counts.items()}

    effective_token_rows: list[dict[str, float | int | str]] = []
    for i, flat_idx in enumerate(flat_indices_cpu.tolist()):
        batch_index = flat_idx // seqlen
        token_index = flat_idx % seqlen
        row: dict[str, float | int | str] = {
            "layer": layer,
            "component": component,
            "batch_index": batch_index,
            "token_index": token_index,
            "flat_token_index": flat_idx,
            "total_pos_delta_energy": float(raw_token_total_cpu[i].item()),
            "top1_share": float(top1_share_cpu[i].item()),
            f"top{k}_share": float(topk_share_cpu[i].item()),
        }
        for label, counts in threshold_counts_cpu.items():
            row[f"dims_for_{label}"] = int(counts[i].item())
        effective_token_rows.append(row)

    feature_total = pos_delta_energy.sum(dim=0)
    total_delta = feature_total.sum().clamp_min(1e-12)
    summary_k = min(max(k, summary_topn), dim)
    summary_values, summary_features = torch.topk(feature_total, k=summary_k)
    topk_counts = torch.bincount(top_features.reshape(-1), minlength=dim)
    mean_before_energy = before_energy.mean(dim=0)
    mean_after_energy = after_energy.mean(dim=0)
    mean_ratio = (after_energy / before_energy.clamp_min(min_before_energy)).mean(dim=0)

    summary_rows: list[dict[str, float | int | str]] = []
    num_selected_tokens = int(flat_indices.numel())
    for rank, feature in enumerate(summary_features.detach().cpu().tolist(), start=1):
        value = float(summary_values[rank - 1].detach().cpu().item())
        count = int(topk_counts[feature].detach().cpu().item())
        summary_rows.append(
            {
                "layer": layer,
                "component": component,
                "rank": rank,
                "feature_index": feature,
                "total_pos_delta_energy": value,
                "share_total_pos_delta_energy": value / float(total_delta.detach().cpu().item()),
                "token_topk_count": count,
                "token_topk_frac": count / max(1, num_selected_tokens),
                "mean_before_energy": float(mean_before_energy[feature].detach().cpu().item()),
                "mean_after_energy": float(mean_after_energy[feature].detach().cpu().item()),
                "mean_energy_ratio": float(mean_ratio[feature].detach().cpu().item()),
                "selected_tokens": num_selected_tokens,
            }
        )

    effective_summary: dict[str, float | int | str] = {
        "layer": layer,
        "component": component,
        "selected_tokens": num_selected_tokens,
        "top1_share_mean": float(top1_share.mean().detach().cpu().item()),
        "top1_share_p50": float(torch.quantile(top1_share, 0.50).detach().cpu().item()),
        "top1_share_p90": float(torch.quantile(top1_share, 0.90).detach().cpu().item()),
        f"top{k}_share_mean": float(topk_share.mean().detach().cpu().item()),
        f"top{k}_share_p50": float(torch.quantile(topk_share, 0.50).detach().cpu().item()),
        f"top{k}_share_p90": float(torch.quantile(topk_share, 0.90).detach().cpu().item()),
        f"top{k}_share_min": float(topk_share.min().detach().cpu().item()),
    }
    for label, counts in threshold_counts.items():
        counts_float = counts.float()
        effective_summary[f"dims_for_{label}_mean"] = float(counts_float.mean().detach().cpu().item())
        effective_summary[f"dims_for_{label}_p50"] = float(torch.quantile(counts_float, 0.50).detach().cpu().item())
        effective_summary[f"dims_for_{label}_p90"] = float(torch.quantile(counts_float, 0.90).detach().cpu().item())
        effective_summary[f"dims_for_{label}_min"] = int(counts.min().detach().cpu().item())
        effective_summary[f"dims_for_{label}_max"] = int(counts.max().detach().cpu().item())

    return token_rows, summary_rows, effective_token_rows, [effective_summary]


@torch.no_grad()
def trace_teacher(model: GPT, input_ids: Tensor, cfg: BlowupConfig) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    wanted_layers = parse_layers(cfg.diag_layers)
    components = parse_components(cfg.components)
    effective_thresholds = parse_thresholds(cfg.effective_thresholds)
    token_rows: list[dict] = []
    summary_rows: list[dict] = []
    effective_token_rows: list[dict] = []
    effective_summary_rows: list[dict] = []

    x = model.tok_emb(input_ids)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    skips: list[Tensor] = []
    layer_idx = 0

    flat_indices = selected_flat_token_indices(
        num_tokens=input_ids.numel(),
        cfg=cfg,
        device=input_ids.device,
    )

    def add_component(layer: int, component: str, before: Tensor, after: Tensor) -> None:
        rows, summary, effective_rows, effective_summary = gather_rows_for_component(
            layer=layer,
            component=component,
            before=before,
            after=after,
            flat_indices=flat_indices,
            topk=cfg.topk,
            summary_topn=cfg.summary_topn,
            effective_thresholds=effective_thresholds,
            min_before_energy=cfg.min_before_energy,
        )
        token_rows.extend(rows)
        summary_rows.extend(summary)
        effective_token_rows.extend(effective_rows)
        effective_summary_rows.extend(effective_summary)

    for i in range(model.num_encoder_layers):
        x, traces = block_forward_with_traces(model.blocks[i], x, x0)
        if layer_idx in wanted_layers:
            for component in components:
                if component in traces:
                    before, after = traces[component]
                    add_component(layer_idx, component, before, after)
        skips.append(x)
        layer_idx += 1

    for i in range(model.num_decoder_layers):
        if skips:
            x_before_skip = x
            skip = skips.pop()
            skip_update = model.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skip
            x = x + skip_update
            if layer_idx in wanted_layers and "skip" in components:
                add_component(layer_idx, "skip", x_before_skip, x)

        block_id = model.num_encoder_layers + i
        x, traces = block_forward_with_traces(model.blocks[block_id], x, x0)
        if layer_idx in wanted_layers:
            for component in components:
                if component in traces:
                    before, after = traces[component]
                    add_component(layer_idx, component, before, after)
        layer_idx += 1

    return token_rows, summary_rows, effective_token_rows, effective_summary_rows


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"no rows to write for {path}")
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cfg = BlowupConfig()
    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = load_teacher(cfg, device)
    input_ids = make_random_tokens(cfg, device)

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            token_rows, summary_rows, effective_token_rows, effective_summary_rows = trace_teacher(model, input_ids, cfg)

    write_csv(cfg.token_csv, token_rows)
    write_csv(cfg.summary_csv, summary_rows)
    write_csv(cfg.effective_token_csv, effective_token_rows)
    write_csv(cfg.effective_summary_csv, effective_summary_rows)

    print(f"wrote token rows:   {cfg.token_csv} ({len(token_rows)} rows)")
    print(f"wrote summary rows: {cfg.summary_csv} ({len(summary_rows)} rows)")
    print(f"wrote effective token rows:   {cfg.effective_token_csv} ({len(effective_token_rows)} rows)")
    print(f"wrote effective summary rows: {cfg.effective_summary_csv} ({len(effective_summary_rows)} rows)")
    print(f"teacher_path: {cfg.teacher_path}")
    print(
        "input: random tokens "
        f"batch_seqs={cfg.batch_seqs} seq_len={cfg.seq_len} vocab_size={cfg.vocab_size} seed={cfg.seed}"
    )
    print(
        "trace: "
        f"layers={cfg.diag_layers} components={cfg.components} topk={cfg.topk} "
        f"token_stride={cfg.token_stride} max_tokens={cfg.max_tokens or 'all'} "
        f"effective_thresholds={cfg.effective_thresholds}"
    )
    print("")
    print("Interpretation:")
    print("  token CSV: for each token, rank 1..topk are the feature indices with the largest positive energy gain.")
    print("  summary CSV: features that recur across tokens/layers have high token_topk_frac and high energy share.")
    print("  effective CSVs: count how many features are needed to explain 50/80/90/95/99% of each token's gain.")


if __name__ == "__main__":
    main()
