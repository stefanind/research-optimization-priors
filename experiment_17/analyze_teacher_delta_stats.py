"""
Analyze teacher residual-write statistics produced by compute_teacher_delta_stats.py.

Outputs:
    layer_summary.csv
    dimension_summary.csv
    report.md
    plots/*.png  (if matplotlib is available)

Example:
    python scripts/analyze_teacher_delta_stats.py \
      --stats outputs/activation_audit/teacher_delta_stats.pt \
      --output-dir outputs/activation_audit
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


EPS = 1e-30


def safe_float(x: Tensor | float | int) -> float:
    if isinstance(x, Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def parse_exclude_dims(s: str) -> list[int]:
    """Parse comma-separated hidden dimensions to remove from energy metrics."""
    if not s:
        return []
    dims: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            dims.append(int(part))
    return sorted(set(dims))


def remove_dims(x: Tensor, exclude_dims: list[int]) -> Tensor:
    """Return x with excluded hidden dimensions removed."""
    if not exclude_dims:
        return x
    hidden_dim = x.numel()
    keep = torch.ones(hidden_dim, dtype=torch.bool, device=x.device)
    for dim in exclude_dims:
        if dim < 0 or dim >= hidden_dim:
            raise ValueError(f"exclude dim {dim} is outside hidden dim range [0, {hidden_dim - 1}]")
        keep[dim] = False
    return x[keep]


def energy_metrics(energy: Tensor) -> dict[str, float | int]:
    """Compute PR, entropy, and k90 from a nonnegative energy vector."""
    e = energy.detach().double().cpu().clamp_min(0.0)
    total = e.sum()
    if total <= 0:
        return {
            "total_energy": 0.0,
            "participation_ratio": 0.0,
            "energy_entropy": 0.0,
            "energy_entropy_norm": 0.0,
            "k90": 0,
            "top1_frac": 0.0,
            "top5_frac": 0.0,
            "top10_frac": 0.0,
        }

    p = e / total
    pr = (e.sum().square() / e.square().sum().clamp_min(EPS)).item()
    entropy = -(p[p > 0] * torch.log(p[p > 0])).sum().item()
    entropy_norm = entropy / math.log(max(int(e.numel()), 2))

    sorted_e, _ = torch.sort(e, descending=True)
    cum = torch.cumsum(sorted_e, dim=0) / total
    k90 = int(torch.searchsorted(cum, torch.tensor(0.9, dtype=cum.dtype)).item()) + 1

    def top_frac(k: int) -> float:
        k = min(k, int(e.numel()))
        return float(sorted_e[:k].sum().item() / total.item())

    return {
        "total_energy": float(total.item()),
        "participation_ratio": float(pr),
        "energy_entropy": float(entropy),
        "energy_entropy_norm": float(entropy_norm),
        "k90": k90,
        "top1_frac": top_frac(1),
        "top5_frac": top_frac(5),
        "top10_frac": top_frac(10),
    }


def analyze_component(component_name: str, component: dict[str, Any], top_k: int, exclude_dims: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mean = component["mean"].double().cpu()
    mean_sq = component["mean_sq"].double().cpu()
    var = component["var"].double().cpu().clamp_min(0.0)
    count = int(component["count"])

    if mean.ndim != 2:
        raise ValueError(f"Expected mean to be [items, hidden_dim], got {tuple(mean.shape)} for {component_name}")

    layer_rows: list[dict[str, Any]] = []
    dim_rows: list[dict[str, Any]] = []

    num_items, hidden_dim = mean.shape
    for layer_idx in range(num_items):
        mean_l = mean[layer_idx]
        raw_energy = mean_sq[layer_idx].clamp_min(0.0)
        centered_energy = var[layer_idx].clamp_min(0.0)
        constant_energy = mean_l.square()

        raw = energy_metrics(raw_energy)
        centered = energy_metrics(centered_energy)
        constant = energy_metrics(constant_energy)

        raw_no_excluded = energy_metrics(remove_dims(raw_energy, exclude_dims))
        centered_no_excluded = energy_metrics(remove_dims(centered_energy, exclude_dims))
        constant_no_excluded = energy_metrics(remove_dims(constant_energy, exclude_dims))

        total_raw = raw_energy.sum().clamp_min(EPS)
        total_centered = centered_energy.sum().clamp_min(EPS)
        total_constant = constant_energy.sum()

        raw_energy_no_excluded = remove_dims(raw_energy, exclude_dims)
        centered_energy_no_excluded = remove_dims(centered_energy, exclude_dims)
        constant_energy_no_excluded = remove_dims(constant_energy, exclude_dims)

        total_raw_no_excluded = raw_energy_no_excluded.sum().clamp_min(EPS)
        total_centered_no_excluded = centered_energy_no_excluded.sum().clamp_min(EPS)
        total_constant_no_excluded = constant_energy_no_excluded.sum().clamp_min(EPS)

        raw_energy_removed_frac = float(((total_raw - total_raw_no_excluded) / total_raw).item())
        centered_energy_removed_frac = float(((total_centered - total_centered_no_excluded) / total_centered).item())
        constant_energy_removed_frac = float(((total_constant - total_constant_no_excluded) / total_constant.clamp_min(EPS)).item())

        constant_energy_fraction = float((total_constant / total_raw).item())
        centered_energy_fraction = float((total_centered / total_raw).item())

        const_sorted, const_idx = torch.sort(constant_energy, descending=True)
        var_sorted, var_idx = torch.sort(centered_energy, descending=True)
        raw_sorted, raw_idx = torch.sort(raw_energy, descending=True)

        layer_rows.append(
            {
                "component": component_name,
                "layer": layer_idx,
                "tokens_seen": count,
                "hidden_dim": hidden_dim,
                "constant_energy_fraction": constant_energy_fraction,
                "centered_energy_fraction": centered_energy_fraction,
                "raw_total_energy": raw["total_energy"],
                "centered_total_energy": centered["total_energy"],
                "constant_total_energy": constant["total_energy"],
                "raw_pr": raw["participation_ratio"],
                "centered_pr": centered["participation_ratio"],
                "constant_pr": constant["participation_ratio"],
                "raw_entropy_norm": raw["energy_entropy_norm"],
                "centered_entropy_norm": centered["energy_entropy_norm"],
                "raw_k90": raw["k90"],
                "centered_k90": centered["k90"],
                "constant_k90": constant["k90"],
                "excluded_dims": ",".join(str(d) for d in exclude_dims),
                "raw_k90_no_excluded": raw_no_excluded["k90"],
                "centered_k90_no_excluded": centered_no_excluded["k90"],
                "constant_k90_no_excluded": constant_no_excluded["k90"],
                "raw_pr_no_excluded": raw_no_excluded["participation_ratio"],
                "centered_pr_no_excluded": centered_no_excluded["participation_ratio"],
                "constant_pr_no_excluded": constant_no_excluded["participation_ratio"],
                "raw_top1_frac_no_excluded": raw_no_excluded["top1_frac"],
                "centered_top1_frac_no_excluded": centered_no_excluded["top1_frac"],
                "constant_top1_frac_no_excluded": constant_no_excluded["top1_frac"],
                "raw_energy_removed_frac": raw_energy_removed_frac,
                "centered_energy_removed_frac": centered_energy_removed_frac,
                "constant_energy_removed_frac": constant_energy_removed_frac,
                "raw_k90_jump_no_excluded": raw_no_excluded["k90"] - raw["k90"],
                "centered_k90_jump_no_excluded": centered_no_excluded["k90"] - centered["k90"],
                "raw_top1_frac": raw["top1_frac"],
                "centered_top1_frac": centered["top1_frac"],
                "constant_top1_frac": constant["top1_frac"],
                "raw_top_dim": int(raw_idx[0].item()),
                "constant_top_dim": int(const_idx[0].item()),
                "variable_top_dim": int(var_idx[0].item()),
                "constant_top_dim_energy_frac_of_raw": float((const_sorted[0] / total_raw).item()),
                "variable_top_dim_energy_frac_of_centered": float((var_sorted[0] / total_centered).item()),
            }
        )

        # Dimension-level summaries. Keep top-K by three criteria.
        selected: dict[int, str] = {}
        for rank, idx in enumerate(const_idx[:top_k].tolist(), start=1):
            selected[int(idx)] = selected.get(int(idx), "") + f"constant_rank_{rank};"
        for rank, idx in enumerate(var_idx[:top_k].tolist(), start=1):
            selected[int(idx)] = selected.get(int(idx), "") + f"variable_rank_{rank};"
        for rank, idx in enumerate(raw_idx[:top_k].tolist(), start=1):
            selected[int(idx)] = selected.get(int(idx), "") + f"raw_rank_{rank};"

        for dim_idx, reason in sorted(selected.items()):
            raw_i = raw_energy[dim_idx].clamp_min(EPS)
            const_i = constant_energy[dim_idx]
            var_i = centered_energy[dim_idx]
            constant_ratio = float((const_i / raw_i).item())
            dim_rows.append(
                {
                    "component": component_name,
                    "layer": layer_idx,
                    "dim": dim_idx,
                    "is_excluded": dim_idx in set(exclude_dims),
                    "selected_because": reason.rstrip(";"),
                    "mean": float(mean_l[dim_idx].item()),
                    "mean_abs": float(abs(mean_l[dim_idx].item())),
                    "mean_sq": float(raw_energy[dim_idx].item()),
                    "var": float(centered_energy[dim_idx].item()),
                    "constant_energy": float(constant_energy[dim_idx].item()),
                    "constant_ratio": constant_ratio,
                    "raw_energy_frac": float((raw_energy[dim_idx] / total_raw).item()),
                    "centered_energy_frac": float((centered_energy[dim_idx] / total_centered).item()),
                    "constant_energy_frac_of_raw": float((constant_energy[dim_idx] / total_raw).item()),
                }
            )

    return layer_rows, dim_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_make_plots(output_dir: Path, layer_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable, skipping plots: {exc}")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    components = sorted(set(row["component"] for row in layer_rows))

    for metric, ylabel, filename in [
        ("constant_energy_fraction", "constant energy fraction", "constant_energy_fraction_by_layer.png"),
        ("raw_k90", "raw k90 dimension count", "raw_k90_by_layer.png"),
        ("centered_k90", "centered k90 dimension count", "centered_k90_by_layer.png"),
        ("raw_pr", "raw participation ratio", "raw_pr_by_layer.png"),
        ("centered_pr", "centered participation ratio", "centered_pr_by_layer.png"),
    ]:
        plt.figure()
        for component in components:
            rows = [row for row in layer_rows if row["component"] == component]
            if not rows:
                continue
            xs = [int(row["layer"]) for row in rows]
            ys = [float(row[metric]) for row in rows]
            plt.plot(xs, ys, marker="o", label=component)
        plt.xlabel("layer / skip index")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / filename, dpi=160)
        plt.close()

    # One combined raw-vs-centered k90 plot per component for quick inspection.
    for component in components:
        rows = [row for row in layer_rows if row["component"] == component]
        if not rows:
            continue
        xs = [int(row["layer"]) for row in rows]
        plt.figure()
        plt.plot(xs, [float(row["raw_k90"]) for row in rows], marker="o", label="raw k90")
        plt.plot(xs, [float(row["centered_k90"]) for row in rows], marker="o", label="centered k90")
        plt.xlabel("layer / skip index")
        plt.ylabel("k90 dimension count")
        plt.title(f"raw vs centered k90: {component}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"k90_raw_vs_centered_{component}.png", dpi=160)
        plt.close()


def make_report(path: Path, payload: dict[str, Any], layer_rows: list[dict[str, Any]]) -> None:
    metadata = payload.get("metadata", {})
    lines: list[str] = []
    lines.append("# Teacher Delta Activation Audit")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    for key in ["checkpoint", "data_path", "seq_len", "batch_tokens", "num_batches", "tokens_processed", "dtype"]:
        if key in metadata:
            lines.append(f"- **{key}**: `{metadata[key]}`")
    if "model_config" in metadata:
        lines.append("- **model_config**:")
        for key, value in metadata["model_config"].items():
            lines.append(f"  - `{key}`: `{value}`")
    lines.append("")

    lines.append("## How to read the main metrics")
    lines.append("")
    lines.append("- `constant_energy_fraction`: fraction of expected delta energy explained by the mean write.")
    lines.append("- `raw_k90`: number of dimensions needed to explain 90% of raw expected delta energy.")
    lines.append("- `centered_k90`: same, but after subtracting the mean write; this is the input-dependent part.")
    lines.append("- `raw_pr` / `centered_pr`: participation ratio of raw vs centered energy distributions.")
    lines.append("- `*_no_excluded`: the same metric after removing the requested excluded dimension(s), default dim 690.")
    lines.append("- `centered_energy_removed_frac`: how much input-dependent energy was carried by the excluded dimension(s).")
    lines.append("")

    lines.append("## Most suspicious constant/outlier rows")
    lines.append("")
    sorted_rows = sorted(layer_rows, key=lambda r: float(r["constant_energy_fraction"]), reverse=True)
    top_rows = sorted_rows[:20]
    lines.append("| component | layer | constant_frac | raw_k90 | centered_k90 | raw_k90_no690 | centered_k90_no690 | centered_removed_frac | constant_top_dim | raw_top1_frac |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in top_rows:
        lines.append(
            "| {component} | {layer} | {constant_energy_fraction:.6f} | {raw_k90} | {centered_k90} | "
            "{raw_k90_no_excluded} | {centered_k90_no_excluded} | {centered_energy_removed_frac:.6f} | "
            "{constant_top_dim} | {raw_top1_frac:.6f} |".format(**row)
        )
    lines.append("")

    lines.append("## Quick interpretation")
    lines.append("")
    lines.append("Use this decision rule:")
    lines.append("")
    lines.append("- If `raw_k90` is tiny but `centered_k90` becomes much larger, the raw signal was mostly a constant/outlier write.")
    lines.append("- If both `raw_k90` and `centered_k90` remain tiny, the input-dependent transformation itself is low-dimensional.")
    lines.append("- If `constant_energy_fraction` is high in `mlp_delta`, the MLP is likely producing the massive dimension.")
    lines.append("- If it is high in `skip_delta`, the skip pathway is importing/amplifying the outlier.")
    lines.append("- If it is high in `block_total_delta` but not `attn_delta` or `mlp_delta`, inspect `mix_delta` and skip behavior.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze teacher residual-write delta statistics.")
    parser.add_argument("--stats", required=True, help="Path to teacher_delta_stats.pt")
    parser.add_argument("--output-dir", default="outputs/activation_audit")
    parser.add_argument("--top-k-dims", type=int, default=10)
    parser.add_argument("--exclude-dims", default="690", help="Comma-separated hidden dimensions to remove from extra no_excluded metrics, e.g. '690'")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats_path = Path(args.stats)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exclude_dims = parse_exclude_dims(args.exclude_dims)

    payload = torch.load(stats_path, map_location="cpu")
    if "components" not in payload:
        raise ValueError(f"Stats file missing 'components': {stats_path}")

    all_layer_rows: list[dict[str, Any]] = []
    all_dim_rows: list[dict[str, Any]] = []
    for component_name, component in payload["components"].items():
        layer_rows, dim_rows = analyze_component(component_name, component, top_k=args.top_k_dims, exclude_dims=exclude_dims)
        all_layer_rows.extend(layer_rows)
        all_dim_rows.extend(dim_rows)

    layer_csv = output_dir / "layer_summary.csv"
    dim_csv = output_dir / "dimension_summary.csv"
    report_md = output_dir / "report.md"

    write_csv(layer_csv, all_layer_rows)
    write_csv(dim_csv, all_dim_rows)
    make_report(report_md, payload, all_layer_rows)
    if not args.no_plots:
        maybe_make_plots(output_dir, all_layer_rows)

    print(f"wrote: {layer_csv}")
    print(f"wrote: {dim_csv}")
    print(f"wrote: {report_md}")
    print(f"excluded dims for *_no_excluded metrics: {exclude_dims}")
    if not args.no_plots:
        print(f"wrote plots under: {output_dir / 'plots'}")

    # Console preview: top rows by constant-energy fraction.
    print("\nTop suspicious rows:")
    for row in sorted(all_layer_rows, key=lambda r: float(r["constant_energy_fraction"]), reverse=True)[:10]:
        print(
            f"{row['component']:>18} layer={row['layer']:>2} "
            f"const_frac={row['constant_energy_fraction']:.6f} "
            f"raw_k90={row['raw_k90']:>4} centered_k90={row['centered_k90']:>4} "
            f"raw_pr={row['raw_pr']:.2f} centered_pr={row['centered_pr']:.2f} "
            f"top_const_dim={row['constant_top_dim']}"
        )


if __name__ == "__main__":
    main()
