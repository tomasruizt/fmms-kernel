"""Plot median TPOT vs concurrency for vLLM benchmark results.

Usage:
    python benchmarking/vllm/plot_tpot.py [--results-dir <path>]
"""

import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FMMS_VARIANTS = [
    ("fmms-triton", "FMMS (Triton)"),
]

# Consistent color palette: FMMS stands out in bold red, baseline is gray.
VARIANT_COLORS: dict[str, str] = {
    "Baseline (PyTorch compiled)": "#7f7f7f",  # gray
    "FMMS (Triton)": "#d62728",  # bold red
}

VARIANT_MARKERS: dict[str, str] = {
    "Baseline (PyTorch compiled)": "s",  # square
    "FMMS (Triton)": "o",  # circle
}

MODELS = [
    "Qwen3-1.7B",
    "Qwen3-8B",
    "gpt-oss-120b",
]

MAX_CONCURRENCY = 256


def latest_run(variant_dir: Path) -> Path:
    dirs = sorted(d for d in variant_dir.iterdir() if d.is_dir())
    if not dirs:
        raise FileNotFoundError(f"No runs found in {variant_dir}")
    return dirs[-1]


def load_variant(model_dir: Path, variant_key: str) -> pd.DataFrame | None:
    variant_dir = model_dir / variant_key
    if not variant_dir.exists():
        print(f"Warning: {variant_dir} not found, skipping")
        return None
    run_dir = latest_run(variant_dir)
    return pd.read_csv(run_dir / "summary.csv")


def load_all_data(results_dir: Path) -> pd.DataFrame:
    all_variants = [("baseline", "Baseline (PyTorch compiled)")] + FMMS_VARIANTS
    frames = []
    for model in MODELS:
        model_dir = results_dir / model
        if not model_dir.exists():
            print(f"Warning: {model_dir} not found, skipping")
            continue
        for variant_key, display_name in all_variants:
            df = load_variant(model_dir, variant_key)
            if df is None:
                continue
            df["variant"] = display_name
            df["model"] = model
            frames.append(
                df[["model", "variant", "max_concurrency", "median_tpot_ms", "run_number"]]
            )
    return pd.concat(frames, ignore_index=True)


def hodges_lehmann_speedups(model_dir: Path, max_concurrency: int) -> pd.DataFrame:
    """Compute all pairwise speedup ratios between baseline and FMMS runs.

    For each concurrency level, pairs every baseline run with every FMMS run
    (3x3=9 pairs) and computes speedup % = (1 - fmms/baseline) * 100.
    The median of these pairs is the Hodges-Lehmann estimator.
    """
    baseline_df = load_variant(model_dir, "baseline")
    if baseline_df is None:
        return pd.DataFrame()

    rows = []
    for variant_key, display_name in FMMS_VARIANTS:
        fmms_df = load_variant(model_dir, variant_key)
        if fmms_df is None:
            continue
        concurrencies = sorted(baseline_df["max_concurrency"].unique())
        for conc in concurrencies:
            if conc > max_concurrency:
                continue
            b_vals = baseline_df.loc[
                baseline_df["max_concurrency"] == conc, "median_tpot_ms"
            ].values
            f_vals = fmms_df.loc[fmms_df["max_concurrency"] == conc, "median_tpot_ms"].values
            # All pairwise speedup ratios
            for b, f in product(b_vals, f_vals):
                rows.append(
                    {
                        "variant": display_name,
                        "max_concurrency": conc,
                        "speedup_pct": (1 - f / b) * 100,
                    }
                )
    return pd.DataFrame(rows)


def plot_tpot(df: pd.DataFrame, imgs_dir: Path):
    all_variants = [("baseline", "Baseline (PyTorch compiled)")] + FMMS_VARIANTS
    concurrencies = sorted(df["max_concurrency"].unique())

    fig, axes = plt.subplots(1, len(MODELS), figsize=(14, 5), sharey=False)
    if len(MODELS) == 1:
        axes = [axes]

    for ax, model in zip(axes, MODELS):
        mdf = df[df["model"] == model]
        palette = {v: VARIANT_COLORS[v] for v in mdf["variant"].unique()}
        markers = {v: VARIANT_MARKERS[v] for v in mdf["variant"].unique()}
        sns.lineplot(
            mdf,
            x="max_concurrency",
            y="median_tpot_ms",
            hue="variant",
            style="variant",
            markers=markers,
            dashes=False,
            ax=ax,
            palette=palette,
        )

        ax.set_title(model)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Median TPOT (ms)")
        ax.set_xscale("log")
        ax.set_xticks(concurrencies, labels=[int(x) for x in concurrencies], minor=False)
        ax.set_xticks([], minor=True)
        ax.grid(alpha=0.5)
        ax.legend_.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        title="Method",
        bbox_to_anchor=(0.5, 1.15),
        ncol=len(all_variants),
    )

    fig.tight_layout()
    out = imgs_dir / "tpot_vs_concurrency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


def plot_speedup(results_dir: Path, imgs_dir: Path, max_concurrency: int):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(14, 5), sharey=False)
    if len(MODELS) == 1:
        axes = [axes]

    for ax, model in zip(axes, MODELS):
        model_dir = results_dir / model
        if not model_dir.exists():
            continue
        sdf = hodges_lehmann_speedups(model_dir, max_concurrency)
        if sdf.empty:
            continue
        concurrencies = sorted(sdf["max_concurrency"].unique())

        for variant in sdf["variant"].unique():
            vdf = sdf[sdf["variant"] == variant]
            color = VARIANT_COLORS[variant]
            medians = []
            for conc in concurrencies:
                vals = vdf.loc[vdf["max_concurrency"] == conc, "speedup_pct"].values
                medians.append(np.median(vals))

            marker = VARIANT_MARKERS.get(variant, "o")
            ax.plot(concurrencies, medians, marker=marker, zorder=3, label=variant, color=color)
            ax.scatter(
                vdf["max_concurrency"],
                vdf["speedup_pct"],
                alpha=0.35,
                s=20,
                zorder=2,
                color=color,
            )

        ax.set_title(model)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Speedup (%)")
        ax.set_xscale("log")
        ax.set_xticks(concurrencies, labels=[int(x) for x in concurrencies], minor=False)
        ax.set_xticks([], minor=True)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.grid(alpha=0.5)
        if ax.get_legend():
            ax.legend_.remove()

    fig.tight_layout()
    out = imgs_dir / "speedup_vs_concurrency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot TPOT and speedup from vLLM benchmarks")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory containing model subdirectories (default: benchmarking/vllm/)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    imgs_dir = results_dir / "imgs"
    imgs_dir.mkdir(parents=True, exist_ok=True)

    sns.set_context("talk")
    df = load_all_data(results_dir)
    df = df[df["max_concurrency"] <= MAX_CONCURRENCY]

    plot_tpot(df, imgs_dir)
    plot_speedup(results_dir, imgs_dir, MAX_CONCURRENCY)


if __name__ == "__main__":
    main()
