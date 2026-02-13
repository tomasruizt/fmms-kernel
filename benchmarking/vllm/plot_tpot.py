"""Plot median TPOT vs concurrency for vLLM benchmark results.

Usage:
    python benchmarking/vllm/plot_tpot.py
"""

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FMMS_VARIANTS = [
    ("fmms-triton", "FMMS (Triton)"),
]

MODELS = [
    "Qwen3-1.7B",
    "gpt-oss-120b",
]

RESULTS_DIR = Path(__file__).parent
IMGS_DIR = RESULTS_DIR / "imgs"
IMGS_DIR.mkdir(parents=True, exist_ok=True)

# Concurrency 256 causes KV cache pressure (59% peak, up to 64 waiting reqs on
# Qwen3-1.7B), inflating TPOT due to scheduling delays rather than sampler cost.
# Truncate to 128 to keep the comparison fair.
MAX_CONCURRENCY = 128


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


def load_all_data() -> pd.DataFrame:
    all_variants = [("baseline", "Baseline (PyTorch compiled)")] + FMMS_VARIANTS
    frames = []
    for model in MODELS:
        model_dir = RESULTS_DIR / model
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


def plot_tpot(df: pd.DataFrame):
    all_variants = [("baseline", "Baseline (PyTorch compiled)")] + FMMS_VARIANTS
    concurrencies = sorted(df["max_concurrency"].unique())

    fig, axes = plt.subplots(1, len(MODELS), figsize=(14, 5), sharey=False)
    if len(MODELS) == 1:
        axes = [axes]

    for ax, model in zip(axes, MODELS):
        mdf = df[df["model"] == model]
        sns.lineplot(mdf, x="max_concurrency", y="median_tpot_ms", hue="variant", marker="o", ax=ax)

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
    out = IMGS_DIR / "tpot_vs_concurrency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


def plot_speedup(max_concurrency: int):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(14, 5), sharey=False)
    if len(MODELS) == 1:
        axes = [axes]

    for ax, model in zip(axes, MODELS):
        model_dir = RESULTS_DIR / model
        if not model_dir.exists():
            continue
        sdf = hodges_lehmann_speedups(model_dir, max_concurrency)
        if sdf.empty:
            continue
        concurrencies = sorted(sdf["max_concurrency"].unique())

        for variant in sdf["variant"].unique():
            vdf = sdf[sdf["variant"] == variant]
            medians = []
            for conc in concurrencies:
                vals = vdf.loc[vdf["max_concurrency"] == conc, "speedup_pct"].values
                medians.append(np.median(vals))

            ax.plot(concurrencies, medians, marker="o", zorder=3, label=variant)
            ax.scatter(
                vdf["max_concurrency"],
                vdf["speedup_pct"],
                alpha=0.35,
                s=20,
                zorder=2,
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
    out = IMGS_DIR / "speedup_vs_concurrency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


def main():
    sns.set_context("talk")
    df = load_all_data()
    df = df[df["max_concurrency"] <= MAX_CONCURRENCY]

    plot_tpot(df)
    plot_speedup(MAX_CONCURRENCY)


if __name__ == "__main__":
    main()
