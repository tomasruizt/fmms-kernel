"""Collect median TPOT from vllm bench sweep results and print a table.

Usage:
    python collect_results.py <results_dir>
    python collect_results.py <results_dir> --per-run <variant>

The first form prints the summary table (averaged across runs).
The second form prints per-run TPOT for a single variant.
"""

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

VARIANTS = [
    ("baseline", "Baseline"),
    ("fmms-triton", "FMMS Triton"),
    ("fmms-flashinfer", "FMMS FlashInfer"),
]


def latest_run(variant_dir: Path) -> Path:
    """Return the most recent timestamped subdirectory."""
    dirs = sorted(d for d in variant_dir.iterdir() if d.is_dir())
    if not dirs:
        raise FileNotFoundError(f"No runs found in {variant_dir}")
    return dirs[-1]


def read_summary(results_dir: Path, variant_key: str) -> tuple[pd.DataFrame, str]:
    variant_dir = results_dir / variant_key
    run_dir = latest_run(variant_dir)
    df = pd.read_csv(run_dir / "summary.csv")
    return df, run_dir.name


def hodges_lehmann_pct(baseline_vals, fmms_vals):
    """Hodges-Lehmann estimator: median of all pairwise speedup percentages."""
    pairs = [(f - b) / b * 100 for b, f in product(baseline_vals, fmms_vals)]
    return np.median(pairs)


def print_summary(results_dir: Path):
    raw_frames = {}
    num_runs = None
    for variant_key, display_name in VARIANTS:
        variant_dir = results_dir / variant_key
        if not variant_dir.exists():
            print(f"Warning: {variant_dir} not found, skipping")
            continue
        df, run_name = read_summary(results_dir, variant_key)
        print(f"{variant_key}: {run_name}")
        num_runs = df["run_number"].nunique()
        raw_frames[variant_key] = df

    # Median TPOT across all runs per concurrency level
    tpot_frames = {}
    for variant_key, display_name in VARIANTS:
        if variant_key not in raw_frames:
            continue
        df = raw_frames[variant_key]
        median_tpot = df.groupby("max_concurrency")["median_tpot_ms"].median().rename(display_name)
        tpot_frames[variant_key] = median_tpot

    tpot = pd.concat(tpot_frames.values(), axis=1)

    if "Baseline" not in tpot.columns:
        print("\nNo baseline found, printing available variants only.\n")
        print(tabulate(tpot, headers="keys", tablefmt="grid", floatfmt=".2f"))
        return

    baseline = tpot["Baseline"]

    print(f"\nMedian across {num_runs} runs. Speedup: Hodges-Lehmann estimator.\n")

    # Build result with TPOT and Speedup columns interleaved
    result = pd.DataFrame(index=tpot.index)
    result["Baseline"] = baseline
    for variant_key, display_name in VARIANTS[1:]:  # skip baseline
        if display_name not in tpot.columns:
            continue
        result[display_name] = tpot[display_name]
        # Hodges-Lehmann: median of all pairwise speedup percentages
        baseline_df = raw_frames["baseline"]
        fmms_df = raw_frames[variant_key]
        concurrencies = sorted(baseline_df["max_concurrency"].unique())
        hl_speedups = {}
        for conc in concurrencies:
            b_vals = baseline_df.loc[
                baseline_df["max_concurrency"] == conc, "median_tpot_ms"
            ].values
            f_vals = fmms_df.loc[fmms_df["max_concurrency"] == conc, "median_tpot_ms"].values
            hl_speedups[conc] = hodges_lehmann_pct(b_vals, f_vals)
        hl_series = pd.Series(hl_speedups, name=f"{display_name} Speedup")
        result[f"{display_name} Speedup"] = hl_series.map(lambda x: f"{x:+.1f}%")
    result.index.name = "Concurrency"

    print(tabulate(result, headers="keys", tablefmt="grid", floatfmt=".2f"))


def print_per_run(results_dir: Path, variant_key: str):
    display_name = dict(VARIANTS)[variant_key]
    df, run_name = read_summary(results_dir, variant_key)
    print(f"{variant_key}: {run_name}")

    pivoted = df.pivot(index="max_concurrency", columns="run_number", values="median_tpot_ms")
    pivoted.columns = [f"Run {c}" for c in pivoted.columns]
    pivoted.index.name = "Concurrency"
    print(f"\n{display_name} — Median TPOT (ms) per run\n")
    print(tabulate(pivoted, headers="keys", tablefmt="grid", floatfmt=".2f"))


def main():
    results_dir = Path(sys.argv[1])

    print_summary(results_dir)
    for variant_key, _ in VARIANTS:
        variant_dir = results_dir / variant_key
        if not variant_dir.exists():
            continue
        print()
        print_per_run(results_dir, variant_key)


if __name__ == "__main__":
    main()
