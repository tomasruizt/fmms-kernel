"""Collect median TPOT from vllm bench sweep results and print a table.

Usage: python collect_results.py <results_dir>

Where <results_dir> contains baseline/, fmms-triton/, fmms-flashinfer/ subdirs.
"""

import sys
from pathlib import Path

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


def main():
    results_dir = Path(sys.argv[1])

    tpot_frames = {}
    runs_frames = {}
    for variant_key, display_name in VARIANTS:
        variant_dir = results_dir / variant_key
        if not variant_dir.exists():
            print(f"Warning: {variant_dir} not found, skipping")
            continue
        run_dir = latest_run(variant_dir)
        print(f"{variant_key}: {run_dir.name}")
        df = pd.read_csv(run_dir / "summary.csv")
        agg = df.groupby("max_concurrency").agg(
            tpot=("median_tpot_ms", "mean"),
            runs=("run_number", "count"),
        )
        tpot_frames[variant_key] = agg["tpot"].rename(display_name)
        runs_frames[variant_key] = agg["runs"].rename(variant_key)

    # Assert all variants have the same num_runs per concurrency level
    runs_df = pd.concat(runs_frames.values(), axis=1)
    assert runs_df.nunique(axis=1).eq(1).all(), f"Mismatched num_runs across variants:\n{runs_df}"

    tpot = pd.concat(tpot_frames.values(), axis=1)
    baseline = tpot["Baseline"]

    # Build result with TPOT and Speedup columns interleaved
    result = pd.DataFrame(index=tpot.index)
    result["Baseline"] = baseline
    for variant_key, display_name in VARIANTS[1:]:  # skip baseline
        if display_name not in tpot.columns:
            continue
        result[display_name] = tpot[display_name]
        pct = (tpot[display_name] - baseline) / baseline * 100
        result[f"{display_name} Speedup"] = pct.map(lambda x: f"{x:+.1f}%")
    result["runs"] = runs_df.iloc[:, 0]
    result.index.name = "Concurrency"
    print()

    print(tabulate(result, headers="keys", tablefmt="grid", floatfmt=".2f"))


if __name__ == "__main__":
    main()
