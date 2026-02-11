from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pydantic_settings import BaseSettings


def plot_batch_scaling(bdf_long: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    sns.lineplot(bdf_long, x="n_hidden_states", y="time[ms]", hue="provider", marker="o", ax=ax1)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    unique_n_hidden = sorted(bdf_long["n_hidden_states"].unique())
    ax1.set_xticks(unique_n_hidden, labels=[int(x) for x in unique_n_hidden])
    ax1.grid(alpha=0.5)
    ax1.set_xlabel("Inference batch size")
    ax1.set_ylabel("Time (ms)")
    ax1.legend_.remove()

    sns.lineplot(bdf_long, x="n_hidden_states", y="samples/ms", hue="provider", marker="o", ax=ax2)
    ax2.set_xscale("log")
    ax2.set_xticks(unique_n_hidden, labels=[int(x) for x in unique_n_hidden])
    ax2.grid(alpha=0.5)
    ax2.set_xlabel("Inference batch size")
    ax2.set_ylabel("Samples/ms")

    ncol = 1
    bbox_to_anchor = (1.05, 1)
    sns.move_legend(ax2, "upper left", title="Method", bbox_to_anchor=bbox_to_anchor, ncol=ncol)

    fig.tight_layout()
    return ax1


def plot_relative_performance(
    bdf_rel_long: pd.DataFrame, ref_method: str, show_providers: list[str]
) -> None:
    plot_df = bdf_rel_long.query("provider in @show_providers")
    ax = sns.barplot(
        plot_df,
        x="n_hidden_states",
        y="relative-perf",
        hue="provider",
    )
    ax.grid(alpha=0.5, axis="y")
    ncol = 1  # min(len(show_providers), 2)
    sns.move_legend(ax, "upper center", title="Method", bbox_to_anchor=(0.5, 1.35), ncol=ncol)
    ax.set_xlabel("Inference batch size")
    ax.set_ylabel("Relative Performance")
    ax.set_xticks(ax.get_xticks(), labels=bdf_rel_long["n_hidden_states"].unique().astype(int))
    ax.figure.tight_layout()
    return ax


def assign_col_samples_per_ms(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(**{"samples/ms": lambda df: df["n_hidden_states"] / df["time[ms]"]})


def read_triton_bench_csv(path: Path) -> pd.DataFrame:
    """Read a Triton benchmark CSV, stripping the ' (Time (ms))' column suffix."""
    df = pd.read_csv(path)
    df.columns = [c.removesuffix(" (Time (ms))") for c in df.columns]
    return df


def plot_relative_performance_from_wide(
    bdf: pd.DataFrame,
    ref_method: str,
    ref_slug: str,
    show_providers: list[str],
    case: str,
    plot_folder: Path,
    csv_folder: Path,
):
    """Compute relative performance vs ref_method and save plot + CSV."""
    methods = [c for c in bdf.columns if c != ref_method and c != "n_hidden_states"]
    bdf_rel = bdf.copy()
    bdf_rel[[*methods, ref_method]] = bdf[[*methods, ref_method]].div(bdf[ref_method], axis=0)
    bdf_rel_long = bdf_rel.melt(
        id_vars=["n_hidden_states"], var_name="provider", value_name="relative-time"
    )
    bdf_rel_long["relative-perf"] = 1 / bdf_rel_long["relative-time"]
    bdf_rel_long.round(3).to_csv(
        csv_folder / f"relative-performance-vs-{ref_slug}-{case}.csv", index=False
    )
    ax = plot_relative_performance(bdf_rel_long, ref_method, show_providers)
    ax.figure.savefig(
        plot_folder / f"relative-performance-vs-{ref_slug}-{case}.png", dpi=300, bbox_inches="tight"
    )
    return ax


def create_and_triton_bench_plots(folder: Path):
    tgt_folder = folder / "custom-plots"
    tgt_folder.mkdir(parents=True, exist_ok=True)

    csv_prefix = "fused-mm-sample-batch-scaling-"
    for csv_path in sorted(folder.glob(f"{csv_prefix}*.csv")):
        case = csv_path.stem.removeprefix(csv_prefix)
        print(f"Plotting case: {case}")

        bdf = read_triton_bench_csv(csv_path)
        bdf_long = bdf.melt(id_vars=["n_hidden_states"], var_name="provider", value_name="time[ms]")
        bdf_long = assign_col_samples_per_ms(bdf_long)

        ax = plot_batch_scaling(bdf_long)
        ax.figure.savefig(tgt_folder / f"batch-scaling-{case}.png", dpi=300)
        plt.close(ax.figure)

        FMMS = "FMMS (Triton)"  # noqa: N806
        NAIVE = "Naive PyTorch Compiled"  # noqa: N806
        FI_SAMPLE = "flashinfer:sampling_from_logits"  # noqa: N806
        FI_TOPK = "flashinfer:top_k_top_p_sampling_from_logits"  # noqa: N806

        rel_plots = [
            # (1) FMMS vs PyTorch Compiled (baseline)
            {"ref": NAIVE, "slug": "pytorch", "show": [FMMS, NAIVE]},
            # (2) FMMS vs both FlashInfer kernels (top_k_top_p as baseline)
            {"ref": FI_TOPK, "slug": "flashinfer", "show": [FMMS, FI_SAMPLE, FI_TOPK]},
        ]
        for rp in rel_plots:
            if rp["ref"] not in bdf.columns:
                continue
            show = [p for p in rp["show"] if p in bdf.columns]
            ax = plot_relative_performance_from_wide(
                bdf, rp["ref"], rp["slug"], show, case, tgt_folder, folder
            )
            plt.close(ax.figure)


class Args(BaseSettings, cli_parse_args=True):
    tgt_dir: Path = Path(__file__).parent / "profiles/triton-bench/"


if __name__ == "__main__":
    args = Args()
    create_and_triton_bench_plots(args.tgt_dir)
