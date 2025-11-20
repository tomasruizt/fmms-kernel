from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_batch_scaling(bdf_long: pd.DataFrame) -> None:
    ax = sns.lineplot(bdf_long, x="n_hidden_states", y="samples/ms", hue="provider", marker="o")
    ax.set_xscale("log")
    ax.set_xticks(bdf["n_hidden_states"], labels=bdf["n_hidden_states"].astype(int))
    ax.grid(alpha=0.5)
    ax.set_xlabel("Inference batch size")
    ax.set_ylabel("Samples/ms")
    sns.move_legend(ax, "upper center", title="Method", bbox_to_anchor=(0.5, 1.3), ncol=2)
    ax.figure.tight_layout()
    return ax


def plot_vocab_scaling(vdf_long: pd.DataFrame) -> None:
    ax = sns.lineplot(vdf_long, x="vocab_size", y="time[ms]", hue="provider", marker="o")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(40_000, 500_000)
    ax.xaxis.set_ticks(
        vdf_long["vocab_size"], labels=[f"{int(x / 1000)}k" for x in vdf_long["vocab_size"]]
    )
    ax.grid(alpha=0.5)
    sns.move_legend(ax, "upper center", title="Method", bbox_to_anchor=(0.5, 1.3), ncol=2)
    ax.figure.tight_layout()
    return ax


def plot_relative_performance(bdf_rel_long: pd.DataFrame) -> None:
    ax = sns.barplot(
        bdf_rel_long.query("provider == @ref_method or provider == 'Fused Triton'"),
        x="n_hidden_states",
        y="samples/ms",
        hue="provider",
    )
    ax.grid(alpha=0.5, axis="y")
    sns.move_legend(ax, "upper center", title="Method", bbox_to_anchor=(0.5, 1.3), ncol=2)
    ax.set_xlabel("Inference batch size")
    ax.set_ylabel("Relative Performance")
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([int(x) for x in ax.get_xticks()])
    ax.figure.tight_layout()
    return ax


if __name__ == "__main__":
    folder = Path("profiles/triton-bench/")
    tgt_folder = folder / "custom-plots"
    bdf = pd.read_csv(folder / "fused-mm-sample-batch-scaling.csv")
    bdf_long = bdf.melt(id_vars=["n_hidden_states"], var_name="provider", value_name="samples/ms")
    bdf_long = bdf_long.query("provider != 'JL Compiled'")

    ax = plot_batch_scaling(bdf_long)
    tgt_folder.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(tgt_folder / "batch-scaling.png", dpi=300)
    plt.close(ax.figure)

    vdf = pd.read_csv(folder / "fused-mm-sample-vocab-scaling.csv")
    vdf_long = vdf.melt(id_vars=["vocab_size"], var_name="provider", value_name="time[ms]")
    vdf_long = vdf_long.query("provider != 'JL Compiled'")
    ax = plot_vocab_scaling(vdf_long)
    ax.figure.savefig(tgt_folder / "vocab-scaling.png", dpi=300)
    plt.close(ax.figure)

    ref_method = "flashinfer:sampling_from_logits"
    methods = [c for c in bdf.columns if c != ref_method and c != "n_hidden_states"]
    bdf_rel = bdf.copy()
    bdf_rel[[*methods, ref_method]] = bdf[[*methods, ref_method]].div(bdf[ref_method], axis=0)
    bdf_rel_long = bdf_rel.melt(
        id_vars=["n_hidden_states"], var_name="provider", value_name="samples/ms"
    )
    ax = plot_relative_performance(bdf_rel_long)
    ax.figure.savefig(tgt_folder / "relative-performance.png", dpi=300)
    plt.close(ax.figure)
