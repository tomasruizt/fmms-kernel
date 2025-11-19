import os

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

import pandas as pd
import torch
import triton

from fused_mm_sampling.core import get_sampler

torch.set_default_device("cuda")

# Constants from speed_test.py
BASE_VOCAB_SIZE = 256000
HIDDEN_SIZE = 8192
N_SAMPLES = 1
TEMPERATURE = 1.0


def create_benchmark(mode: str):
    """Create a benchmark function for the specified mode."""

    if mode == "batch":
        # Scale over batch size (n_hidden_states)
        config = triton.testing.Benchmark(
            x_names=["n_hidden_states"],
            x_vals=[1, 4, 16, 32, 64, 128, 256, 512, 1024],
            x_log=True,
            line_arg="provider",
            line_vals=["fused-triton", "naive-pt", "naive-compiled", "jl-compiled"],
            line_names=["Fused Triton", "Naive PyTorch", "Naive Compiled", "JL Compiled"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-"), ("red", "-")],
            ylabel="Samples/ms",
            plot_name="fused-mm-sample-batch-scaling",
            args={},
        )

        @triton.testing.perf_report(config)
        def benchmark(n_hidden_states, provider):
            # Prepare inputs with fixed vocab_size, varying batch
            hidden_states = torch.randn((HIDDEN_SIZE, n_hidden_states), dtype=torch.bfloat16)
            weights = torch.randn((BASE_VOCAB_SIZE, HIDDEN_SIZE), dtype=torch.bfloat16)
            ms, min_ms, max_ms = _run_benchmark(hidden_states, weights, provider)
            total_n_samples = n_hidden_states * N_SAMPLES
            samples_per_ms = total_n_samples / ms
            max_samples_per_ms = total_n_samples / min_ms
            min_samples_per_ms = total_n_samples / max_ms
            return samples_per_ms, min_samples_per_ms, max_samples_per_ms

    elif mode == "vocab":
        # Scale over vocabulary size
        config = triton.testing.Benchmark(
            x_names=["vocab_size"],
            x_vals=[250_000, 200_000, 175_000, 150_000, 125_000, 100_000, 80_000, 64_000],
            x_log=True,
            line_arg="provider",
            line_vals=["fused-triton", "naive-pt", "naive-compiled", "jl-compiled"],
            line_names=["Fused Triton", "Naive PyTorch", "Naive Compiled", "JL Compiled"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-"), ("red", "-")],
            ylabel="Time (ms)",
            plot_name="fused-mm-sample-vocab-scaling",
            args={},
        )

        @triton.testing.perf_report(config)
        def benchmark(vocab_size, provider):
            # Prepare inputs with fixed batch, varying vocab_size
            hidden_states = torch.randn((HIDDEN_SIZE, 256), dtype=torch.bfloat16)  # Fixed batch=256
            weights = torch.randn((vocab_size, HIDDEN_SIZE), dtype=torch.bfloat16)
            return _run_benchmark(hidden_states, weights, provider)

    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'batch' or 'vocab'.")

    return benchmark


def _run_benchmark(hidden_states, weights, provider):
    """Common benchmark logic for all modes."""
    print(f"Running benchmark for provider: {provider}")

    kwargs = dict(
        hidden_states=hidden_states,
        weights=weights,
        num_samples=N_SAMPLES,
        temperature=TEMPERATURE,
    )

    sampler = get_sampler(provider, weights=weights)
    sampler.prepare()

    def fn():
        return sampler.sample(**kwargs)

    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    quantiles = [0.1, 0.5, 0.9]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    return ms, min_ms, max_ms


if __name__ == "__main__":
    modes = ["batch", "vocab"]

    for mode in modes:
        print("=" * 80)
        print(f"Benchmark Mode: {mode}")
        print("Configuration:")
        if mode == "batch":
            print(f"  vocab_size: {BASE_VOCAB_SIZE} (fixed)")
            print("  n_hidden_states: 1 → 1024 (scaling)")
        else:  # vocab mode
            print(f"  vocab_size: {BASE_VOCAB_SIZE} → {BASE_VOCAB_SIZE // 16} (scaling)")
            print("  n_hidden_states: 256 (fixed)")
        print(f"  hidden_size: {HIDDEN_SIZE}")
        print(f"  n_samples: {N_SAMPLES}")
        print(f"  temperature: {TEMPERATURE}")
        print()

        benchmark = create_benchmark(mode)
        directory = "profiles/plots/"
        os.makedirs(directory, exist_ok=True)
        df: pd.DataFrame = benchmark.run(
            print_data=True, show_plots=True, save_path=directory, return_df=True
        )
        csv = f"profiles/triton/{mode}.csv"
        os.makedirs(os.path.dirname(csv), exist_ok=True)
        df.to_csv(csv, index=False)
        print(f"Saved benchmark results to {csv}")
        print()
