import os
from pathlib import Path

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

import torch
import triton
from pydantic_settings import BaseSettings

from ..core import get_gpu_name, get_sampler

# prevent torch._dynamo.exc.FailOnRecompileLimitHit: recompile_limit reached with fullgraph=True
assert torch._dynamo.config.cache_size_limit == 8
torch._dynamo.config.cache_size_limit = 1_000

device = torch.device("cuda")

BASE_VOCAB_SIZE = 256000
HIDDEN_SIZE = 8192
N_SAMPLES = 1
TEMPERATURE = 1.0


class Args(BaseSettings):
    tgt_dir: Path
    name: str | None = None
    n_hidden_states: int | None = None


class CliArgs(Args, cli_parse_args=True):
    pass


provider_names = {
    "fused-triton": "Fused Matmul-Sampling",
    "naive-compiled": "Naive PyTorch Compiled",
    # "sequential-compiled": "Sequential PyTorch Compiled",
    # "naive-tl-matmul": "Naive Triton Matmul",
    # "jl-compiled": "JL Compiled",
    "flashinfer:top_k_top_p_sampling_from_logits": "flashinfer:top_k_top_p_sampling_from_logits",
    "flashinfer:sampling_from_logits": "flashinfer:sampling_from_logits",
}

all_styles = [("blue", "-"), ("green", "-"), ("orange", "-"), ("red", "-"), ("purple", "-")]


def create_benchmark(args: Args, mode: str):
    """Create a benchmark function for the specified mode."""

    if args.n_hidden_states is not None:
        x_vals = [args.n_hidden_states]
    else:
        x_vals = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    if args.name is not None:
        providers = [args.name]
    else:
        providers = list(provider_names.keys())

    lines_names = [provider_names[prov] for prov in providers]

    if mode == "batch":
        # Scale over batch size (n_hidden_states)
        config = triton.testing.Benchmark(
            x_names=["n_hidden_states"],
            x_vals=x_vals,
            # x_vals=[1, 8, 16, 64, 128],
            x_log=True,
            line_arg="provider",
            line_vals=providers,
            line_names=lines_names,
            styles=all_styles[: len(providers)],
            ylabel="Time (ms)",
            plot_name="fused-mm-sample-batch-scaling",
            args={},
        )

        @triton.testing.perf_report(config)
        def benchmark(n_hidden_states, provider):
            # Prepare inputs with fixed vocab_size, varying batch
            hidden_states = torch.randn(
                (n_hidden_states, HIDDEN_SIZE), dtype=torch.bfloat16, device=device
            )
            weights = torch.randn(
                (BASE_VOCAB_SIZE, HIDDEN_SIZE), dtype=torch.bfloat16, device=device
            )
            return _run_benchmark(hidden_states, weights, provider)

    elif mode == "vocab":
        raise NotImplementedError("Please fix")
        # Scale over vocabulary size
        config = triton.testing.Benchmark(
            x_names=["vocab_size"],
            x_vals=[250_000, 200_000, 175_000, 150_000, 125_000, 100_000, 80_000, 64_000],
            # x_vals=[250_000, 175_000, 125_000, 64_000],
            x_log=True,
            line_arg="provider",
            line_vals=[
                "fused-triton",
                "naive-compiled",
                "jl-compiled",
                "flashinfer:top_k_top_p_sampling_from_logits",
                "flashinfer:sampling_from_logits",
            ],
            line_names=[
                "Fused Matmul-Sampling",
                "Naive Compiled",
                "JL Compiled",
                "flashinfer:top_k_top_p_sampling_from_logits",
                "flashinfer:sampling_from_logits",
            ],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-"), ("red", "-"), ("purple", "-")],
            ylabel="Time (ms)",
            plot_name="fused-mm-sample-vocab-scaling",
            args={},
        )

        @triton.testing.perf_report(config)
        def benchmark(vocab_size, provider):
            # Prepare inputs with fixed batch, varying vocab_size
            hidden_states = torch.randn(
                (256, HIDDEN_SIZE), dtype=torch.bfloat16, device=device
            )  # Fixed batch=256
            weights = torch.randn((HIDDEN_SIZE, vocab_size), dtype=torch.bfloat16, device=device)
            return _run_benchmark(hidden_states, weights, provider)

    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'batch' or 'vocab'.")

    return benchmark


def _run_benchmark(hidden_states: torch.Tensor, weights: torch.Tensor, provider: str) -> float:
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

    quantiles = [0.1, 0.5, 0.9]
    return triton.testing.do_bench(fn, quantiles=quantiles)


def run_triton_bechmark(args: Args):
    print("GPU:", get_gpu_name())
    print("Arguments:", args.model_dump_json())
    modes = ["batch"]  # , "vocab"]

    for mode in modes:
        print("=" * 80)
        print(f"Benchmark Mode: {mode}")
        print("Configuration:")
        print(f"  hidden_size: {HIDDEN_SIZE}")
        print(f"  n_samples: {N_SAMPLES}")
        print(f"  temperature: {TEMPERATURE}")
        print()

        benchmark = create_benchmark(args, mode)
        directory = args.tgt_dir
        os.makedirs(directory, exist_ok=True)
        benchmark.run(print_data=True, save_path=directory)
        print()
