import os

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

from pathlib import Path

import torch
import triton

from .. import tl_matmul

cublas = tl_matmul.get_cublas()

DEVICE = torch.device("cuda")

configs = [
    triton.testing.Benchmark(
        x_names=["M"],  # Argument names to use as an x-axis for the plot
        x_vals=[1, 4, 16, 64, 128, 256, 512],  # Different possible values for `x_name`
        x_log=True,
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        line_vals=["triton", "torch", "cublas"],  # Label name for the lines
        line_names=["Triton", "Torch", "cuBLAS"],  # Line styles
        styles=[("green", "-"), ("red", "-"), ("blue", "-")],
        # ylabel="TFLOPS",
        ylabel="Time (ms)",
        plot_name="matmul-performance-fp16",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    )
]


@triton.testing.perf_report(configs)
def benchmark(
    M: int,  # noqa: N803
    provider: str,
):
    print(f"Running benchmark for M={M}, provider: {provider}")
    N = 256_000  # noqa: N806
    K = 8_192  # noqa: N806
    dtype = torch.bfloat16
    a = torch.randn((M, K), device=DEVICE, dtype=dtype)
    b = torch.randn((N, K), device=DEVICE, dtype=dtype)
    quantiles = [0.5, 0.0, 1.0]

    def cublas_matmul(a, b):
        c = torch.empty((M, N), device=DEVICE, dtype=dtype)
        cublas.matmul(a, b, c)
        return c

    mapping = {
        "triton": lambda: tl_matmul.matmul(a, b),
        "torch": lambda: torch.matmul(a, b.T),
        "cublas": lambda: cublas_matmul(a, b),
        # "helion": helion_impl.matmul,
    }
    fn = mapping[provider]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    # perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    # return perf(ms), perf(max_ms), perf(min_ms)
    return ms, min_ms, max_ms


def matmul_comparison_main():
    out_dir = Path(__file__).parent / "profiles" / "matmul"
    out_dir.mkdir(parents=True, exist_ok=True)
    benchmark.run(show_plots=True, print_data=True, save_path=out_dir)
