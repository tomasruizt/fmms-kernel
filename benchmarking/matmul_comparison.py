from pathlib import Path

from fused_mm_sampling.bench.matmul_comparison import benchmark

if __name__ == "__main__":
    out_dir = Path(__file__).parent / "profiles" / "matmul"
    out_dir.mkdir(parents=True, exist_ok=True)
    benchmark.run(show_plots=True, print_data=True, save_path=out_dir)
