import timeit
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import triton
import triton.profiler as proton
from pydantic_settings import BaseSettings

from ..core import fused_mm_sample_triton_kernel, get_gpu_name, get_sampler, sample

device = torch.device("cuda")


class Args(BaseSettings):
    name: str | None = None
    n_runs_warmup: int = 25
    n_runs_benchmark: int = 100

    n_hidden_states: int = 256
    n_samples: int = 1
    tgt_dir: Path | None = None
    use_proton: bool = False

    def as_case(self, name: str | None = None) -> "Case":
        if name is None:
            name = self.name
        assert self.n_runs_warmup is not None
        assert self.n_runs_benchmark is not None
        return Case(
            name=name,
            n_runs_benchmark=self.n_runs_benchmark,
            n_runs_warmup=self.n_runs_warmup,
            n_hidden_states=self.n_hidden_states,
            n_samples=self.n_samples,
            use_proton=self.use_proton,
        )

    def all_cases(self) -> list["Case"]:
        return [self.as_case(name=provider) for provider in all_providers]


class CliArgs(Args, cli_parse_args=True):
    pass


vocab_size = 256000
hidden_size = 8192


sample_compiled = torch.compile(sample)


@dataclass
class Case:
    name: str
    n_runs_benchmark: int
    n_runs_warmup: int
    n_hidden_states: int
    n_samples: int
    use_proton: bool

    def make_fn_kwargs(self) -> dict:
        """This function can be slow because it allocates tensors."""
        return dict(
            hidden_states=torch.randn(
                (self.n_hidden_states, hidden_size), dtype=torch.bfloat16, device=device
            ),
            weights=torch.randn((vocab_size, hidden_size), dtype=torch.bfloat16, device=device),
            num_samples=self.n_samples,
            temperature=1.0,
        )


all_providers = [
    "fused-triton",
    "helion",
    "naive-compiled",
    # "sequential-compiled",
    # "naive-tl-matmul",
    # "jl-compiled",
    "flashinfer:top_k_top_p_sampling_from_logits",
    "flashinfer:sampling_from_logits",
]


def setup_proton() -> None:
    # Start proton BEFORE kernel compilation so hook="triton" can instrument the JIT
    print("⚙️ Proton profiling enabled")
    proton.start(name="kernel", hook="triton", backend="cupti", mode="pcsampling")

    def enter_autotune(args, reset_only=False):
        if reset_only:
            return
        proton.enter_scope("<autotune>")

    def exit_autotune(args, exception):
        proton.exit_scope()

    fused_mm_sample_triton_kernel.pre_hook = enter_autotune
    fused_mm_sample_triton_kernel.post_hook = exit_autotune


@proton.scope("clear-l2-cache")
def clear_l2_cache(cache):
    with torch.cuda.nvtx.range("clear-l2-cache"):
        triton.runtime.driver.active.clear_cache(cache)


def benchmark(case: Case) -> pd.DataFrame:
    """Inspired by triton.testing.do_bench"""
    print("=" * 80)
    print(f"Benchmarking {case.name}...")
    kwargs = case.make_fn_kwargs()
    sampler = get_sampler(case.name, weights=kwargs["weights"])
    sampler.prepare()

    def fn():
        return sampler.sample(**kwargs)

    di = triton.runtime.driver.active.get_device_interface()

    if case.use_proton:
        setup_proton()

    with proton.scope("first-run"):
        # Compile, etc.
        fn()
        di.synchronize()

    cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()

    start_events = [di.Event(enable_timing=True) for _ in range(case.n_runs_benchmark)]
    end_events = [di.Event(enable_timing=True) for _ in range(case.n_runs_benchmark)]

    print("Warming up...")
    with proton.scope("warmup"):
        for _ in range(case.n_runs_warmup):
            clear_l2_cache(cache)
            fn()

    print("Timing...")
    with proton.scope("timing"):
        for _, start_event, end_event in zip(
            range(case.n_runs_benchmark), start_events, end_events
        ):
            clear_l2_cache(cache)
            with torch.cuda.nvtx.range("kernel"):
                start_event.record()
                timeit.timeit(fn, number=1)
                end_event.record()
            di.synchronize()

    if case.use_proton:
        proton.finalize()

    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    results = {
        "name": case.name,
        "total[s]": sum(times_ms) / 1_000,
        "time[s]": [t / 1_000 for t in times_ms],
        "time[ms]": times_ms,
        "time[µs]": [t * 1_000 for t in times_ms],
    }
    df = pd.DataFrame(results)
    return df


def benchmark_all(cases: list[Case]) -> pd.DataFrame:
    dfs = [benchmark(case) for case in cases]
    return pd.concat(dfs, ignore_index=True)


def run_speed_test(args: Args) -> None:
    """Run a speed test for a given set of arguments."""
    print("GPU:", get_gpu_name())
    print("Arguments: ", args)
    if args.name is not None:
        cases = [args.as_case()]
    else:
        cases = args.all_cases()
    df = benchmark_all(cases)
    print(f"{vocab_size=}")
    print(f"{hidden_size=}")
    print(f"{args.n_hidden_states=}")
    print(f"{args.n_samples=}")

    total_runtimes = df.groupby(["name", "total[s]"], as_index=False).size()
    print(total_runtimes.sort_values("total[s]").round(2))

    time_distribution = df.groupby("name")[["time[ms]"]].describe()
    print(time_distribution.sort_values(("time[ms]", "min")).round(2))

    if args.tgt_dir is not None:
        args.tgt_dir.mkdir(parents=True, exist_ok=True)
        total_runtimes.to_csv(args.tgt_dir / "total-runtimes.csv")
        time_distribution.to_csv(args.tgt_dir / "time-distribution.csv")
        print("Saved results to ", args.tgt_dir)
