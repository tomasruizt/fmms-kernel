import timeit
from dataclasses import dataclass

import pandas as pd
import torch
from pydantic_settings import BaseSettings

from ..core import get_sampler, sample

device = torch.device("cuda")


class Args(BaseSettings):
    name: str | None = None
    n_runs_warmup: int = 25
    n_runs_benchmark: int = 100

    n_hidden_states: int = 256
    n_samples: int = 1

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

    def make_fn_kwargs(self) -> dict:
        """This function can be slow because it allocates tensors."""
        return dict(
            hidden_states=torch.randn(
                (self.n_hidden_states, hidden_size), dtype=torch.bfloat16, device=device
            ),
            weights=torch.randn((hidden_size, vocab_size), dtype=torch.bfloat16, device=device),
            num_samples=self.n_samples,
            temperature=1.0,
        )


all_providers = [
    "fused-triton",
    "naive-compiled",
    "naive-tl-matmul",
    "jl-compiled",
    "flashinfer:top_k_top_p_sampling_from_logits",
    "flashinfer:sampling_from_logits",
]


def benchmark(case: Case) -> pd.DataFrame:
    """Inspired by triton.testing.do_bench"""
    import triton  # defer import for Modal compatibility

    print("=" * 80)
    print(f"Benchmarking {case.name}...")
    kwargs = case.make_fn_kwargs()
    sampler = get_sampler(case.name, weights=kwargs["weights"])
    sampler.prepare()

    def fn():
        return sampler.sample(**kwargs)

    di = triton.runtime.driver.active.get_device_interface()

    # Compile, etc.
    fn()
    di.synchronize()

    cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()
    start_events = [di.Event(enable_timing=True) for _ in range(case.n_runs_benchmark)]
    end_events = [di.Event(enable_timing=True) for _ in range(case.n_runs_benchmark)]

    print("Warming up...")
    for _ in range(case.n_runs_warmup):
        triton.runtime.driver.active.clear_cache(cache)
        fn()

    print("Timing...")
    with torch.cuda.nvtx.range("kernel"):
        for _, start_event, end_event in zip(
            range(case.n_runs_benchmark), start_events, end_events
        ):
            triton.runtime.driver.active.clear_cache(cache)
            start_event.record()
            timeit.timeit(fn, number=1)
            end_event.record()
        di.synchronize()
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
