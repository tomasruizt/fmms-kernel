import os

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
# os.environ["TRITON_INTERPRET"] = "1"

import timeit
from dataclasses import dataclass

import pandas as pd
import torch
from pydantic_settings import BaseSettings

from fused_mm_sampling import fused_mm_sample_triton
from fused_mm_sampling.core import (
    JLSampler,
    Sampler,
    SimpleSampler,
    sample,
)

torch.set_default_device("cuda")


class Args(BaseSettings, cli_parse_args=True):
    name: str | None = None
    n_runs_warmup: int = 10
    n_runs_benchmark: int = 1

    n_hidden_states: int = 256
    n_samples: int = 1

    def as_case(self) -> "Case":
        assert self.n_runs_warmup is not None
        assert self.n_runs_benchmark is not None
        return Case(
            name=self.name,
            n_runs_benchmark=self.n_runs_benchmark,
            n_runs_warmup=self.n_runs_warmup,
            n_hidden_states=self.n_hidden_states,
            n_samples=self.n_samples,
        )


vocab_size = 256000
hidden_size = 8192


sample_compiled = torch.compile(sample)


@dataclass
class Case:
    name: str
    n_runs_benchmark: int
    n_runs_warmup: int = 10

    n_hidden_states: int = 256
    n_samples: int = 1

    def make_fn_kwargs(self) -> dict:
        """This function can be slow because it allocates tensors."""
        return dict(
            hidden_states=torch.randn((hidden_size, self.n_hidden_states), dtype=torch.bfloat16),
            weights=torch.randn((vocab_size, hidden_size), dtype=torch.bfloat16),
            num_samples=self.n_samples,
            temperature=1.0,
        )


all_cases = [
    Case(name="fused-triton", n_runs_benchmark=10),
    Case(name="naive-pt", n_runs_benchmark=10),
    Case(name="naive-compiled", n_runs_benchmark=10),
    Case(name="jl-compiled", n_runs_benchmark=10),
]


def benchmark(case: Case) -> pd.DataFrame:
    kwargs = case.make_fn_kwargs()
    samplers: dict[str, Sampler] = {
        "fused-triton": SimpleSampler(lambda **kwargs: fused_mm_sample_triton(**kwargs, seed=0)),
        "naive-pt": SimpleSampler(sample),
        "naive-compiled": SimpleSampler(sample_compiled),
        "jl-compiled": JLSampler(kwargs["weights"], k=100),
    }
    sampler = samplers[case.name]
    sampler.prepare()

    def fn():
        return sampler.sample(**kwargs)

    print("Warming up...")
    for _ in range(case.n_runs_warmup):
        fn()
    torch.cuda.synchronize()

    print("Timing...")
    times_ms = []
    for _ in range(case.n_runs_benchmark):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        timeit.timeit(fn, number=1)
        end_event.record()
        end_event.synchronize()
        ms_elapsed = start_event.elapsed_time(end_event)
        times_ms.append(ms_elapsed)

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


if __name__ == "__main__":
    args = Args()
    if args.name is not None:
        cases = [args.as_case()]
    else:
        cases = all_cases
    df = benchmark_all(cases)
    print(f"{vocab_size=}")
    print(f"{hidden_size=}")
    print(f"{args.n_hidden_states=}")
    print(f"{args.n_samples=}")

    total_runtimes = df.groupby(["name", "total[s]"], as_index=False).size()
    print(total_runtimes.round(2))

    time_distribution = df.groupby("name")[["time[ms]"]].describe()
    print(time_distribution.round(2))
