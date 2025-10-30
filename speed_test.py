import os

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

from dataclasses import dataclass
from time import time
import timeit
import pandas as pd
import torch
from fused_mm_sampling import fused_mm_sample_triton, sample
from pydantic_settings import BaseSettings

torch.set_default_device("cuda")


class Args(BaseSettings, cli_parse_args=True):
    name: str | None = None
    n_runs_warmup: int = 10
    n_runs_benchmark: int = 1

    def as_case(self) -> "Case":
        assert self.n_runs_warmup is not None
        assert self.n_runs_benchmark is not None
        return Case(
            name=self.name,
            n_runs_benchmark=self.n_runs_benchmark,
            n_runs_warmup=self.n_runs_warmup,
        )


vocab_size = 256000
hidden_size = 5120
seq_len = 256
num_samples = 1
speedtest_kwargs = dict(
    hidden_states=torch.randn((hidden_size, seq_len)).bfloat16(),
    weights=torch.randn((vocab_size, hidden_size)).bfloat16(),
    num_samples=num_samples,
    temperature=1.0,
)

sample_compiled = torch.compile(sample)


@dataclass
class Case:
    name: str
    n_runs_benchmark: int
    n_runs_warmup: int = 10


fns = {
    "fused-triton": lambda: fused_mm_sample_triton(**speedtest_kwargs, seed=0),
    "naive-pt": lambda: sample(**speedtest_kwargs),
    "naive-compiled": lambda: sample_compiled(**speedtest_kwargs),
}


all_cases = [
    Case(name="fused-triton", n_runs_benchmark=10),
    Case(name="naive-pt", n_runs_benchmark=10),
    Case(name="naive-compiled", n_runs_benchmark=10),
]


def benchmark(case: Case) -> pd.DataFrame:
    print(f"Benchmarking fn='{case.name}'")

    print("Warming up...")
    fn = fns[case.name]
    for _ in range(case.n_runs_warmup):
        fn()
    torch.cuda.synchronize()

    print("Timing...")
    start = time()
    with torch.cuda.nvtx.range("kernel"):
        times = timeit.repeat(fn, repeat=case.n_runs_benchmark, number=1)
        torch.cuda.synchronize()
    end = time()
    total_time = end - start
    # According to time.repeat() the min() is the most informative statistic
    results = {
        "name": case.name,
        "total[s]": total_time,
        "time[s]": times,
        "time[ms]": [t * 1_000 for t in times],
        "time[µs]": [t * 1_000_000 for t in times],
    }
    df = pd.DataFrame(results)
    return df


def benchmark_all(cases: list[Case]) -> pd.DataFrame:
    import gc

    gc.disable()
    try:
        dfs = [benchmark(case) for case in cases]
        return pd.concat(dfs, ignore_index=True)
    finally:
        gc.enable()


if __name__ == "__main__":
    args = Args()
    if args.name is not None:
        cases = [args.as_case()]
    else:
        cases = all_cases
    df = benchmark_all(cases)
    print(f"{vocab_size=}")
    print(f"{hidden_size=}")
    print(f"{seq_len=}")
    print(f"{num_samples=}")

    total_runtimes = df.groupby(["name", "total[s]"], as_index=False).size()
    print(total_runtimes.round(2))

    time_distribution = df.groupby("name")[["time[ms]"]].describe()
    print(time_distribution.round(2))
