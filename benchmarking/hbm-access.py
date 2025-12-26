import pandas as pd
from pydantic_settings import BaseSettings


def our_method(vocab_size: int, hidden_size: int, n_hidden_states: int):
    k = hidden_size
    n = n_hidden_states
    m = vocab_size
    reads = m * k + n * k
    writes = m * n / 128
    return reads, writes


def naive_method(vocab_size: int, hidden_size: int, n_hidden_states: int):
    k = hidden_size
    n = n_hidden_states
    m = vocab_size
    reads = m * k + n * k + m * n
    writes = 2 * m * n
    return reads, writes


class Args(BaseSettings):
    vocab_size: int
    hidden_size: int


class CliArgs(Args, cli_parse_args=True):
    pass


args = CliArgs()


rows = []
for n_hidden_states in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    our_r, our_w = our_method(args.vocab_size, args.hidden_size, n_hidden_states)
    naive_r, naive_w = naive_method(args.vocab_size, args.hidden_size, n_hidden_states)
    rows.append(
        {
            "n_hidden_states": n_hidden_states,
            "our_reads": our_r,
            "our_writes": our_w,
            "naive_reads": naive_r,
            "naive_writes": naive_w,
            "ratio": (our_r + our_w) / (naive_r + naive_w),
        }
    )

df = pd.DataFrame(rows)
print(df.round(2))
