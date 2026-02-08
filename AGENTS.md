# AGENTS.md

Development notes and lessons learned while building this project.

## Development environment

- Use the `.venv` in the repo root (not system Python). Run tests/scripts with `.venv/bin/python` or `.venv/bin/pytest`.

## Findings

The `findings/` directory contains detailed write-ups of bugs, workarounds, and design decisions discovered during development:

- `upcasting-before-softmax.md` — `torch.multinomial` produces wrong distributions with bfloat16 due to CDF precision loss. Fix: upcast to float32 before softmax.
- `helion-hl-rand-specialize-1-bug.md` — `hl.rand` crashes when a dimension is `hl.specialize(1)`. Includes root cause analysis, in-place fix, and minimal reproduction.

## Architecture

- **Weights**: `[V, D]`, **hidden_states**: `[H, D]` everywhere in public APIs.
- The Helion kernel internally uses `hidden_states` as `[D, H]` (transposed) for matmul efficiency. The wrapper handles the transpose.
- All sampler variants are registered in `get_sampler()` in `core.py` via a match/case. New samplers only need a case there.
- The `Sampler` Protocol requires `prepare()` and `sample(**kwargs)`. Wrap simple callables with `SimpleSampler`.

## Helion kernel pitfalls

### `torch.argmax()` returns global indices

Inside a Helion kernel, `torch.argmax(tensor, dim=0)` returns **global** indices, not tile-local ones. The generated Triton code uses `triton_helpers.max_with_index` with the tile's global `indices_0` offset baked in. **Do NOT add `tile_v.begin`** — that double-counts the offset.

```python
# WRONG — double counts offset
new_max_idx_local = torch.argmax(summed, dim=0)
new_max_idx_global = tile_v.begin + new_max_idx_local  # BUG

# CORRECT — argmax already returns global indices
new_max_idx = torch.argmax(summed, dim=0)
```

### `for tile in hl.tile(N)` is parallel, not sequential

Each tile becomes a separate GPU program (thread block) that runs in parallel. You **cannot** do cross-tile communication via shared tensors — it's a race condition. This manifests as correct-looking results for some vocab sizes (e.g. 256, evenly divisible) but broken distributions for others (e.g. 100).

**Fix**: Use a two-stage approach:
1. **Stage 1 (kernel)**: Each tile writes its local max/argmax to `tile_maxs[tile_v.id, :]`.
2. **Stage 2 (Python)**: Reduce across tiles with `argmax` + `gather`.

This matches the pattern used by the hand-written Triton kernel.

### Tensor allocations inside kernels trigger warnings

`TensorOperationInWrapper` warning fires for tensor ops outside `hl.tile` loops. Allocate output buffers in the Python wrapper and pass them as kernel arguments instead.

### Random number generation

Use `hl.rand([tile_v, n], seed=seed)` — not `torch.rand` or `torch.rand_like`. The `hl.rand` API uses Philox PRNG with proper per-tile offsets. Historical issues #1041 and #1309 are fixed in current Helion.

**Bug: `hl.rand` crashes when a dimension is `hl.specialize(1)`**. The `_rand_codegen` in `random_ops.py` tries to look up a block ID for every dimension, but a specialized size-1 dimension has no associated tile loop. `hl.zeros`/`hl.full` don't have this problem because they only need the shape, not index variables. Fix: in `_rand_codegen` (and `_randint_codegen`), when `size == 1`, use `tl.full([1], 0, tl.int32)` as the index var and `"1"` as the size name instead of trying to allocate a reduction dimension. We applied this fix in-place in `.venv/lib/python3.12/site-packages/helion/language/random_ops.py` — it will be lost on reinstall. Filed upstream: https://github.com/pytorch/helion/issues/1397

### Autotuning

- `autotune_effort`: `"none"` / `"quick"` / `"full"`. Controlled via `HELION_AUTOTUNE_EFFORT` env var (default: `"quick"`). Tests set it to `"none"` for speed.
- `LocalAutotuneCache` caches best config per GPU on disk. Cache dir set to `helion-cache/` in the repo root via `HELION_CACHE_DIR` env var (gitignored). Different GPUs autotune independently.
- First run with a new specialization key (e.g. new `n_hidden_states` value via `hl.specialize`) triggers autotuning (~3 min for `"full"`). Subsequent runs use the cache instantly.
- Set `HELION_AUTOTUNE_ACCURACY_CHECK=0` for stochastic kernels (the kernel output changes each run, so accuracy checks would always fail).
- To force re-tuning: delete the cache dir or set `HELION_SKIP_CACHE=1`.

### Performance: BLOCK_SIZE_V matters for stage 2

With `BLOCK_SIZE_V=32` and vocab=128K, there are 4000 tiles. The stage-2 `argmax` over 4000 rows costs ~1ms. Increasing to `BLOCK_SIZE_V=128` (1000 tiles) drops stage-2 to 0.02ms with no kernel slowdown.

## `torch.multinomial` and bfloat16

`torch.multinomial` produces incorrect sampling distributions when given bfloat16 probabilities. The internal CDF accumulation loses precision. Fix: upcast to float32 before softmax:

```python
probs = (logits.float() / temperature).softmax(dim=1)
```

See `findings/upcasting-before-softmax.md` for details.

## Testing

- `test_sampling_distribution` uses a chi-squared goodness-of-fit test comparing empirical samples against theoretical softmax probabilities.
- Parametrized over all providers, multiple vocab sizes (100, 256), and n_hidden_states (1, 2) to catch tile-boundary and dimension-edge-case bugs.
- Bins with expected count < 5 are excluded (chi-squared assumption). Expected counts are rescaled to match observed totals.
- `make_synthetic_inputs()` in `src/fused_mm_sampling/testing.py` constructs weights/hidden_states that produce known logit vectors (ascending and descending) via SVD + pseudoinverse.

## Naming conventions

The algorithm is called **FMMS** (Fused Matrix Multiplication & Sampling). Provider display names in benchmarks follow the pattern:
- `"FMMS (Triton)"` — hand-written Triton kernel
- `"FMMS (Helion)"` — Helion kernel
- `"FMMS (Triton NoNoise)"` — Triton kernel without Gumbel noise (for profiling)

These names are defined in `provider_names` in `src/fused_mm_sampling/bench/triton_benchmark.py` and used in plots, CSVs, and the README.

## Triton benchmark CSV format

Triton's `perf_report` appends ` (Time (ms))` to column names based on `ylabel`. The plotting code strips this suffix via `read_triton_bench_csv()` in `benchmarking/plot-triton-bench.py`.
