# AGENTS.md

Development notes and lessons learned while building this project.

## Code style

- **Top-down structure**: Define high-level functions first, helpers below. A reader should encounter the main logic before the details it delegates to.
- **Never introduce GPU-CPU synchronizations.** Operations like `tensor.item()`, `float(tensor)`, `tensor.cpu()`, or `print(tensor)` on CUDA tensors force the CPU to wait for all pending GPU work to finish, destroying pipeline parallelism. Pass scalar values as 0-d CUDA tensors instead of extracting Python floats. Both the Triton kernel (`tl.load(temperature_ptr)`) and the Helion kernel (`temperature: torch.Tensor`) accept 0-d tensors directly.
- **Always save logs to the output folder.** When running servers, benchmarks, or evals, pipe stdout/stderr to a log file in the results directory so logs are always accessible after the run. Never discard or hide process output.

## Development environment

- Use the `.venv` in the repo root (not system Python). Run tests/scripts with `.venv/bin/python` or `.venv/bin/pytest`.
- **Save all learnings in this file (`CLAUDE.md`), not in `~/.claude/` MEMORY.md.** The `~/.claude/` directory is local to the server and will be lost when switching machines. This file is checked into git and travels with the code.

### NVIDIA Brev machine quirks

The Brev cloud GPU environment (shadeform) has several non-standard behaviors:

- **`$HOME` is unset** in non-login shells. Always pass `HOME=/home/shadeform` explicitly when running `make` or scripts that depend on `~` expansion. The Makefile's `$(HOME)` resolves to empty string otherwise.
- **Single global venv at `/home/shadeform/.venv/`**, not per-project. Both vLLM and fused-mm-sampling are installed there. The project's `.venv/` (referenced in the Makefile as `$(HOME)/code/fused-mm-sample/.venv/`) and vLLM's `venv/` (`$(HOME)/code/vllm/venv/`) do not exist.
- **vLLM binary**: `/home/shadeform/.venv/bin/vllm` (not `~/code/vllm/venv/bin/vllm`).
- **Python**: `/home/shadeform/.venv/bin/python` (Python 3.10.12).
- **GPU**: 1x NVIDIA H100 PCIe, 81,559 MiB VRAM, CUDA 13.0.
- **`datasets` / `pyarrow` conflict**: The pre-installed `datasets==2.14.4` is incompatible with `pyarrow==23.0.0` (`pa.PyExtensionType` was removed). Fix: `pip install --upgrade datasets` (upgrades to 4.5.0+).
- **HuggingFace**: Not logged in by default. Set `HF_TOKEN` env var for gated models.
- **Pip cache**: `/ephemeral/cache/pip` has wrong permissions; pip disables cache automatically (harmless warning).
- **Makefile portability**: Both `benchmarking/Makefile` and `benchmarking/vllm/Makefile` use `$(shell which python)` / `$(shell which vllm)` to discover binaries dynamically. No hard-coded paths — just activate the correct venv before running `make`. Example:
  ```bash
  HOME=/home/shadeform make -C benchmarking/vllm quick \
    MODEL=openai/gpt-oss-120b \
    HF_TOKEN=<token>
  ```

### CUDA toolkit installation on Brev (H100)

The Brev image ships with the NVIDIA driver (CUDA runtime 13.0) but **no nvcc** by default. Several components require nvcc for JIT compilation:

- **`fused-cuda` provider**: Uses `torch.utils.cpp_extension.load()` to JIT-compile a CUDA C++ extension. Needs nvcc + a compatible C++ compiler (g++-12).
- **flashinfer**: JIT-compiles CUDA kernels on first use. Requires nvcc with sm_90a support.
- **`tvm_ffi`**: The `_optional_torch_c_dlpack.py` module JIT-compiles a helper shared library. Non-fatal if it fails (just a warning).

**Installation steps:**

1. The CUDA 12.2 local repo deb is pre-cached at `/var/cuda-repo-ubuntu2204-12-2-local/`. Install from there:
   ```bash
   sudo dpkg -i /var/cuda-repo-ubuntu2204-12-2-local/cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
   sudo cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
   sudo apt-get update
   sudo apt-get install -y cuda-toolkit-12-2
   ```
2. Install a compatible C++ compiler:
   ```bash
   sudo apt-get install -y g++-12
   ```
3. Set environment variables:
   ```bash
   export CUDA_HOME=/usr/local/cuda-12.2
   export PATH=$CUDA_HOME/bin:$PATH
   ```

**Key pitfalls:**
- The Brev image may have CUDA 11.5 nvcc pre-installed (`/usr/local/cuda-11.5/bin/nvcc`). This is **too old for H100** (doesn't support `compute_90a` / `sm_90a`). You must use CUDA 12.0+ for H100.
- `tests/conftest.py` auto-discovers CUDA_HOME by searching `/usr/local/cuda-*` (preferring highest version) and validates that nvcc supports the current GPU's compute capability. If validation fails, it raises a clear error message.
- After installing a new CUDA toolkit, delete stale JIT caches: `rm -rf /ephemeral/cache/torch_extensions/` to force recompilation with the new nvcc.

## Triton TMA (Tensor Memory Access) pitfalls

TMA uses `tl.make_tensor_descriptor` / `desc.load()` / `desc.store()` for hardware-accelerated memory access on H100. Three hard-won lessons:

### 1. Innermost dimension must be aligned to 16 bytes

TMA descriptors require the **innermost (stride-1) dimension** to be a multiple of 16 bytes. For bfloat16 (2 bytes/element), that means **multiples of 8 elements**. Non-aligned dimensions cause **silent data corruption** — no error, just wrong results.

```
K=304 (304 % 8 == 0) → PASS
K=300 (300 % 8 == 4) → FAIL, max_err=92.0
N=200 (200 % 8 == 0) → PASS
N=33  (33 % 8 == 1)  → FAIL, max_err=34.75
```

**Fix:** Pad tensors in the Python wrapper before passing to the kernel. Zero-padding doesn't affect matmul results. See `_tma_pad()` in `tl_matmul.py`. After the kernel, slice output back to the original dimensions.

### 2. `tl.dot(a, b.T)` does NOT work with TMA-loaded blocks

`.T` only swaps the logical view without rearranging shared memory layout. Tensor core MMA instructions depend on physical (row-major) layout, so the dot product produces wrong results. You must pre-transpose the matrix in the wrapper to make it physically contiguous in the layout the kernel expects.

### 3. Triton enforces `strides[-1] == 1`

You cannot describe a transpose via TMA strides — Triton's `semantic.py` checks that the last stride is 1 and raises `CompilationError` otherwise. The only option is to pre-transpose and make the matrix contiguous in the desired layout.

## Findings

The `findings/` directory contains detailed write-ups of bugs, workarounds, and design decisions discovered during development:

- `upcasting-before-softmax.md` — `torch.multinomial` produces wrong distributions with bfloat16 due to CDF precision loss. Fix: upcast to float32 before softmax.
- `helion-hl-rand-specialize-1-bug.md` — `hl.rand` crashes when a dimension is `hl.specialize(1)`. Includes root cause analysis, in-place fix, and minimal reproduction.
- `helion-barrier-single-kernel.md` — Merging stage 2 into the Helion kernel with `hl.barrier()`. Eliminates host-side reduction, reduces kernel launches from 3 to 1. Rigorous benchmarking shows barrier is ~3% slower at H=1 (host overhead is negligible). Barrier code is on the `barrier-kernel` branch.
- `rtx3090-barrier-comparison/` — Raw benchmark results (speed test, proton, NCU) for barrier vs two-stage on RTX 3090.
- `fused-top-k-top-p-feasibility.md` — Analysis of fusing top-k/top-p into the FMMS kernel. Top-k is feasible (tile-local top-k + merge); top-p is not directly fusible (needs global softmax + sorted cumsum). Practical path: fuse top-k, apply top-p on survivors post-kernel.
- `arithmetic-intensity-decode-matmul.md` — The decode matmul has arithmetic intensity ≈ H (batch size). Memory-bound up to H≈295 on H100 (BF16), H≈152 on RTX 3090. Includes ops:byte ratio derivation and data sources.

## Architecture

- **Weights**: `[V, D]`, **hidden_states**: `[H, D]` everywhere in public APIs.
- The Helion kernel internally uses `hidden_states` as `[D, H]` (transposed) for matmul efficiency. The wrapper handles the transpose.
- All sampler variants are registered in `get_sampler()` in `core.py` via a match/case. New samplers only need a case there.
- The `Sampler` Protocol requires `prepare()` and `sample(**kwargs)`. Wrap simple callables with `SimpleSampler`.

## Helion kernel pitfalls

Helion has an API reference: https://helionlang.com/api/index.html

and also reference examples: https://helionlang.com/examples/index.html

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

**Fix**: Use `hl.barrier()` to synchronize stages within a single kernel:
1. **Stage 1**: Each `(V, H)` tile writes its local max/argmax to `tile_maxs[tile_v.id, :]`.
2. `hl.barrier()` — grid-wide sync.
3. **Stage 2**: Reduce across tiles with `argmax` + `gather` inside the same kernel.

This eliminates Python-side tensor allocations and separate kernel launches for the reduction. However, rigorous benchmarking (25 warmup + 100 runs) shows the barrier version is ~3% **slower** at H=1 on RTX 3090 (2.38ms vs 2.32ms) due to cooperative launch constraints (`num_stages=1`, persistent scheduling, barrier sync stalls). The host-side overhead it eliminates is only ~0.01ms — negligible. The ~5ms gap seen under Proton profiling is an instrumentation artifact (Proton adds fixed overhead per kernel launch). See `findings/helion-barrier-single-kernel.md`.

### Tensor allocations inside kernels trigger warnings

`TensorOperationInWrapper` warning fires for tensor ops outside `hl.tile` loops. Allocate output buffers in the Python wrapper and pass them as kernel arguments instead.

### Advanced indexing does not work for gather

Helion interprets `tensor[idx_tensor, tile_var]` as a Cartesian product (producing a higher-rank result), not element-wise gather. Use `torch.gather` instead:

```python
# WRONG — Cartesian product, produces 2D
out[tile_h] = tile_max_idxs[best_tile, tile_h]  # RankMismatch error

# CORRECT — element-wise gather
out[tile_h] = torch.gather(
    tile_max_idxs[:, tile_h], 0, best_tile.unsqueeze(0)
).squeeze(0)
```

### Random number generation

Use `hl.rand([tile_v, n], seed=seed)` — not `torch.rand` or `torch.rand_like`. The `hl.rand` API uses Philox PRNG with proper per-tile offsets. Historical issues #1041 and #1309 are fixed in current Helion.

**Bug: `hl.rand` crashes when a dimension is `hl.specialize(1)`**. The `_rand_codegen` in `random_ops.py` tries to look up a block ID for every dimension, but a specialized size-1 dimension has no associated tile loop. `hl.zeros`/`hl.full` don't have this problem because they only need the shape, not index variables. Fix: in `_rand_codegen` (and `_randint_codegen`), when `size == 1`, use `tl.full([1], 0, tl.int32)` as the index var and `"1"` as the size name instead of trying to allocate a reduction dimension. We applied this fix in-place in `.venv/lib/python3.12/site-packages/helion/language/random_ops.py` — it will be lost on reinstall. Filed upstream: https://github.com/pytorch/helion/issues/1397

### Autotuning

- `autotune_effort`: `"none"` / `"quick"` / `"full"`. Controlled via `HELION_AUTOTUNE_EFFORT` env var (default: `"quick"`). Tests set it to `"none"` for speed.
- `LocalAutotuneCache` caches best config per GPU on disk. Cache dir set to `helion-cache/` in the repo root via `HELION_CACHE_DIR` env var (gitignored). Different GPUs autotune independently.
- First run with a new specialization key (e.g. new `n_hidden_states` value via `hl.specialize`) triggers autotuning (~3 min for `"full"`). Subsequent runs use the cache instantly.
- Set `HELION_AUTOTUNE_ACCURACY_CHECK=0` for stochastic kernels (the kernel output changes each run, so accuracy checks would always fail).
- To force re-tuning: delete the cache dir or set `HELION_SKIP_CACHE=1`.

### Performance: barrier kernel vs two-stage

Rigorous benchmarking (25 warmup + 100 runs, RTX 3090, V=128K, D=8192, H=1) shows the **two-stage version is ~3% faster** (2.32ms vs 2.38ms median). The host-side overhead eliminated by the barrier (tensor alloc + 3 auxiliary kernel launches) is only ~0.01ms. The barrier kernel pays for cooperative launch constraints: `num_stages=1` (no pipelining), 164 persistent blocks vs 1,002 one-shot blocks, and 52% of CPI spent on barrier sync stalls.

Initial Proton profiling showed a ~5ms wall-clock advantage for the barrier version — this is a **profiling artifact**. Proton adds fixed instrumentation overhead per kernel launch, making the 4-launch two-stage appear much slower than the 1-launch barrier. Always cross-reference Proton with uninstrumented speed tests.

See `findings/helion-barrier-single-kernel.md` for full NCU and Proton analysis, and `findings/rtx3090-barrier-comparison/` for raw data.

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

## Proton profiling

Documentation: https://github.com/triton-lang/triton/tree/main/third_party/proton

`speed_test.py --use_proton=True` enables Proton profiling with `mode="pcsampling"` (instruction sampling), which gives per-line runtime breakdowns for Triton kernels. Key API:

- `proton.start(name, hook="triton", backend="cupti", mode="pcsampling")` — initialize profiling. `mode="pcsampling"` enables PC sampling for per-line stats (~20x end-to-end overhead, but per-kernel overhead is negligible).
- `proton.scope(name)` — context manager to annotate regions (warmup, timing, etc.).
- `proton.finalize()` — flush and write profile data.
- `proton-viewer` CLI to render `.hatchet` files as trees.

**Known issue**: `mode="pcsampling"` segfaults when non-Triton CUDA kernels (e.g. `torch.gather`) are launched during profiling. This affects the Helion barrier kernel which calls `torch.gather` in stage 2. Workaround: use `--name fused-triton` to profile only the hand-written Triton kernel, or omit `mode="pcsampling"` (loses per-line granularity).

**Pitfall: Proton inflates per-launch overhead.** Proton adds fixed instrumentation cost per kernel launch. When comparing approaches with different numbers of launches (e.g. 1 vs 4), the wall-clock difference under Proton is misleading. For example, the barrier vs two-stage comparison showed a ~5ms gap under Proton that doesn't exist in uninstrumented runs (~0.01ms real overhead). Always cross-reference Proton wall-clock with `speed_test.py --use_proton=False`.

## Triton benchmark CSV format

Triton's `perf_report` appends ` (Time (ms))` to column names based on `ylabel`. The plotting code strips this suffix via `read_triton_bench_csv()` in `benchmarking/plot-triton-bench.py`.

## vLLM integration

The FMMS sampler is integrated into vLLM on the `feature/fmms-sampler` branch in `~/code/vllm`. Key files:

- `vllm/v1/sample/fmms_sampler.py` — thin wrapper adapting FMMS kernel to vLLM's `SamplerOutput`
- `vllm/envs.py` — `VLLM_USE_FMMS_SAMPLER` and `VLLM_FMMS_PROVIDER` env vars
- `vllm/v1/worker/gpu_model_runner.py` — calls `FMMSSampler` in `sample_tokens()` when enabled

### Benchmarking

End-to-end vLLM benchmarks live in `benchmarking/vllm/`. Key files:

- `Makefile` — `make all` (full sweep, 3 runs) and `make quick` (smoke test, 1 run, `--enforce-eager`). Supports `MODEL=` override for different models.
- `bench-params.json` / `quick-bench-params.json` — sweep parameters (concurrency levels, num_prompts, request_rate)
- `collect_results.py` — reads `summary.csv` from each variant's latest timestamped run, prints summary table (last run only) and per-run breakdown. Usage: `python collect_results.py <model_dir>`
- Results are organized as `<model_slug>/baseline/`, `<model_slug>/fmms-triton/`, `<model_slug>/fmms-flashinfer/`

### `.item()` CPU-GPU synchronization bug

`temperature[0].item()` in `fmms_sampler.py` caused a CPU-GPU sync on every decode step. At concurrency 32, TPOT regressed from 9ms to 18ms. Fix: use `temperature[0]` (scalar tensor) instead. This applies broadly — never call `.item()`, `float()`, `.cpu()`, or `print()` on GPU tensors in the hot path.

### Triton autotuning at runtime

The Triton kernel's `@triton.autotune` originally had `n_hidden_states` in its `key=` parameter. Every unique batch size triggered autotuning (benchmarking all configs). In vLLM, high concurrency produces many unique batch sizes (33, 34, ..., 256), each causing an autotune run **during the benchmark**. This inflated TPOT by 2-10x at concurrency 32+.

**Fix applied**: Replaced `n_hidden_states` with `BLOCK_SIZE_H` in the autotune `key=`, and changed `n_hidden_states` from `tl.constexpr` to a regular runtime int in the kernel signature. `BLOCK_SIZE_H` has only 3 possible values (16, 32, 64), so autotuning runs at most 3 times per (V, D) combination instead of once per unique batch size. All three uses of `n_hidden_states` inside the kernel (`tl.cdiv`, comparison, arithmetic) work fine with runtime values.
