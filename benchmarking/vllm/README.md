# vLLM Benchmark Results

## Setup

- **Model**: Qwen/Qwen3-1.7B
- **GPU**: NVIDIA RTX 3090
- **vLLM**: v1 engine, `--max-model-len 1024`, `--no-enable-prefix-caching`
- **Dataset**: AI-MO/aimo-validation-aime (math reasoning), `--hf-output-len 256`
- **Sampling**: `temperature=0.6`, `top_k=-1`, `top_p=1.0`
- **Tool**: `vllm bench sweep serve` with `--num-runs 1`
- **Sweep params**: `num_prompts = 10 * concurrency`, `request_rate = concurrency`

## Results

Three variants tested:
- **Baseline**: vLLM default (`compute_logits` via cuBLAS + FlashInfer sampler)
- **FMMS Triton**: Fused matmul+sampling Triton kernel (`fused-triton` provider)
- **FMMS FlashInfer**: Unfused control — matmul + FlashInfer top-k/top-p sampling through FMMS integration path (`flashinfer:top_k_top_p_sampling_from_logits` provider)

### Median TPOT (ms)

| Concurrency | Baseline | FMMS Triton | FMMS FlashInfer |
|---|---|---|---|
| 1 | 5.24 | 5.11 | 5.30 |
| 32 | 8.95 | 8.79 | 8.93 |

## Analysis

All three variants perform equivalently at both low and high concurrency. FMMS Triton matches baseline TPOT within noise (~1%).

An earlier version of the integration used `temperature[0].item()` to extract a scalar from the per-request temperature tensor. This `.item()` call caused a CPU-GPU synchronization on every decode step, which compounded at high concurrency (TPOT was 18.66ms vs 8.98ms baseline at concurrency 32). Replacing it with `temperature[0]` (keeping the value as a scalar tensor) eliminated the regression.

## Evidence

- Server logs: `logs/`
- Raw JSON results: `baseline/`, `fmms-triton/`, `fmms-flashinfer/`
- Sweep parameters: `bench-params.json`

## Reproducing

```bash
# Run all three sweeps
make -C findings/vllm-bench-results all

# Or individually
make -C findings/vllm-bench-results baseline
make -C findings/vllm-bench-results fmms-triton
make -C findings/vllm-bench-results fmms-flashinfer
```

Requires vLLM at `~/code/vllm` on the `feature/fmms-sampler` branch with `fused-mm-sample` installed in its venv.
