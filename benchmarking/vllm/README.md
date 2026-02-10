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

## Quality evaluation (GSM8K)

We use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with the `gsm8k_cot_zeroshot` task to verify that FMMS sampling does not degrade model quality compared to the vLLM baseline.

### Expected performance

The Qwen3-1.7B-Base model scores **75.44%** on GSM8K (4-shot CoT) according to the [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) (Table 8). An independent evaluation by [MathGPT](https://resources.mathgpt.ai/2025/06/03/are-the-best-open-source-models-qwen-phi-nvidia-deepseek-robust-mathematical-reasoners-insights-from-large-scale-evaluations/) reports **89.84%** on GSM1K (a 1,000-problem test set derived from GSM8K) with hybrid reasoning (thinking mode).

However, reproducing these scores with lm-evaluation-harness is non-trivial. A [known issue](https://github.com/EleutherAI/lm-evaluation-harness/issues/3129) documents significant discrepancies: Qwen3-32B claims 92% in the paper but lm-eval reproduces only ~70% (without chat template) or ~22% (with chat template). The strict-match filter expects "The answer is X." format, which Qwen3's thinking mode often does not produce, yielding 0% strict-match even when the model reasons correctly.

### Eval setup

- **Task**: `gsm8k_cot_zeroshot` (0-shot, generate-until)
- **Sampling**: `temperature=0.6`, `top_p=0.95`, `top_k=20` (Qwen3-1.7B `generation_config.json` defaults)
- **Max tokens**: 16384 (for thinking chains)
- **vLLM flags**: `--max-model-len 32768 --reasoning-parser qwen3`
- **Concurrency**: 32

### Baseline result (lm-eval regex)

| Filter | Accuracy | Stderr |
|---|---|---|
| flexible-extract | 35.33% | ±1.32% |
| strict-match | 0.00% | ±0.00% |

The 0% strict-match and low flexible-extract score are consistent with the discrepancies documented in the lm-eval issue above. Manual inspection shows the model outputs correct answers in `\boxed{X}` format, which the regex filters don't match.

### LLM-as-a-judge re-scoring

Since lm-eval's regex scoring is unreliable for thinking-mode outputs, `judge_eval.py` re-evaluates the saved JSONL samples using an LLM judge (`qwen/qwen3-30b-a3b-instruct-2507` via OpenRouter). For each sample, the judge sees the reference answer and the last 500 characters of the model's response, then replies YES or NO.

| Variant | Accuracy (judge) | 95% CI |
|---|---|---|
| Baseline | 1182/1319 (89.6%) | [87.9%, 91.2%] |
| Baseline (FMMS params) | 1166/1319 (88.4%) | [86.7%, 90.1%] |
| FMMS Triton | 1179/1319 (89.4%) | [87.7%, 91.0%] |

"Baseline (FMMS params)" uses `top_k=-1, top_p=1.0` (FMMS-compatible sampling) to isolate the effect of removing top-k/top-p filtering from the effect of the fused kernel itself.

**Pairwise comparisons** (paired bootstrap, 10,000 samples):

| Comparison | Diff | 95% CI | p-value |
|---|---|---|---|
| Baseline vs Baseline (FMMS params) | +1.2pp | [-0.1pp, +2.5pp] | 0.080 (n.s.) |
| Baseline vs FMMS Triton | +0.2pp | [-1.1pp, +1.5pp] | 0.776 (n.s.) |
| Baseline (FMMS params) vs FMMS Triton | -1.0pp | [-2.4pp, +0.5pp] | 0.181 (n.s.) |

No statistically significant differences between any variants. The baseline score (89.6%) is consistent with the 89.84% reported by [MathGPT](https://resources.mathgpt.ai/2025/06/03/are-the-best-open-source-models-qwen-phi-nvidia-deepseek-robust-mathematical-reasoners-insights-from-large-scale-evaluations/) on GSM1K with hybrid reasoning. FMMS Triton (89.4%) is within noise of the baseline, confirming the fused kernel does not degrade model quality.

### Eval runtime

Total wall-clock time for each lm-eval run (1,319 GSM8K questions, concurrency 32, single run):

| Variant | Sampling params | Runtime |
|---|---|---|
| Baseline | temp=0.6, top_p=0.95, top_k=20 | 29.7 min |
| Baseline (FMMS params) | temp=0.6, top_p=1.0, top_k=-1 | 23.4 min |
| FMMS Triton | temp=0.6, top_p=1.0, top_k=-1 | 33.1 min |

Runtimes vary across runs due to stochastic generation lengths (thinking chains). These are single-run measurements and should not be interpreted as throughput benchmarks — see the [TPOT results](#results) above for controlled latency comparisons.

### Reproducing

```bash
# Run eval + judge (judge runs automatically after lm-eval):
make -C benchmarking/vllm eval-baseline
make -C benchmarking/vllm eval-baseline-fmms-params
make -C benchmarking/vllm eval-fmms-triton

# Re-run judge on existing samples (requires OPENROUTER_API_KEY):
make -C benchmarking/vllm judge
```

Results are saved in `eval/baseline/`, `eval/baseline-fmms-params/`, and `eval/fmms-triton/`. Each eval run produces a timestamped samples file (e.g. `samples_gsm8k_cot_zeroshot_<timestamp>.jsonl`) and a matching judge file (`judge_gsm8k_cot_zeroshot_<timestamp>.jsonl`). The judge supports resume — re-running skips already-judged samples.

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
