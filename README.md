# FMMS Algorithm: Fused Matrix Multiplication & Sampling

High-performance GPU implementation of fused matrix multiplication + sampling using Triton.
This package provides an efficient kernel for sampling from categorical distributions where logits are computed on-the-fly from matrix multiplication, avoiding the need to materialize the full logit tensor in GPU main memory (GMEM).
The key insight is that in LLM decode workloads, both the matmul and the sampling are memory-bound (the matmul collapses to a matrix-vector product).
By fusing both operations, we avoid round-trips to GPU main memory (GMEM) and speed up the sampling process.

## Features

- **Bandwidth-Efficient**: Fuses matrix multiplication and sampling into a single Triton kernel, avoiding materialization of intermediate logit tensors, and preventing round-trips to GMEM.
- **Exact**: Uses Gumbel-max trick for efficient categorical sampling. No approximations.
- **Flexible**: Supports temperature scaling and multiple samples per hidden state vector.

## Installation

```bash
# Clone the repository
git clone https://github.com/tomasruizt/fused-mm-sample.git
cd fused-mm-sample

# Install the package (assumes you're in a virtual environment)
uv pip install -e ".[dev]"

# Verify installation
python examples/basic_usage.py
```

## Usage

For a complete working example, see [`examples/basic_usage.py`](examples/basic_usage.py).
The basic usage pattern:

```python
from fused_mm_sampling import fused_mm_sample_triton

samples = fused_mm_sample_triton(
    weights=weights,        # [vocab_size, hidden_size]
    hidden_states=hidden_states,  # [n_hidden_states, hidden_size]
    num_samples=1,
    temperature=torch.tensor(1.0, device="cuda"),  # scalar (0-d) CUDA tensor
    seed=42  # Optional: for reproducibility
)
# Returns: [n_hidden_states, num_samples]
```

### Parameters

- **`weights`** (Tensor): Weight matrix of shape `[vocab_size, hidden_size]`
- **`hidden_states`** (Tensor): Hidden states of shape `[n_hidden_states, hidden_size]`
- **`num_samples`** (int): Number of samples to draw per sequence position
- **`temperature`** (Tensor): Scalar (0-d) CUDA tensor for temperature scaling (higher = more random)
- **`seed`** (int, optional): Random seed for reproducibility

### Returns

- Tensor of shape `[n_hidden_states, num_samples]` containing sampled indices

### Algorithm

The FMMS kernel implements the Gumbel-max trick for categorical sampling:

1. **Matrix Multiplication**: Compute a tile of logits = hidden_states @ weights in SRAM
2. **Temperature Scaling**: Scale logits by temperature
3. **Gumbel Noise**: Add Gumbel noise to scaled logits tile
4. **Argmax**: Take argmax within the tile to get samples

The FMMS kernel computes these steps in blocks without materializing the full logit tensor, preventing memory accesses, and relieving the bottleneck on the memory bandwidth.

## Benchmarking Method

The FMMS kernel is benchmarked against competitive baselines used in the vLLM inference pipeline.
The baselines follow a two-step pattern:

1. Compute the full logits via a cuBLAS matmul (`hidden_states @ weights.T`)
2. Sample from those logits.

The three baselines implementing this approach are:

1. **PyTorch Compiled**: matmul + softmax + multinomial. Used in vLLM when top-k and top-p are unset.
2. **`flashinfer:top_k_top_p_sampling_from_logits`**: matmul + FlashInfer's dual-pivot rejection sampling kernel. Used in vLLM when top-k or top-p is set.
3. **`flashinfer:sampling_from_logits`**: matmul + FlashInfer's Gumbel-max kernel. Not used in vLLM, but the fastest baseline sampler benchmarked.

Everything is then torch compiled, so the matmul dispatches to cuBLAS.
Their total runtime (matmul + sampling) is compared against FMMS.

In terms of the **hidden_dim** used in the benchmark, two differnt configs are used: d=4,096 and d=8,192.
These configs are representative for many popular LLMs, see the [analysis of LM head shapes across popular LLMs](findings/lm-head-configurations.md).
The **batch size** (N) ranges from 1 to 256, covering the typical LLM decode regime.

The **arithmetic intensity** of the LM head matmul is approximately equal to the batch size N.
Here N is the batch size, V the vocab size, and D the hidden dimension:

```
FLOPs  = 2 · N · V · D
Bytes  = 2 · D · (V + N)   ≈ 2 · D · V   (since V >> N)

Arithmetic Intensity = FLOPs / Bytes ≈ N
```

The matmul is memory-bound when the arithmetic intensity is below the GPU's ops:byte ratio.
For a detailed derivation, see [`findings/arithmetic-intensity-decode-matmul.md`](findings/arithmetic-intensity-decode-matmul.md).

The following GPUs are used:

| GPU       | HBM Bandwidth (GB/s) | Peak BF16 (TFLOP/s) | Ops:Byte Ratio |
| --------- | -------------------- | ------------------- | -------------- |
| A100-80GB | 2,039                | 312                 | 153            |
| H100      | 3,350                | 989                 | 295            |
| H200      | 4,800                | 989                 | 206            |
| B200      | 8,000                | 2,250               | 281            |
| B300      | 8,000                | 2,250               | 281            |

The ops:byte ratio is peak BF16 TFLOP/s divided by HBM bandwidth (in TB/s).
It determines the crossover point where the matmul transitions from memory-bound to compute-bound.

All benchmarks use PyTorch 2.10.0, CUDA 13.0, and are run on Modal. Results as of 2026-02-11.

## Results

### FMMS vs PyTorch Compiled

This is the sampling path used in vLLM when top-k and top-p are unset.
The relative speedup tables show `baseline_time / fmms_time`, so values > 1.0 mean FMMS is faster.

#### Case 1: d=4,096

| GPU / Batch Size | 1    | 2    | 4    | 8    | 16   | 32   | 64   | 128  | 256  |
| ---------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| A100-80GB        | 1.24 | 1.22 | 1.21 | 1.23 | 1.25 | 1.28 | 1.27 | 1.06 | 0.93 |
| H100             | 1.35 | 1.33 | 1.31 | 1.31 | 1.31 | 1.32 | 1.38 | 1.31 | 1.08 |
| H200             | 1.40 | 1.34 | 1.31 | 1.32 | 1.34 | 1.33 | 1.35 | 1.06 | 0.91 |
| B200             | 1.52 | 1.43 | 1.41 | 1.46 | 1.45 | 1.41 | 1.32 | 1.04 | 0.85 |
| B300             | 1.51 | 1.44 | 1.44 | 1.45 | 1.44 | 1.42 | 1.30 | 1.03 | 0.84 |

#### Case 2: d=8,192

| GPU / Batch Size | 1    | 2    | 4    | 8    | 16   | 32   | 64   | 128  | 256  |
| ---------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| A100-80GB        | 1.20 | 1.11 | 1.09 | 1.10 | 1.12 | 1.20 | 1.27 | 0.98 | 0.84 |
| H100             | 1.22 | 1.21 | 1.20 | 1.20 | 1.20 | 1.21 | 1.27 | 1.17 | 0.90 |
| H200             | 1.23 | 1.21 | 1.19 | 1.20 | 1.20 | 1.24 | 1.27 | 0.94 | 0.78 |
| B200             | 1.37 | 1.31 | 1.27 | 1.25 | 1.26 | 1.26 | 1.25 | 0.97 | 0.71 |
| B300             | 1.34 | 1.31 | 1.26 | 1.25 | 1.27 | 1.27 | 1.26 | 0.97 | 0.71 |

**Results:** FMMS is faster than the baseline across all GPUs and all batch sizes from 1 to 64, with speedups of 20-52% on d=4,096 and 9-37% on d=8,192.
Peak speedup is **1.52x** (B200, N=1, d=4,096).
The advantage of FMMS over the baseline tends to grow larger with better GPUs: 1.28x on A100, 1.38x on H100, 1.40x on H200, 1.52x on B200.

The d=4,096 config consistently shows larger speedups than d=8,192 across all GPUs.
The smaller hidden dimension makes the matmul smaller, taking less overall time and being more memory-bound, so the bandwidth savings from fusion have more impact.
H100 retains its advantage at larger batch sizes than other GPUs: at N=128, it still achieves 1.31x (d=4,096) and 1.17x (d=8,192), while other GPUs drop to ~1.0 or below.
This is likely because H100 has the highest ops:byte ratio (295), keeping the matmul memory-bound longer.

Around batch sizes (128 to 256), the baselines tend to catch up and even outperform FMMS.
One explanation is the following: the baseline spends most of their runtime in the matmul.
This fraction grows with batch size and dominates at N=256.
Here the cuBLAS matmul is very fast and hard to beat with Triton.
Therefore, FMMS becomes less competitive in this regime.
Perhaps a CUDA C++ implementation of FMMS with an optimal matmul would close the gap.
However, in the compute-bound regime, the memory bandwidth savings of FMMS are not as significant anymore.

### FMMS vs `flashinfer:top_k_top_p_sampling_from_logits`

This is the FlashInfer sampling function used in vLLM when top-k or top-p is set.
Nevertheless, I set top-k=-1 and top-p=1.0 to disable top-k and top-p filtering and compare runtimes.

#### Case 1: d=4,096

| GPU / Batch Size | 1    | 2    | 4    | 8    | 16   | 32   | 64   | 128  | 256  |
| ---------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| A100-80GB        | 1.23 | 1.25 | 1.24 | 1.25 | 1.26 | 1.25 | 1.07 | 0.96 | 0.85 |
| H100             | 1.35 | 1.34 | 1.31 | 1.31 | 1.32 | 1.28 | 1.24 | 1.10 | 0.94 |
| H200             | 1.33 | 1.35 | 1.32 | 1.34 | 1.34 | 1.29 | 1.24 | 0.89 | 0.82 |
| B200             | 1.39 | 1.37 | 1.39 | 1.35 | 1.36 | 1.30 | 1.18 | 0.83 | 0.74 |
| B300             | 1.69 | 1.78 | 2.16 | 1.98 | 2.12 | 2.01 | 1.78 | 1.21 | 1.12 |

#### Case 2: d=8,192

| GPU / Batch Size | 1    | 2    | 4    | 8    | 16   | 32   | 64   | 128  | 256  |
| ---------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| A100-80GB        | 1.21 | 1.10 | 1.10 | 1.10 | 1.12 | 1.18 | 1.18 | 0.92 | 0.79 |
| H100             | 1.20 | 1.21 | 1.19 | 1.19 | 1.20 | 1.19 | 1.19 | 1.03 | 0.82 |
| H200             | 1.20 | 1.21 | 1.20 | 1.20 | 1.20 | 1.21 | 1.20 | 0.83 | 0.74 |
| B200             | 1.25 | 1.26 | 1.22 | 1.22 | 1.22 | 1.19 | 1.14 | 0.83 | 0.64 |
| B300             | 1.63 | 1.56 | 1.62 | 1.62 | 1.65 | 1.64 | 1.53 | 1.09 | 0.88 |

**Results:** FMMS is up to **2.16x faster** on d=4,096 (B300, N=4) and up to **1.65x faster** on d=8,192 (B300, N=16).
Across typical decode batch sizes (1 to 64), FMMS is 20-45% faster on d=4,096 and 15-29% faster on d=8,192.
B300 is a striking outlier: 1.6-2.2x speedups while A100-B200 show 1.2-1.4x.
B200 and B300 have identical bandwidth and compute specs, so the cause of this gap is unclear.
B300 has more VRAM (288 GB vs 192 GB), but that has no bearing on runtime since the kernel is bandwidth- and compute-bound, not capacity-bound.

### FMMS vs `flashinfer:sampling_from_logits`

The flashinfer:sampling_from_logits function is not used in vLLM, but its is the fastest FlashInfer function benchmarked in this project.
It also uses a Gumbel-max trick for sampling, but requires pre-materialized logits.

#### Case 1: d=4,096

| GPU / Batch Size | 1    | 2    | 4    | 8    | 16   | 32   | 64   | 128  | 256  |
| ---------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| A100-80GB        | 1.10 | 1.11 | 1.10 | 1.11 | 1.11 | 1.10 | 0.94 | 0.79 | 0.72 |
| H100             | 1.18 | 1.17 | 1.15 | 1.14 | 1.14 | 1.11 | 1.08 | 0.92 | 0.73 |
| H200             | 1.15 | 1.14 | 1.12 | 1.11 | 1.11 | 1.08 | 1.02 | 0.72 | 0.64 |
| B200             | 1.16 | 1.14 | 1.13 | 1.12 | 1.11 | 1.05 | 0.96 | 0.68 | 0.58 |
| B300             | 1.16 | 1.16 | 1.14 | 1.13 | 1.11 | 1.07 | 0.97 | 0.69 | 0.59 |

#### Case 2: d=8,192

| GPU / Batch Size | 1    | 2    | 4    | 8    | 16   | 32   | 64   | 128  | 256  |
| ---------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| A100-80GB        | 1.12 | 1.04 | 1.03 | 1.04 | 1.05 | 1.11 | 1.10 | 0.81 | 0.71 |
| H100             | 1.13 | 1.13 | 1.12 | 1.12 | 1.11 | 1.10 | 1.10 | 0.93 | 0.70 |
| H200             | 1.10 | 1.10 | 1.08 | 1.08 | 1.08 | 1.09 | 1.07 | 0.72 | 0.62 |
| B200             | 1.15 | 1.15 | 1.08 | 1.07 | 1.06 | 1.05 | 1.02 | 0.74 | 0.57 |
| B300             | 1.14 | 1.14 | 1.08 | 1.08 | 1.07 | 1.07 | 1.03 | 0.75 | 0.55 |

**Results:** FMMS is up to **1.18x faster** on d=4,096 (H100, N=1) and up to **1.15x faster** on d=8,192 (B200, N=1).
FMMS tends to outperform `sampling_from_logits` at typical decode batch sizes (1 to 64).
At larger batch sizes (128+), the baseline wins because the cuBLAS matmul.
`sampling_from_logits` is quite competitive, and unlike FMMS, its performance continues to scale with larger batch sizes.
It is not currently used in vLLM -- it should be considered as a drop-in improvement for vLLM's sampling path.

#### Runnnig the Benchmarks

```bash
# Navigate to benchmarking directory
cd benchmarking

# Benchmark all implementations
python speed_test.py

# Benchmark specific implementation
python speed_test.py --name fused-triton
python speed_test.py --name naive-compiled
python speed_test.py --name naive-pt

# Compare performance over many batch sizes
make triton-benchmark
```

## Sampling Quality (GSM8K)

The Gumbel-max trick samples exactly from the categorical distribution. It is mathematically equivalent to softmax + multinomial, not an approximation.
To showcase this, I integrated FMMS in vLLM and ran the GSM8K benchmark (1,319 questions, 0-shot CoT) via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) on Qwen3-1.7B, with answers graded by an LLM judge. Results:

| Variant                 | Accuracy | 95% CI         |
| ----------------------- | -------- | -------------- |
| Baseline (vLLM default) | 89.6%    | [87.9%, 91.2%] |
| FMMS Triton             | 89.4%    | [87.7%, 91.0%] |

The difference is +0.2 percentage points (p=0.776, paired bootstrap), not statistically significant, meaning that FMMS does not degrade model accuracy.
For full methodology and pairwise comparisons, see [`benchmarking/vllm/README.md`](benchmarking/vllm/README.md#quality-evaluation-gsm8k).

## Profiling

All profiling scripts are located in the `benchmarking/` directory.

### Memory Profiling

```bash
cd benchmarking
make profile-mem
```

This will generate a memory snapshot and HTML visualization in `benchmarking/memory/`.

### NVIDIA Nsight Compute Profiling

```bash
cd benchmarking

# Profile fused Triton kernel
make ncu-profile-fused-triton

# Profile naive compiled implementation
make ncu-profile-naive-compiled
```

### NVIDIA Nsight Systems Profiling

```bash
cd benchmarking

# Profile fused Triton kernel
make nsight-profile-fused-triton

# Profile naive compiled implementation
make nsight-profile-naive-compiled
```

## Development

### Development Environment

The dev dependencies permit running the scripts in the `benchmarking/` directory. To install them, run:

```bash
uv pip install -e ".[dev]"
```

### Modal Setup
The experiments involving many differnt GPUs were run on Modal. To install and login to Modal:

```bash
uv pip install modal
modal setup
```

Run the speed-test on modal:

```bash
make modal-speed-test
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to create an issue or submit a pull request.
