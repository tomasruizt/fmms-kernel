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

The two baselines implementing this appraoch are:

1. PyTorch Compiled: matmul + softmax + multinomial (all torch.compiled)
2. FlashInfer functions: matmul + flashinfer sampling (all torch.compiled)

The matmul is highly optimized (cuBLAS dispatched through `torch.compile`), and the sampling step uses either `torch.compiled` PyTorch ops or FlashInfer's CUDA sampling kernels.
The first baseline is used in vLLM sampling when top-p and top-k are not set, while the flashinfer functionss are used when either of them is set.
Their total runtime (matmul + sampling) is compared against FMMS.

In terms of the **hidden_dim** used in the benchmark, two differnt configs are used: d=4,096 and d=8,192.
These configs are representative for many popular LLMs, see the [analysis of LM head shapes across popular LLMs](findings/lm-head-configurations.md).
The **batch size** (N) ranges from 1 to 256, covering the typical LLM decode regime.

All benchmarks use PyTorch 2.10.0, CUDA 13.0, and are run on Modal. Results as of 2026-02-11.

## Results

### FMMS vs PyTorch Compiled

The relative speedup tables show `baseline_time / fmms_time`, so values > 1.0 mean FMMS is faster.

#### Case 1: d=4,096

| GPU / Batch Size | 1     | 2     | 4     | 8     | 16    | 32    | 64    | 128   | 256   |
| ---------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| A100-80GB        | 1.24 | 1.22 | 1.21 | 1.23 | 1.25 | 1.28 | 1.27 | 1.06 | 0.93 |
| H100             | 1.35 | 1.33 | 1.31 | 1.31 | 1.31 | 1.32 | 1.38 | 1.31 | 1.08 |
| H200             | 1.40 | 1.34 | 1.31 | 1.32 | 1.34 | 1.33 | 1.35 | 1.06 | 0.91 |
| B200             | 1.52 | 1.43 | 1.41 | 1.46 | 1.45 | 1.41 | 1.32 | 1.04 | 0.85 |
| B300             | 1.51 | 1.44 | 1.44 | 1.45 | 1.44 | 1.42 | 1.30 | 1.03 | 0.84 |

#### Case 2: d=8,192

| GPU / Batch Size | 1     | 2     | 4     | 8     | 16    | 32    | 64    | 128   | 256   |
| ---------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| A100-80GB        | 1.20 | 1.11 | 1.09 | 1.10 | 1.12 | 1.20 | 1.27 | 0.98 | 0.84 |
| H100             | 1.22 | 1.21 | 1.20 | 1.20 | 1.20 | 1.21 | 1.27 | 1.17 | 0.90 |
| H200             | 1.23 | 1.21 | 1.19 | 1.20 | 1.20 | 1.24 | 1.27 | 0.94 | 0.78 |
| B200             | 1.37 | 1.31 | 1.27 | 1.25 | 1.26 | 1.26 | 1.25 | 0.97 | 0.71 |
| B300             | 1.34 | 1.31 | 1.26 | 1.25 | 1.27 | 1.27 | 1.26 | 0.97 | 0.71 |

**Results:** FMMS is up to **1.52x faster** on d=4,096 (B200, N=1) and up to **1.37x faster** on d=8,192 (B200, N=1).
Across typical decode batch sizes (1 to 64), FMMS is ~30% faster on d=4,096 and ~20% faster on d=8,192.
At large batch sizes (128 to 256), the matmul becomes compute-bound and the unfused baseline with cuBLAS catches up.

### FMMS vs FlashInfer Functions

#### Case 1: d=4,096

| GPU / Batch Size | 1     | 2     | 4     | 8     | 16    | 32    | 64    | 128   | 256   |
| ---------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| A100-80GB        | 1.23 | 1.25 | 1.24 | 1.25 | 1.26 | 1.25 | 1.07 | 0.96 | 0.85 |
| H100             | 1.35 | 1.34 | 1.31 | 1.31 | 1.32 | 1.28 | 1.24 | 1.10 | 0.94 |
| H200             | 1.33 | 1.35 | 1.32 | 1.34 | 1.34 | 1.29 | 1.24 | 0.89 | 0.82 |
| B200             | 1.39 | 1.37 | 1.39 | 1.35 | 1.36 | 1.30 | 1.18 | 0.83 | 0.74 |
| B300             | 1.69 | 1.78 | 2.16 | 1.98 | 2.12 | 2.01 | 1.78 | 1.21 | 1.12 |

#### Case 2: d=8,192

| GPU / Batch Size | 1     | 2     | 4     | 8     | 16    | 32    | 64    | 128   | 256   |
| ---------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| A100-80GB        | 1.21 | 1.10 | 1.10 | 1.10 | 1.12 | 1.18 | 1.18 | 0.92 | 0.79 |
| H100             | 1.20 | 1.21 | 1.19 | 1.19 | 1.20 | 1.19 | 1.19 | 1.03 | 0.82 |
| H200             | 1.20 | 1.21 | 1.20 | 1.20 | 1.20 | 1.21 | 1.20 | 0.83 | 0.74 |
| B200             | 1.25 | 1.26 | 1.22 | 1.22 | 1.22 | 1.19 | 1.14 | 0.83 | 0.64 |
| B300             | 1.63 | 1.56 | 1.62 | 1.62 | 1.65 | 1.64 | 1.53 | 1.09 | 0.88 |

**Results:** FMMS is up to **2.16x faster** on d=4,096 (B300, N=4) and up to **1.65x faster** on d=8,192 (B300, N=16).
Across typical decode batch sizes (1 to 64), FMMS is 20-45% faster on d=4,096 and 15-29% faster on d=8,192.

### FMMS vs FlashInfer (Fastest Kernel)

FlashInfer's `sampling_from_logits` is a lean Gumbel-max kernel (no top-k/top-p filtering) that is the fastest unfused sampler in our benchmarks. The tables below show FMMS performance relative to `sampling_from_logits` (baseline = 1.0).

#### Case 1: d=4,096

| GPU / Batch Size | 1     | 2     | 4     | 8     | 16    | 32    | 64    | 128   | 256   |
| ---------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| A100-80GB        | 1.10 | 1.11 | 1.10 | 1.11 | 1.11 | 1.10 | 0.94 | 0.79 | 0.72 |
| H100             | 1.18 | 1.17 | 1.15 | 1.14 | 1.14 | 1.11 | 1.08 | 0.92 | 0.73 |
| H200             | 1.15 | 1.14 | 1.12 | 1.11 | 1.11 | 1.08 | 1.02 | 0.72 | 0.64 |
| B200             | 1.16 | 1.14 | 1.13 | 1.12 | 1.11 | 1.05 | 0.96 | 0.68 | 0.58 |
| B300             | 1.16 | 1.16 | 1.14 | 1.13 | 1.11 | 1.07 | 0.97 | 0.69 | 0.59 |

#### Case 2: d=8,192

| GPU / Batch Size | 1     | 2     | 4     | 8     | 16    | 32    | 64    | 128   | 256   |
| ---------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| A100-80GB        | 1.12 | 1.04 | 1.03 | 1.04 | 1.05 | 1.11 | 1.10 | 0.81 | 0.71 |
| H100             | 1.13 | 1.13 | 1.12 | 1.12 | 1.11 | 1.10 | 1.10 | 0.93 | 0.70 |
| H200             | 1.10 | 1.10 | 1.08 | 1.08 | 1.08 | 1.09 | 1.07 | 0.72 | 0.62 |
| B200             | 1.15 | 1.15 | 1.08 | 1.07 | 1.06 | 1.05 | 1.02 | 0.74 | 0.57 |
| B300             | 1.14 | 1.14 | 1.08 | 1.08 | 1.07 | 1.07 | 1.03 | 0.75 | 0.55 |

**Results:** FMMS is up to **1.18x faster** on d=4,096 (H100, N=1) and up to **1.15x faster** on d=8,192 (B200, N=1).
FMMS is 5-19% faster at typical decode batch sizes (1 to 32), despite `sampling_from_logits` being a highly optimized unfused Gumbel-max kernel.
At larger batch sizes (128+), the unfused kernel wins because cuBLAS dominates and the fusion overhead grows.

### H100 Absolute Performance

The following tables show absolute execution times (in milliseconds) on H100.

#### Case 1: d=4,096

| Algorithm / Batch Size                      | 1     | 2     | 4     | 8     | 16    | 32    | 64    | 128   | 256   |
| ------------------------------------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| FMMS (Triton)                               | 0.37 | 0.37 | 0.38 | 0.38 | 0.39 | 0.40 | 0.44 | 0.57 | 0.93 |
| Naive PyTorch Compiled                      | 0.50 | 0.49 | 0.49 | 0.49 | 0.51 | 0.53 | 0.61 | 0.75 | 1.01 |
| flashinfer:top_k_top_p_sampling_from_logits | 0.49 | 0.49 | 0.49 | 0.50 | 0.51 | 0.51 | 0.55 | 0.63 | 0.88 |
| flashinfer:sampling_from_logits             | 0.43 | 0.43 | 0.43 | 0.43 | 0.44 | 0.45 | 0.48 | 0.53 | 0.68 |

#### Case 2: d=8,192

| Algorithm / Batch Size                      | 1     | 2     | 4     | 8     | 16    | 32    | 64    | 128   | 256   |
| ------------------------------------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| FMMS (Triton)                               | 0.71 | 0.71 | 0.72 | 0.72 | 0.73 | 0.74 | 0.78 | 0.96 | 1.55 |
| Naive PyTorch Compiled                      | 0.86 | 0.86 | 0.86 | 0.87 | 0.88 | 0.90 | 0.99 | 1.12 | 1.40 |
| flashinfer:top_k_top_p_sampling_from_logits | 0.85 | 0.86 | 0.86 | 0.86 | 0.87 | 0.88 | 0.93 | 0.99 | 1.27 |
| flashinfer:sampling_from_logits             | 0.80 | 0.80 | 0.80 | 0.81 | 0.81 | 0.82 | 0.86 | 0.90 | 1.09 |



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
