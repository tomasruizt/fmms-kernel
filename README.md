# FMMS Algorithm: Fused Matrix Multiplication & Sampling

High-performance GPU implementation of fused matrix multiplication + sampling using Triton. This package provides an efficient kernel for sampling from categorical distributions where logits are computed on-the-fly from matrix multiplication, avoiding the need to materialize the full logit tensor in GPU main memory (HBM).

## Features

- **Memory Efficient**: Fuses matrix multiplication and sampling into a single Triton kernel, avoiding materialization of large intermediate logit tensors
- **GPU Optimized**: Uses Gumbel-max trick for efficient categorical sampling on GPUs
- **Flexible**: Supports temperature scaling, large batch sizes, and multiple samples per hidden state vector.

## Installation
We use uv, but its not strictly necessary.

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

```bash
python examples/basic_usage.py
```

The basic usage pattern:

```python
from fused_mm_sampling import fused_mm_sample_triton

samples = fused_mm_sample_triton(
    weights=weights,        # [hidden_size, vocab_size]
    hidden_states=hidden_states,  # [n_hidden_states, hidden_size]
    num_samples=1,
    temperature=1.0,
    seed=42  # Optional: for reproducibility
)
# Returns: [n_hidden_states, num_samples]
```

### Parameters

- **`weights`** (Tensor): Weight matrix of shape `[hidden_size, vocab_size]`
- **`hidden_states`** (Tensor): Hidden states of shape `[n_hidden_states, hidden_size]`
- **`num_samples`** (int): Number of samples to draw per sequence position
- **`temperature`** (float): Temperature for sampling (higher = more random)
- **`seed`** (int, optional): Random seed for reproducibility

### Returns

- Tensor of shape `[n_hidden_states, num_samples]` containing sampled indices

## How It Works

The package implements the Gumbel-max trick for categorical sampling:

1. **Matrix Multiplication**: Compute logits = hidden_states @ weights
2. **Temperature Scaling**: Scale logits by temperature
3. **Gumbel Noise**: Add Gumbel noise to scaled logits
4. **Argmax**: Take argmax to get samples

The fused implementations compute these steps in blocks without materializing the full logit tensor, saving memory and improving performance for large vocabulary sizes.

## Benchmarking

Run speed benchmarks:

```bash
# Navigate to benchmarking directory
cd benchmarking

# Benchmark all implementations
python speed_test.py

# Benchmark specific implementation
python speed_test.py --name fused-triton
python speed_test.py --name naive-compiled
python speed_test.py --name naive-pt
```

### FMMS vs Naive PyTorch Compiled

vLLM's default (baseline) sampling path uses plain PyTorch ops (`matmul` + `softmax` + `multinomial`). The tables below show FMMS performance relative to this baseline compiled with `torch.compile`. Values > 1.0 mean FMMS is faster.

**Llama 3 8B (V=128,256, d=4,096):**

| GPU / Batch Size | 1     | 2     | 4     | 8     | 16    | 32    | 64    | 128   | 256   |
| ---------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| A100-80GB        | 1.240 | 1.219 | 1.213 | 1.229 | 1.255 | 1.278 | 1.267 | 1.056 | 0.932 |
| H100             | 1.354 | 1.327 | 1.308 | 1.306 | 1.314 | 1.323 | 1.378 | 1.312 | 1.084 |
| H200             | 1.400 | 1.340 | 1.315 | 1.317 | 1.342 | 1.332 | 1.346 | 1.058 | 0.906 |
| B200             | 1.524 | 1.432 | 1.414 | 1.458 | 1.450 | 1.411 | 1.318 | 1.037 | 0.851 |
| B300             | 1.507 | 1.439 | 1.438 | 1.454 | 1.443 | 1.419 | 1.302 | 1.033 | 0.845 |

**Llama 3 70B (V=128,256, d=8,192):**

| GPU / Batch Size | 1     | 2     | 4     | 8     | 16    | 32    | 64    | 128   | 256   |
| ---------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| A100-80GB        | 1.199 | 1.111 | 1.091 | 1.102 | 1.119 | 1.197 | 1.267 | 0.979 | 0.838 |
| H100             | 1.219 | 1.208 | 1.198 | 1.200 | 1.201 | 1.213 | 1.272 | 1.166 | 0.904 |
| H200             | 1.230 | 1.207 | 1.191 | 1.200 | 1.204 | 1.240 | 1.273 | 0.941 | 0.782 |
| B200             | 1.370 | 1.313 | 1.269 | 1.255 | 1.265 | 1.264 | 1.253 | 0.969 | 0.715 |
| B300             | 1.344 | 1.309 | 1.262 | 1.255 | 1.271 | 1.272 | 1.257 | 0.975 | 0.708 |

FMMS is **~30% faster** on the 8B model and **~20% faster** on the 70B model across typical decode batch sizes (1--64) on datacenter GPUs. At large batch sizes (128--256), the matmul becomes compute-bound and the unfused baseline with cuBLAS catches up.

### FMMS vs FlashInfer (As Used in vLLM)

`top_k_top_p_sampling_from_logits` is the one used for top-k/top-p sampling in vLLM.
Both require pre-materialized logits, so their runtime includes the matmul.
The tables below show FMMS performance relative to `top_k_top_p_sampling_from_logits` (baseline = 1.0).

**Llama 3 8B (V=128,256, d=4,096):**

| GPU / Batch Size | 1     | 2     | 4     | 8     | 16    | 32    | 64    | 128   | 256   |
| ---------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| A100-80GB        | 1.230 | 1.251 | 1.240 | 1.253 | 1.265 | 1.250 | 1.075 | 0.961 | 0.855 |
| H100             | 1.346 | 1.339 | 1.311 | 1.309 | 1.320 | 1.276 | 1.240 | 1.098 | 0.943 |
| H200             | 1.332 | 1.352 | 1.319 | 1.340 | 1.341 | 1.288 | 1.238 | 0.887 | 0.821 |
| B200             | 1.390 | 1.373 | 1.394 | 1.352 | 1.362 | 1.297 | 1.180 | 0.830 | 0.742 |
| B300             | 1.685 | 1.778 | 2.160 | 1.976 | 2.124 | 2.010 | 1.779 | 1.211 | 1.123 |

**Llama 3 70B (V=128,256, d=8,192):**

| GPU / Batch Size | 1     | 2     | 4     | 8     | 16    | 32    | 64    | 128   | 256   |
| ---------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| A100-80GB        | 1.205 | 1.098 | 1.104 | 1.105 | 1.119 | 1.175 | 1.178 | 0.917 | 0.793 |
| H100             | 1.202 | 1.205 | 1.192 | 1.192 | 1.195 | 1.185 | 1.191 | 1.035 | 0.818 |
| H200             | 1.200 | 1.214 | 1.198 | 1.204 | 1.201 | 1.208 | 1.197 | 0.828 | 0.745 |
| B200             | 1.250 | 1.263 | 1.222 | 1.220 | 1.216 | 1.194 | 1.142 | 0.828 | 0.642 |
| B300             | 1.633 | 1.559 | 1.625 | 1.620 | 1.653 | 1.635 | 1.532 | 1.089 | 0.883 |

FMMS is **between 20% and 45% faster** than the `top_k_top_p` kernel on the 8B model, and **between 15% and 29% faster** on the 70B model on datacenter GPUs at typical decode batch sizes (1--64).

### FMMS vs FlashInfer (Fastest Kernel)

FlashInfer's `sampling_from_logits` is a lean Gumbel-max kernel (no top-k/top-p filtering) that is the fastest unfused sampler in our benchmarks. The tables below show FMMS performance relative to `sampling_from_logits` (baseline = 1.0).

**Llama 3 8B (V=128,256, d=4,096):**

| GPU / Batch Size | 1     | 2     | 4     | 8     | 16    | 32    | 64    | 128   | 256   |
| ---------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| A100-80GB        | 1.103 | 1.110 | 1.103 | 1.107 | 1.113 | 1.105 | 0.935 | 0.786 | 0.720 |
| H100             | 1.179 | 1.173 | 1.150 | 1.144 | 1.142 | 1.115 | 1.076 | 0.920 | 0.729 |
| H200             | 1.154 | 1.143 | 1.116 | 1.110 | 1.114 | 1.078 | 1.018 | 0.716 | 0.637 |
| B200             | 1.164 | 1.142 | 1.128 | 1.117 | 1.114 | 1.053 | 0.964 | 0.680 | 0.582 |
| B300             | 1.162 | 1.155 | 1.136 | 1.126 | 1.113 | 1.070 | 0.973 | 0.693 | 0.586 |

**Llama 3 70B (V=128,256, d=8,192):**

| GPU / Batch Size | 1     | 2     | 4     | 8     | 16    | 32    | 64    | 128   | 256   |
| ---------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| A100-80GB        | 1.119 | 1.037 | 1.033 | 1.038 | 1.047 | 1.106 | 1.097 | 0.814 | 0.714 |
| H100             | 1.134 | 1.129 | 1.118 | 1.117 | 1.111 | 1.103 | 1.104 | 0.934 | 0.700 |
| H200             | 1.103 | 1.100 | 1.083 | 1.083 | 1.080 | 1.092 | 1.075 | 0.724 | 0.622 |
| B200             | 1.151 | 1.147 | 1.082 | 1.069 | 1.063 | 1.054 | 1.024 | 0.736 | 0.572 |
| B300             | 1.143 | 1.141 | 1.079 | 1.079 | 1.073 | 1.069 | 1.035 | 0.746 | 0.553 |

FMMS is **between 5% and 19% faster** than `sampling_from_logits` on datacenter GPUs at typical decode batch sizes (1--32), despite `sampling_from_logits` being a highly optimized unfused Gumbel-max kernel. At larger batch sizes (128+), the unfused kernel wins because cuBLAS dominates and the fusion overhead grows.

### H100 Absolute Performance

The following tables show absolute execution times (in milliseconds) on H100.

**Llama 3 8B (V=128,256, d=4,096):**

| Algorithm / Batch Size                      | 1     | 2     | 4     | 8     | 16    | 32    | 64    | 128   | 256   |
| ------------------------------------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| FMMS (Triton)                               | 0.367 | 0.368 | 0.375 | 0.379 | 0.386 | 0.402 | 0.443 | 0.574 | 0.933 |
| Naive PyTorch Compiled                      | 0.497 | 0.488 | 0.491 | 0.495 | 0.508 | 0.532 | 0.610 | 0.753 | 1.011 |
| flashinfer:top_k_top_p_sampling_from_logits | 0.494 | 0.493 | 0.492 | 0.496 | 0.510 | 0.513 | 0.549 | 0.630 | 0.880 |
| flashinfer:sampling_from_logits             | 0.433 | 0.432 | 0.432 | 0.434 | 0.441 | 0.448 | 0.477 | 0.528 | 0.681 |

**Llama 3 70B (V=128,256, d=8,192):**

| Algorithm / Batch Size                      | 1     | 2     | 4     | 8     | 16    | 32    | 64    | 128   | 256   |
| ------------------------------------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| FMMS (Triton)                               | 0.709 | 0.711 | 0.719 | 0.722 | 0.730 | 0.744 | 0.781 | 0.960 | 1.550 |
| Naive PyTorch Compiled                      | 0.864 | 0.859 | 0.861 | 0.866 | 0.877 | 0.903 | 0.993 | 1.120 | 1.402 |
| flashinfer:top_k_top_p_sampling_from_logits | 0.851 | 0.857 | 0.857 | 0.860 | 0.873 | 0.882 | 0.930 | 0.993 | 1.268 |
| flashinfer:sampling_from_logits             | 0.803 | 0.803 | 0.804 | 0.806 | 0.811 | 0.821 | 0.862 | 0.897 | 1.086 |

*All benchmarks: PyTorch 2.9.1, CUDA 12.8, run on Modal. Data as of 2026-02-11.*

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

## Project Structure

```
fused-mm-sample/
├── src/
│   └── fused_mm_sampling/
│       └── core.py              # Core implementation
├── examples/
│   └── basic_usage.py           # Basic usage example
└── benchmarking/
    ├── Makefile                 # Profiling commands
    ├── speed_test.py            # Speed bench script
    ├── profile-mem.py           # Memory profiling script
    └── verify-fused-impl.ipynb  # Verification notebook
```

## Development

### Setting Up Development Environment
The dev dependencies permit running the scripts in the `benchmarking/` directory. To install them, run:

```bash
uv pip install -e ".[dev]"
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Modal Setup

Install and login to Modal:

```bash
uv pip install modal
modal setup
```

Run the modal example, or speed-test:

```bash
make modal-example
make modal-speed-test
```
