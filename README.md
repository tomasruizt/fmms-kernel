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
| L4               | 0.948 | 0.974 | 0.977 | 0.979 | 0.994 | 1.071 | 1.161 | 1.500 | 1.553 |
| A100-80GB        | 1.236 | 1.218 | 1.218 | 1.235 | 1.257 | 1.294 | 1.307 | 1.013 | 0.927 |
| H100             | 1.355 | 1.329 | 1.310 | 1.312 | 1.311 | 1.317 | 1.385 | 1.299 | 1.076 |
| H200             | 1.384 | 1.337 | 1.318 | 1.329 | 1.367 | 1.339 | 1.348 | 1.044 | 0.904 |
| B200             | 1.523 | 1.455 | 1.436 | 1.449 | 1.450 | 1.421 | 1.299 | 0.984 | 0.851 |

**Llama 3 70B (V=128,256, d=8,192):**

| GPU / Batch Size | 1     | 2     | 4     | 8     | 16    | 32    | 64    | 128   | 256   |
| ---------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| L4               | 0.914 | 0.954 | 0.955 | 0.959 | 0.963 | 1.050 | 1.084 | 1.251 | 1.161 |
| A100-80GB        | 1.204 | 1.125 | 1.131 | 1.139 | 1.159 | 1.234 | 1.249 | 0.875 | 0.806 |
| H100             | 1.209 | 1.198 | 1.194 | 1.194 | 1.203 | 1.210 | 1.269 | 1.175 | 0.854 |
| H200             | 1.238 | 1.211 | 1.202 | 1.206 | 1.265 | 1.252 | 1.279 | 0.918 | 0.749 |
| B200             | 1.305 | 1.258 | 1.257 | 1.239 | 1.266 | 1.256 | 1.245 | 0.906 | 0.730 |

FMMS is **~30% faster** on the 8B model and **~20% faster** on the 70B model across typical decode batch sizes (1--64) on datacenter GPUs. At large batch sizes (128--256), the matmul becomes compute-bound and the unfused baseline with cuBLAS catches up. On L4, FMMS is slower at small batches but faster at large ones.

### FMMS vs FlashInfer (As Used in vLLM)

`top_k_top_p_sampling_from_logits` is the one used for top-k/top-p sampling in vLLM.
Both require pre-materialized logits, so their runtime includes the matmul.
The tables below show FMMS performance relative to `top_k_top_p_sampling_from_logits` (baseline = 1.0).

**Llama 3 8B (V=128,256, d=4,096):**

| GPU / Batch Size | 1     | 2     | 4     | 8     | 16    | 32    | 64    | 128   | 256   |
| ---------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| L4               | 0.987 | 1.021 | 1.023 | 1.022 | 1.038 | 1.108 | 1.223 | 1.501 | 1.456 |
| A100-80GB        | 1.249 | 1.248 | 1.236 | 1.264 | 1.287 | 1.299 | 1.264 | 1.007 | 0.919 |
| H100             | 1.331 | 1.352 | 1.347 | 1.352 | 1.333 | 1.307 | 1.351 | 1.216 | 1.043 |
| H200             | 1.361 | 1.349 | 1.361 | 1.394 | 1.407 | 1.346 | 1.339 | 0.977 | 0.906 |
| B200             | 1.371 | 1.445 | 1.387 | 1.427 | 1.395 | 1.357 | 1.207 | 0.905 | 0.864 |

**Llama 3 70B (V=128,256, d=8,192):**

| GPU / Batch Size | 1     | 2     | 4     | 8     | 16    | 32    | 64    | 128   | 256   |
| ---------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| L4               | 0.931 | 0.974 | 0.979 | 0.981 | 0.987 | 1.072 | 1.116 | 1.255 | 1.139 |
| A100-80GB        | 1.201 | 1.148 | 1.150 | 1.157 | 1.168 | 1.232 | 1.227 | 0.896 | 0.799 |
| H100             | 1.194 | 1.231 | 1.210 | 1.218 | 1.211 | 1.197 | 1.249 | 1.122 | 0.837 |
| H200             | 1.242 | 1.247 | 1.238 | 1.227 | 1.285 | 1.250 | 1.266 | 0.879 | 0.754 |
| B200             | 1.206 | 1.242 | 1.251 | 1.210 | 1.238 | 1.222 | 1.177 | 0.844 | 0.733 |

FMMS is **between 20% and 45% faster** than the `top_k_top_p` kernel on the 8B model, and **between 15% and 29% faster** on the 70B model on datacenter GPUs at typical decode batch sizes (1--64). On L4, FMMS is faster at batch sizes 32+ (up to 1.5x on the 8B model).

### FMMS vs FlashInfer (Fastest Kernel)

FlashInfer's `sampling_from_logits` is a lean Gumbel-max kernel (no top-k/top-p filtering) that is the fastest unfused sampler in our benchmarks. The tables below show FMMS performance relative to `sampling_from_logits` (baseline = 1.0).

**Llama 3 8B (V=128,256, d=4,096):**

| GPU / Batch Size | 1     | 2     | 4     | 8     | 16    | 32    | 64    | 128   | 256   |
| ---------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| L4               | 0.934 | 0.963 | 0.964 | 0.965 | 0.975 | 1.037 | 1.067 | 1.171 | 1.067 |
| A100-80GB        | 1.108 | 1.108 | 1.102 | 1.107 | 1.114 | 1.114 | 1.010 | 0.781 | 0.724 |
| H100             | 1.188 | 1.181 | 1.159 | 1.157 | 1.144 | 1.115 | 1.084 | 0.909 | 0.730 |
| H200             | 1.161 | 1.149 | 1.125 | 1.128 | 1.152 | 1.099 | 1.038 | 0.721 | 0.669 |
| B200             | 1.150 | 1.129 | 1.107 | 1.116 | 1.112 | 1.069 | 0.966 | 0.675 | 0.639 |

**Llama 3 70B (V=128,256, d=8,192):**

| GPU / Batch Size | 1     | 2     | 4     | 8     | 16    | 32    | 64    | 128   | 256   |
| ---------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| L4               | 0.907 | 0.948 | 0.949 | 0.952 | 0.953 | 1.034 | 1.037 | 1.080 | 0.970 |
| A100-80GB        | 1.143 | 1.068 | 1.070 | 1.073 | 1.084 | 1.139 | 1.086 | 0.740 | 0.697 |
| H100             | 1.133 | 1.132 | 1.122 | 1.126 | 1.112 | 1.098 | 1.103 | 0.937 | 0.663 |
| H200             | 1.112 | 1.108 | 1.097 | 1.097 | 1.146 | 1.116 | 1.094 | 0.733 | 0.618 |
| B200             | 1.087 | 1.075 | 1.068 | 1.057 | 1.061 | 1.053 | 1.023 | 0.705 | 0.600 |

FMMS is **between 5% and 19% faster** than `sampling_from_logits` on datacenter GPUs at typical decode batch sizes (1--32), despite `sampling_from_logits` being a highly optimized unfused Gumbel-max kernel. At larger batch sizes (128+), the unfused kernel wins because cuBLAS dominates and the fusion overhead grows.

### H100 Absolute Performance

The following tables show absolute execution times (in milliseconds) on H100.

**Llama 3 8B (V=128,256, d=4,096):**

| Algorithm / Batch Size                      | 1     | 2     | 4     | 8     | 16    | 32    | 64    | 128   | 256   |
| ------------------------------------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| FMMS (Triton)                               | 0.364 | 0.367 | 0.375 | 0.378 | 0.385 | 0.402 | 0.439 | 0.577 | 0.939 |
| Naive PyTorch Compiled                      | 0.494 | 0.488 | 0.491 | 0.496 | 0.505 | 0.529 | 0.608 | 0.750 | 1.010 |
| flashinfer:top_k_top_p_sampling_from_logits | 0.485 | 0.496 | 0.505 | 0.511 | 0.513 | 0.525 | 0.593 | 0.702 | 0.979 |
| flashinfer:sampling_from_logits             | 0.433 | 0.433 | 0.434 | 0.437 | 0.440 | 0.448 | 0.476 | 0.525 | 0.685 |

**Llama 3 70B (V=128,256, d=8,192):**

| Algorithm / Batch Size                      | 1     | 2     | 4     | 8     | 16    | 32    | 64    | 128   | 256   |
| ------------------------------------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| FMMS (Triton)                               | 0.706 | 0.709 | 0.717 | 0.719 | 0.726 | 0.744 | 0.781 | 0.952 | 1.645 |
| Naive PyTorch Compiled                      | 0.853 | 0.849 | 0.856 | 0.858 | 0.874 | 0.900 | 0.991 | 1.118 | 1.406 |
| flashinfer:top_k_top_p_sampling_from_logits | 0.842 | 0.872 | 0.867 | 0.876 | 0.880 | 0.891 | 0.975 | 1.068 | 1.378 |
| flashinfer:sampling_from_logits             | 0.799 | 0.802 | 0.804 | 0.809 | 0.808 | 0.817 | 0.861 | 0.892 | 1.091 |

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
