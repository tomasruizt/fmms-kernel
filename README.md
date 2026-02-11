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

vLLM's default (baseline) sampling path uses plain PyTorch ops (`matmul` + `softmax` + `multinomial`). The table below shows FMMS performance relative to this baseline compiled with `torch.compile`. Values > 1.0 mean FMMS is faster.

| GPU / Batch Size | 1     | 2     | 4     | 8     | 16    | 32    | 64    | 128   | 256   |
| ---------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| L4               | 0.916 | 0.954 | 0.955 | 0.955 | 0.964 | 1.049 | 1.088 | 1.254 | 1.216 |
| A100-80GB        | 1.184 | 1.109 | 1.091 | 1.103 | 1.117 | 1.175 | 1.215 | 0.993 | 0.858 |
| H100             | 1.221 | 1.199 | 1.199 | 1.195 | 1.212 | 1.215 | 1.274 | 1.174 | 0.884 |
| H200             | 1.225 | 1.204 | 1.193 | 1.195 | 1.252 | 1.238 | 1.260 | 0.944 | 0.747 |
| B200             | 1.296 | 1.271 | 1.251 | 1.262 | 1.277 | 1.276 | 1.262 | 0.900 | 0.735 |

FMMS is **~20% faster** across typical decode batch sizes (1--128) on datacenter GPUs. At large batch sizes (256+), the matmul becomes compute-bound and the unfused baseline with cuBLAS is faster. On L4, FMMS is slower at small batches but faster at large ones.

### FMMS vs FlashInfer

FlashInfer provides two sampling kernels.
`top_k_top_p_sampling_from_logits` is the one used for top-k/top-p sampling in vLLM.
`sampling_from_logits` is a simpler Gumbel-max kernel without filtering that is very performant, but not used in vLLM.
I include it here as a competitive alternative.
Both require pre-materialized logits, so their runtime includes the matmul.
The table below shows performance relative to `top_k_top_p_sampling_from_logits` (baseline = 1.0).

**FMMS (Triton)** relative to `top_k_top_p_sampling_from_logits` (baseline = 1.0):

| GPU / Batch Size | 1     | 2     | 4     | 8     | 16    | 32    | 64    | 128   | 256   |
| ---------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| L4               | 1.088 | 1.132 | 1.130 | 1.132 | 1.141 | 1.228 | 1.390 | 1.655 | 1.662 |
| A100-80GB        | 1.290 | 1.200 | 1.217 | 1.223 | 1.232 | 1.280 | 1.304 | 1.168 | 1.022 |
| H100             | 1.352 | 1.330 | 1.325 | 1.333 | 1.333 | 1.309 | 1.367 | 1.316 | 1.094 |
| H200             | 1.379 | 1.411 | 1.397 | 1.397 | 1.432 | 1.395 | 1.391 | 1.071 | 0.958 |
| B200             | 1.476 | 1.502 | 1.507 | 1.504 | 1.506 | 1.469 | 1.391 | 0.968 | 0.868 |

FMMS is **between 9% and 50% faster** than the `top_k_top_p` kernel on datacenter GPUs at typical decode batch sizes (1--64), because it avoids the extra HBM round-trip for logits. On L4, FMMS is faster across all batch sizes (up to 66% at H=256).

### H100 Absolute Performance

The following table shows absolute execution times (in milliseconds) on H100.

| Algorithm / Batch Size                      | 1     | 2     | 4     | 8     | 16    | 32    | 64    | 128   | 256   |
| ------------------------------------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| FMMS (Triton)                               | 0.706 | 0.710 | 0.716 | 0.720 | 0.726 | 0.744 | 0.781 | 0.956 | 1.594 |
| Naive PyTorch Compiled                      | 0.862 | 0.851 | 0.858 | 0.860 | 0.879 | 0.904 | 0.995 | 1.123 | 1.409 |
| flashinfer:top_k_top_p_sampling_from_logits | 0.954 | 0.944 | 0.949 | 0.959 | 0.968 | 0.975 | 1.068 | 1.258 | 1.744 |
| flashinfer:sampling_from_logits             | 0.802 | 0.806 | 0.808 | 0.811 | 0.811 | 0.821 | 0.864 | 0.894 | 1.065 |

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
