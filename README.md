# Fused Matrix Multiplication & Sampling

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

### Relative Performance

The following table shows the relative performance of the fused-matmul-sample kernel across different GPUs and batch sizes. Values are relative to `flashinfer:sampling_from_logits` (baseline = 1.0). Values > 1.0 indicate the fused kernel is faster, while values < 1.0 indicate it is slower.

| GPU | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|-----|---|---|---|---|----|----|----|-----|-----|-----|------|
| L4 | 1.021 | 1.036 | 1.057 | 1.029 | 1.034 | 1.049 | 1.069 | 1.111 | 0.846 | 0.854 | 0.797 |
| A100-80GB | 1.027 | 1.034 | 1.034 | 1.041 | 1.055 | 0.972 | 0.975 | 0.716 | 0.580 | 0.593 | 0.633 |
| H100 | 1.060 | 1.061 | 1.058 | 1.062 | 1.066 | 1.018 | 1.012 | 0.980 | 0.701 | 0.634 | 0.626 |
| B200 | 1.080 | 1.068 | 1.065 | 1.066 | 1.060 | 0.934 | 0.917 | 0.753 | 0.757 | 0.604 | 0.596 |

*Table values represent relative performance (higher is better). Batch sizes (n_hidden_states) are shown in the column headers. Data as of 2025-12-21. Benchmarks were run on Modal GPUS.*

### H100 Absolute Performance

The following table shows the absolute execution times (in milliseconds) of the fused matmul-sampling kernel versus several strong baselines on H100 across various batch sizes.

| Algorithm | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|-----------|---|---|---|---|----|----|----|-----|-----|-----|------|
| Fused Matmul-Sampling | 1.383 | 1.384 | 1.392 | 1.393 | 1.400 | 1.478 | 1.520 | 1.734 | 3.126 | 6.123 | 12.177 |
| Naive PyTorch Compiled | 1.469 | 1.475 | 1.480 | 1.493 | 1.520 | 1.549 | 1.620 | 1.900 | 2.497 | 4.342 | 8.284 |
| flashinfer:top_k_top_p_sampling_from_logits | 1.708 | 1.739 | 1.741 | 1.753 | 1.774 | 1.847 | 2.048 | 2.407 | 3.486 | 6.303 | 12.347 |
| flashinfer:sampling_from_logits | 1.466 | 1.469 | 1.472 | 1.480 | 1.492 | 1.505 | 1.538 | 1.700 | 2.191 | 3.881 | 7.629 |

*Table values represent execution time in milliseconds (lower is better). Batch sizes (n_hidden_states) are shown in the column headers. Data from H100 benchmarks run on Modal, as of 2025-12-21.*

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
