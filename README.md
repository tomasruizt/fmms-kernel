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