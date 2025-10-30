import os

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

import torch
from fused_mm_sampling import fused_mm_sample_triton


torch.set_default_device("cuda")

# Input dimensions (typical for large language models)
vocab_size = 256_000
hidden_size = 5120
seq_len = 256

print("Running example with:")
print(f"  vocab_size = {vocab_size:,}")
print(f"  hidden_size = {hidden_size:,}")
print(f"  seq_len = {seq_len:,}")

# Create random inputs
weights = torch.randn(vocab_size, hidden_size, dtype=torch.float32)
hidden_states = torch.randn(hidden_size, seq_len, dtype=torch.float32)

# Sample from categorical distribution using fused Triton kernel
samples = fused_mm_sample_triton(
    weights=weights,
    hidden_states=hidden_states,
    num_samples=16,
    temperature=0.8,
    seed=0,
)

print(f"\nOutput shape: {samples.shape}")
print(f"Sample values (first 10): {samples.flatten()[:10].tolist()}")
print("\n✓ Example completed successfully!")
