from pathlib import Path

import torch

from fused_mm_sampling.core import sample

device = torch.device("cuda")

vocab_size = 256000
hidden_size = 5120  # dimension of the hidden states
n_hidden_states = 2  # num vectors in the hidden states

print("Started memory profiling")
torch.cuda.memory._record_memory_history()

hidden_states = torch.randn((n_hidden_states, hidden_size), device=device)
weights = torch.randn((hidden_size, vocab_size), device=device)
samples = sample(weights, hidden_states, num_samples=1, temperature=1.0)
print(samples.shape)
path = Path(__file__).parent / "profiles" / "memory" / "mem-snapshot.pickle"
path.parent.mkdir(parents=True, exist_ok=True)
torch.cuda.memory._dump_snapshot(path)

# torch.cuda.memory._record_memory_history(enabled=None)
print("Finished memory profiling")
