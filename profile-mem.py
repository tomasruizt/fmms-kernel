from pathlib import Path
from fused_mm_sampling import sample
import torch

torch.set_default_device("cuda")

vocab_size = 256000
hidden_size = 5120
seq_len = 2

print("Started memory profiling")
torch.cuda.memory._record_memory_history()

hidden_states = torch.randn((hidden_size, seq_len))
weights = torch.randn((vocab_size, hidden_size))
samples = sample(weights, hidden_states, num_samples=1, temperature=1.0)
print(samples.shape)
path = Path(__file__).parent / "mem-profiles" / "mem-snapshot.pickle"
path.parent.mkdir(parents=True, exist_ok=True)
torch.cuda.memory._dump_snapshot(path)

# torch.cuda.memory._record_memory_history(enabled=None)
print("Finished memory profiling")
