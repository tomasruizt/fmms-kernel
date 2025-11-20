from pathlib import Path

import torch

from fused_mm_sampling.core import JLSampler

device = torch.device("cuda")


def test_jl_sampling_aproximate_correctness():
    folder = Path(__file__).parent / "qwen3-0.6b"
    weights = torch.load(folder / "weights.pt", map_location=device)
    hidden_states = torch.load(folder / "hidden_states.pt", map_location=device)
    expected_logits = hidden_states @ weights
    expected_probs = expected_logits.softmax(dim=1)

    jl_sampler = JLSampler.from_weights(weights, epsilon=0.2).prepare()
    actual_logits = jl_sampler.compute_logits(hidden_states)
    actual_probs = actual_logits.softmax(dim=1)
    assert torch.allclose(actual_probs, expected_probs, atol=0.2)

    print(f"{weights.shape=}")
    print(f"{hidden_states.shape=}")
