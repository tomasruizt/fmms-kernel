from pathlib import Path

import torch

from fused_mm_sampling.core import JLSampler

torch.set_default_device("cuda")


def test_jl_sampling_aproximate_correctness():
    folder = Path(__file__).parent / "qwen3-0.6b"
    weights = torch.load(folder / "weights.pt")
    hidden_states = torch.load(folder / "hidden_states.pt")
    expected_logits = weights @ hidden_states
    expected_probs = expected_logits.softmax(dim=0)

    jl_sampler = JLSampler.from_weights(weights, epsilon=0.2).prepare()
    actual_logits = jl_sampler.compute_logits(hidden_states)
    actual_probs = actual_logits.softmax(dim=0)
    assert torch.allclose(actual_probs, expected_probs, atol=0.2)

    print(f"{weights.shape=}")
    print(f"{hidden_states.shape=}")
