from pathlib import Path

import pytest
import torch

from fused_mm_sampling.core import JLSampler, bsz_h

device = torch.device("cuda")


def test_jl_sampling_aproximate_correctness():
    folder = Path(__file__).parent / "qwen3-0.6b"
    weights = torch.load(folder / "weights.pt", map_location=device).T.contiguous()
    hidden_states = torch.load(folder / "hidden_states.pt", map_location=device)
    expected_logits = hidden_states @ weights.T
    expected_probs = expected_logits.softmax(dim=1)

    jl_sampler = JLSampler.from_weights(weights, epsilon=0.2).prepare()
    actual_logits = jl_sampler.compute_logits(hidden_states)
    actual_probs = actual_logits.softmax(dim=1)
    assert torch.allclose(actual_probs, expected_probs, atol=0.2)

    print(f"{weights.shape=}")
    print(f"{hidden_states.shape=}")


@pytest.mark.parametrize(
    "args",
    [
        # H, expected BSZ_H
        (1, 16),
        (16, 16),
        (17, 32),
        (32, 32),
        (33, 64),
        (64, 64),
        (65, 64),
    ],
)
def test_bsz_h(args):
    h, expected_bsz_h = args
    assert bsz_h(h) == expected_bsz_h
