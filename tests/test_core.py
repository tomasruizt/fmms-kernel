import os
from pathlib import Path

os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import pytest
import torch
from scipy.stats import chisquare

from fused_mm_sampling.core import JLSampler, bsz_h, get_sampler
from fused_mm_sampling.testing import make_synthetic_inputs

device = torch.device("cuda")


def test_jl_sampling_aproximate_correctness():
    folder = Path(__file__).parent / "qwen3-0.6b"
    weights = torch.load(folder / "weights.pt", map_location=device)
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


@pytest.mark.parametrize("n_hidden_states", [1, 2])
@pytest.mark.parametrize("vocab_size", [100, 256])
@pytest.mark.parametrize(
    "provider",
    [
        "fused-triton",
        "naive-pt",
        "naive-compiled",
        "sequential-compiled",
        "helion",
        "flashinfer:sampling_from_logits",
        "flashinfer:top_k_top_p_sampling_from_logits",
    ],
)
def test_sampling_distribution(provider, vocab_size, n_hidden_states):
    """Verify that a sampler produces the correct distribution.

    Uses synthetic inputs with known logit vectors (ascending and/or descending),
    draws many samples, and checks that each empirical distribution fits the
    theoretical softmax probabilities via a chi-squared test.
    """
    inputs = make_synthetic_inputs(vocab_size=vocab_size, n_hidden_states=n_hidden_states)
    num_samples = 10_000
    temperature = 5.0

    sampler = get_sampler(provider, weights=inputs.weights)
    sampler.prepare()
    samples = sampler.sample(
        weights=inputs.weights,
        hidden_states=inputs.hidden_states,
        num_samples=num_samples,
        temperature=temperature,
    )

    for seq_idx in range(inputs.logits.shape[0]):
        # Compare empirical counts against theoretical expected counts from softmax.
        expected_probs = (inputs.logits[seq_idx] / temperature).softmax(dim=0)
        expected_counts = (expected_probs * num_samples).cpu().numpy()
        empirical_counts = (
            torch.bincount(samples[seq_idx], minlength=inputs.vocab_size).float().cpu().numpy()
        )

        # Only test bins with enough expected counts for chi-squared to be valid.
        mask = expected_counts >= 5
        obs = empirical_counts[mask]
        exp = expected_counts[mask]
        # Rescale expected counts so sums match (samples in excluded bins shift the totals).
        exp = exp * (obs.sum() / exp.sum())
        _, p_value = chisquare(obs, exp)
        assert p_value > 0.001, (
            f"Sampling distribution mismatch for seq {seq_idx}: p={p_value:.6f}. "
            f"{provider} does not match the expected softmax distribution."
        )
