from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from scipy.stats import chisquare

from fused_mm_sampling.core import JLSampler, bsz_h, fused_mm_sample_triton

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


def test_fused_triton_sampling_distribution():
    """Verify that the fused Triton kernel samples from the correct distribution.

    Uses synthetic inputs with two known logit vectors (ascending and descending),
    draws many samples, and checks that each empirical distribution fits the
    theoretical softmax probabilities via a chi-squared test.
    """
    inputs = make_synthetic_inputs()
    num_samples = 10_000
    temperature = 5.0

    samples = fused_mm_sample_triton(
        inputs.weights, inputs.hidden_states, num_samples, temperature, seed=42
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
            f"Fused kernel does not match the expected softmax distribution."
        )


@dataclass
class SyntheticInputs:
    weights: torch.Tensor  # [V, D], bfloat16
    hidden_states: torch.Tensor  # [n_hidden_states, D], bfloat16
    logits: (
        torch.Tensor
    )  # [n_hidden_states, V], float32 (the exact logits before bf16 quantization)
    vocab_size: int
    hidden_size: int


def make_synthetic_inputs(vocab_size: int = 256, hidden_size: int = 10) -> SyntheticInputs:
    """Build weights and hidden_states that produce known logits.

    Creates two hidden states: one with ascending logits (favors high token
    indices) and one with descending logits (favors low token indices).
    """
    logits1 = torch.arange(-vocab_size / 2, vocab_size / 2, dtype=torch.float32)[None, :]
    logits2 = torch.arange(vocab_size / 2, -vocab_size / 2, step=-1, dtype=torch.float32)[None, :]
    logits = torch.cat([logits1, logits2], dim=0).to(device)  # [2, V]
    n_hidden_states = logits.shape[0]

    U, _, _ = torch.linalg.svd(logits, full_matrices=False)  # noqa: N806

    torch.manual_seed(0)
    hidden_states = torch.cat(
        [U, torch.rand((n_hidden_states, hidden_size - n_hidden_states), device=device)],
        dim=1,
    ).to(device)
    weights = torch.linalg.pinv(hidden_states) @ logits  # [D, V]

    return SyntheticInputs(
        weights=weights.bfloat16().T.contiguous(),  # [V, D]
        hidden_states=hidden_states.bfloat16(),
        logits=logits,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
    )
