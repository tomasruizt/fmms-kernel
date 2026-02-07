from dataclasses import dataclass

import torch


@dataclass
class SyntheticInputs:
    weights: torch.Tensor  # [V, D], bfloat16
    hidden_states: torch.Tensor  # [n_hidden_states, D], bfloat16
    logits: (
        torch.Tensor
    )  # [n_hidden_states, V], float32 (the exact logits before bf16 quantization)
    vocab_size: int
    hidden_size: int


def make_synthetic_inputs(
    vocab_size: int = 256,
    hidden_size: int = 10,
    n_hidden_states: int = 2,
    device: torch.device = torch.device("cuda"),
) -> SyntheticInputs:
    """Build weights and hidden_states that produce known logits.

    Creates up to two hidden states: one with ascending logits (favors high
    token indices) and one with descending logits (favors low token indices).
    """
    logits1 = torch.arange(-vocab_size / 2, vocab_size / 2, dtype=torch.float32)[None, :]
    logits2 = torch.arange(vocab_size / 2, -vocab_size / 2, step=-1, dtype=torch.float32)[None, :]
    all_logits = [logits1, logits2]
    logits = torch.cat(all_logits[:n_hidden_states], dim=0).to(device)
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
