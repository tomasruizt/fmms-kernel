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
    All logits are shifted negative via :func:`shift_logits_negative`.
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

    weights_bf16 = weights.bfloat16().T.contiguous()  # [V, D]
    hidden_states_bf16 = hidden_states.bfloat16()
    weights_bf16, hidden_states_bf16 = shift_logits_negative(
        weights_bf16,
        hidden_states_bf16,
        offset=float(vocab_size),
    )

    return SyntheticInputs(
        weights=weights_bf16,
        hidden_states=hidden_states_bf16,
        logits=logits,
        vocab_size=vocab_size,
        hidden_size=hidden_size + 1,
    )


def shift_logits_negative(
    weights: torch.Tensor,
    hidden_states: torch.Tensor,
    offset: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift all logits by -offset without touching the existing weights.

    Appends a bias column so that ``h_new @ W_new^T = h @ W^T - offset``.
    Since softmax is shift-invariant the expected sampling distribution is
    unchanged, but the all-negative logits exercise masked-fill handling in
    partial V-tiles (kernels must fill masked rows with -inf, not 0, or the
    0 beats all real negative values in the tile-max reduction).

    We use a bias column instead of baking the offset into the logits before
    computing the pseudoinverse because bf16 cannot represent fine-grained
    all-negative logits for vocab sizes above ~128.  The bias column keeps the
    original weights (centered near 0) intact and encodes the offset exactly.
    """
    vocab_size = weights.shape[0]
    n_hidden_states = hidden_states.shape[0]
    device = weights.device
    dtype = weights.dtype
    bias_w = torch.ones(vocab_size, 1, dtype=dtype, device=device)
    bias_h = torch.full((n_hidden_states, 1), -offset, dtype=dtype, device=device)
    return (
        torch.cat([weights, bias_w], dim=1),
        torch.cat([hidden_states, bias_h], dim=1),
    )
