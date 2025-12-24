"""
Work in progress: The implementation does't work yet.
Helion implementation for benchmarking purposes.
This module is separate from core to avoid making helion a required dependency.
It is only used for performance comparisons.
"""

import os

# Since the helion kernel samples (is stochastic), we should not verify exactness.
os.environ["HELION_AUTOTUNE_ACCURACY_CHECK"] = "0"

import helion
import helion.language as hl
import torch


@helion.kernel(autotune_effort="none")
def fused_sample_helion_ckpt(weights: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    assert weights.size(1) == hidden_states.size(0)
    V, D = weights.size()  # noqa: N806
    n_hidden_states = hidden_states.size(1)
    hl_n_hidden_states = hl.specialize(n_hidden_states)

    gumbel_max = float("-inf") * torch.ones(n_hidden_states, device=weights.device)
    gumbel_max_idx = torch.empty(n_hidden_states, dtype=torch.long, device=weights.device)

    for tile_v in hl.tile(V):
        logits_blk = hl.zeros([tile_v, hl_n_hidden_states], dtype=torch.float32)
        for tile_d in hl.tile(D):
            mm = torch.matmul(weights[tile_v, tile_d], hidden_states[tile_d, :])
            logits_blk = logits_blk + mm
        # We cannot use torch.rand here to generate multiple samples at once,
        # because it is not yet supported by Helion.
        # https://github.com/pytorch/helion/issues/1041
        unif_noise = torch.rand_like(logits_blk, dtype=torch.float32)
        gumbel_noise = -(-unif_noise.log()).log()
        # [num_samples, n_hidden_states]
        summed = logits_blk + gumbel_noise
        new_max = hl.reduce(torch.max, summed, dim=0)
        new_max_idx_local = torch.argmax(summed, dim=0)
        new_max_idx_global = tile_v.begin + new_max_idx_local

        replace_mask = new_max > gumbel_max[:]
        gumbel_max[:] = torch.where(replace_mask, new_max, gumbel_max[:])
        gumbel_max_idx[:] = torch.where(replace_mask, new_max_idx_global, gumbel_max_idx[:])

    return gumbel_max, gumbel_max_idx


def fused_sample_helion(
    hidden_states: torch.Tensor,  # [H, D]
    weights: torch.Tensor,  # [V, D]
    temperature: float,
    num_samples: int,
) -> torch.Tensor:
    maxs, max_idxs, noise = fused_sample_helion_first_stage(
        hidden_states, weights, temperature, num_samples
    )
    # So far, the noise is always the same. It seems the torch.rand() call is not evaluated multiple times.
    # See: https://github.com/pytorch/helion/issues/1309
    # second stage: reduction
    idxs = maxs[:, : weights.shape[0], :].argmax(dim=1)
    samples = max_idxs.gather(dim=1, index=idxs[None, :])
    return samples.squeeze(0)  # [H, num_samples]


# @helion.kernel(
#     # static_shapes=True gives a performance boost for matmuls
#     static_shapes=True,
#     # Disable autotung over unrolling/range_num_stages
#     # tl.dot is pipelined with num_stages
#     autotune_config_overrides={
#         "range_unroll_factors": [0, 0],
#         "range_num_stages": [0, 0],
#     },
# )
@helion.kernel(autotune_effort="none")
def fused_sample_helion_first_stage(
    hidden_states: torch.Tensor,  # [H, D]
    weights: torch.Tensor,  # [V, D]
    temperature: float,
    num_samples: int,
) -> torch.Tensor:
    """
    Performs matrix multiplication of x and y with an optional epilogue function.
    Args:
        x (Tensor): Left matrix of shape [m, k].
        y (Tensor): Right matrix of shape [k, n].
        epilogue (Callable, optional): Function applied to the accumulator and tile indices
            after the matmul. Defaults to identity (no change).
    Returns:
        Tensor: Resulting matrix of shape [m, n].
    """
    m, k = hidden_states.shape
    n, k2 = weights.shape
    bsz_n = 128
    assert k == k2, f"size mismatch {k} != {k2}"
    maxs = torch.full(
        [num_samples, helion.cdiv(n, bsz_n), m],
        fill_value=float("-inf"),
        dtype=torch.promote_types(hidden_states.dtype, weights.dtype),
        device=hidden_states.device,
    )
    max_idxs = torch.full(
        maxs.shape,
        fill_value=-1,
        dtype=torch.long,
        device=maxs.device,
    )
    noise = torch.empty([num_samples, m, n], dtype=weights.dtype, device=weights.device)
    g_cuda = torch.Generator(device="cuda")

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, hidden_states[tile_m, tile_k], weights[tile_n, tile_k].T)
        acc /= temperature

        for sample_idx in range(num_samples):
            unoise = torch.rand(acc.shape, dtype=acc.dtype, device=acc.device, generator=g_cuda)
            gumbel_noise = -(-unoise.log()).log()
            noise[sample_idx, tile_m, tile_n] = unoise
            val = acc + gumbel_noise
            maxs[sample_idx, tile_n.id, tile_m] = hl.reduce(torch.max, val, dim=1)
            max_idxs[sample_idx, tile_n.id, tile_m] = val.argmax(dim=1) + tile_n.begin

    return maxs, max_idxs, noise


@helion.kernel(
    # static_shapes=True gives a performance boost for matmuls
    static_shapes=True,
    # Disable autotung over unrolling/range_num_stages
    # tl.dot is pipelined with num_stages
    autotune_config_overrides={
        "range_unroll_factors": [0, 0],
        "range_num_stages": [0, 0],
    },
)
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication of x and y with an optional epilogue function.
    Args:
        x (Tensor): Left matrix of shape [m, k].
        y (Tensor): Right matrix of shape [k, n].
    Returns:
        Tensor: Resulting matrix of shape [m, n].
    """
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


if __name__ == "__main__":
    device = torch.device("cuda")
    vocab_size = 128  # V
    hidden_size = 16  # D
    logits1 = torch.arange(-vocab_size / 2, vocab_size / 2)[None, :]  # [1, V]
    logits2 = torch.arange(vocab_size / 2, -vocab_size / 2, step=-1)[None, :]  # [1, V]
    logits = torch.cat([logits1, logits2], dim=0)  # [H, V]
    H = logits.shape[0]
    # use SVD to construct the hidden states that yield the logits
    # use pseudoinverse to construct the weights.
    # (there are many ways to do this, this is just one)
    # H @ W = L
    #  -> W = H⁻¹ @ L
    U, S, Vt = torch.linalg.svd(logits, full_matrices=False)
    hidden_states = torch.cat(  # [D, H]
        [
            U.T,
            torch.rand((hidden_size - H, H)),  # padding
        ],
    )
    hidden_states = hidden_states.T.contiguous()  # [H, D]
    weights = torch.linalg.pinv(hidden_states) @ logits  # [D, V]
    weights = weights.T.contiguous()  # [V, D]
    assert torch.allclose(hidden_states @ weights.T, logits)

    # To bfloat 16
    weights = weights.bfloat16().to(device)
    hidden_states = hidden_states.bfloat16().to(device)  # [H, D]

    result = fused_sample_helion(weights, hidden_states, temperature=1.0)
