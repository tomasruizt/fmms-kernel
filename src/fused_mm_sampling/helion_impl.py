"""
Work in progress: The implementation does't work yet.
Helion implementation for benchmarking purposes.
This module is separate from core to avoid making helion a required dependency.
It is only used for performance comparisons.
"""

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


def fused_sample_helion(hidden_states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    maxs, max_idxs = fused_sample_helion_first_stage(hidden_states, weights)
    # second stage: reduction
    return max_idxs[maxs.argmax(dim=0)]


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
    hidden_states: torch.Tensor, weights: torch.Tensor
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
    m, k = hidden_states.size()
    k2, n = weights.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    maxs = torch.empty(
        [m, 1],
        dtype=torch.promote_types(hidden_states.dtype, weights.dtype),
        device=hidden_states.device,
    )
    max_idxs = torch.empty([m, 1], dtype=torch.int32, device=hidden_states.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, hidden_states[tile_m, tile_k], weights[tile_k, tile_n])
        acc += -(-torch.rand_like(acc, dtype=torch.float32).log()).log()
        maxs[tile_m, 0] = hl.reduce(torch.max, acc, dim=1)
        # this line below still fails
        max_idxs[tile_m, 0] = acc.argmax(dim=1) + tile_n.begin

    return maxs, max_idxs


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
    # TODO: This is still in the old format. Not yet migrated to
    # logits with shape [n_hidden_states, V].
    vocab_size = 100  # V
    hidden_size = 10  # D
    logits1 = torch.arange(-vocab_size / 2, vocab_size / 2)[None, :]  # [1, V]
    logits2 = torch.arange(vocab_size / 2, -vocab_size / 2, step=-1)[None, :]  # [1, V]
    logits = torch.cat([logits1, logits2], dim=0)  # [n_hidden_states, V]
    n_hidden_states = logits.shape[0]
    # use SVD to construct the hidden states that yield the logits
    # use pseudoinverse to construct the weights.
    # (there are many ways to do this, this is just one)
    # W @ H = L.T
    #  -> W = L.T @ H⁻¹
    U, S, Vt = torch.linalg.svd(logits, full_matrices=False)
    hidden_states = torch.cat(  # [D, n_hidden_states]
        [
            U.T,
            torch.rand((hidden_size - n_hidden_states, n_hidden_states)),  # padding
        ],
    )
    weights = logits.T @ torch.linalg.pinv(hidden_states)  # [V, D]
    assert torch.allclose(weights @ hidden_states, logits.T)

    # To bfloat 16
    weights = weights.bfloat16().to(device)
    hidden_states = hidden_states.bfloat16().to(device)

    result = fused_sample_helion(weights, hidden_states)
