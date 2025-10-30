"""
Work in progress: The implementation does't work yet.
Helion implementation for benchmarking purposes.
This module is separate from core to avoid making helion a required dependency.
It is only used for performance comparisons.
"""

import torch
import helion
import helion.language as hl


@helion.kernel(autotune_effort="none")
def fused_sample_helion(weights: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    assert weights.size(1) == hidden_states.size(0)
    V, D = weights.size()
    seq_len = hidden_states.size(1)
    hl_seq_len = hl.specialize(seq_len)

    gumbel_max = float("-inf") * torch.ones(seq_len, device=weights.device)
    gumbel_max_idx = torch.empty(seq_len, dtype=torch.long, device=weights.device)

    for tile_v in hl.tile(V):
        logits_blk = hl.zeros([tile_v, hl_seq_len], dtype=torch.float32)
        for tile_d in hl.tile(D):
            mm = torch.matmul(weights[tile_v, tile_d], hidden_states[tile_d, :])
            logits_blk = logits_blk + mm
        # We cannot use torch.rand here to generate multiple samples at once,
        # because it is not yet supported by Helion.
        # https://github.com/pytorch/helion/issues/1041
        unif_noise = torch.rand_like(logits_blk, dtype=torch.float32)
        gumbel_noise = -(-unif_noise.log()).log()
        # [num_samples, seq_len]
        summed = logits_blk + gumbel_noise
        new_max = hl.reduce(torch.max, summed, dim=0)
        new_max_idx_local = torch.argmax(summed, dim=0)
        new_max_idx_global = tile_v.begin + new_max_idx_local

        replace_mask = new_max > gumbel_max[:]
        gumbel_max[:] = torch.where(replace_mask, new_max, gumbel_max[:])
        gumbel_max_idx[:] = torch.where(replace_mask, new_max_idx_global, gumbel_max_idx[:])

    return gumbel_max, gumbel_max_idx
