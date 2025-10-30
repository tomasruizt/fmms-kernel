# import os

# os.environ["TRITON_INTERPRET"] = "1"
import torch
import triton
import triton.language as tl


def sample(
    weights: torch.Tensor,  # [V, D]
    hidden_states: torch.Tensor,  # [D, n_hidden_states]
    num_samples: int,
    temperature: float,
    return_probs: bool = False,
):
    logits = weights @ hidden_states  # [V, n_hidden_states]
    logits -= torch.max(logits, dim=0, keepdim=True).values
    probs = torch.nn.functional.softmax(logits / temperature, dim=0)  # [n_hidden_states, V]
    samples = torch.multinomial(probs.T, num_samples=num_samples, replacement=True)
    if return_probs:
        return samples, probs
    return samples


def incremental_sample_pt(
    weights: torch.Tensor,
    hidden_states: torch.Tensor,
    num_samples: int,
    temperature: float,
):
    V, D = weights.shape  # noqa: N806
    D, n_hidden_states = hidden_states.shape  # noqa: N806
    block_size = 8
    # compute logits blocks
    gumbel_max = float("-inf") * torch.ones(size=(num_samples, n_hidden_states))
    gumbel_max_idx = torch.empty(size=(num_samples, n_hidden_states), dtype=torch.long)
    n_blocks = cdiv(V, block_size)
    for blk_idx in range(n_blocks):
        idx_from = blk_idx * block_size
        idx_to = (blk_idx + 1) * block_size
        w_blk = weights[idx_from:idx_to]  # [block_size, D]
        logits_blk = w_blk @ hidden_states / temperature  # [n_hidden_states, block_size]
        unif_noise = torch.rand((num_samples, *logits_blk.shape))
        gumbel_noise = -(-unif_noise.log()).log()
        new_max, new_max_idx_local = torch.max(logits_blk + gumbel_noise, dim=1)
        new_max_idx_global = idx_from + new_max_idx_local

        replace_mask = new_max > gumbel_max
        gumbel_max = torch.where(replace_mask, new_max, gumbel_max)
        gumbel_max_idx = torch.where(replace_mask, new_max_idx_global, gumbel_max_idx)
    return gumbel_max_idx.T


def cdiv(n: int, div: int) -> int:
    return (n + div - 1) // div


MIN_BLOCK_SIZE_V = 8


def fused_mm_sample_triton(
    weights: torch.Tensor,
    hidden_states: torch.Tensor,
    num_samples: int,
    temperature: float,
    seed: int,
):
    V, D = weights.shape  # noqa: N806
    D, n_hidden_states = hidden_states.shape  # noqa: N806

    max_grid_size = triton.cdiv(V, MIN_BLOCK_SIZE_V)
    maxs = float("-inf") * torch.ones(
        (max_grid_size, n_hidden_states, num_samples),
        dtype=torch.bfloat16,
        device=weights.device,
    )
    maxs_idx = torch.empty_like(maxs, dtype=torch.long)

    def grid(meta):
        return (triton.cdiv(V, meta["BLOCK_SIZE_V"]),)

    n_hidden_states_p2 = triton.next_power_of_2(n_hidden_states)

    fused_mm_sample_triton_kernel[grid](
        weights_ptr=weights,
        hidden_states_ptr=hidden_states,
        max_out_ptr=maxs,
        max_out_idx_ptr=maxs_idx,
        vocab_size=V,
        hidden_size=D,
        n_hidden_states=n_hidden_states,
        num_samples=num_samples,
        temperature=temperature,
        seed=seed,
        n_hidden_states_p2=n_hidden_states_p2,
    )

    # 2nd stage: reduction
    idxs = maxs.max(axis=0).indices
    samples = maxs_idx.gather(dim=0, index=idxs[None, :])
    return samples.squeeze(0)  # [n_hidden_states, num_samples]


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_V": bv,
                "BLOCK_SIZE_D": bd,
                "N_SAMPLES_BSZ": bsz,
            }
        )
        for bv in [MIN_BLOCK_SIZE_V, 32]
        for bd in [16, 32]
        for bsz in [1, 4]
    ],
    key=["vocab_size", "hidden_size", "n_hidden_states", "num_samples"],
    cache_results=True,
)
@triton.jit
def fused_mm_sample_triton_kernel(
    weights_ptr,
    hidden_states_ptr,
    max_out_ptr,  # [grid_size, n_hidden_states, num_samples]
    max_out_idx_ptr,  # [grid_size, n_hidden_states, num_samples]
    vocab_size,  # V
    hidden_size: tl.constexpr,  # D
    n_hidden_states: tl.constexpr,
    num_samples: tl.constexpr,
    temperature: float,
    seed: int,
    BLOCK_SIZE_V: tl.constexpr,  # noqa: N803
    BLOCK_SIZE_D: tl.constexpr,  # noqa: N803
    N_SAMPLES_BSZ: tl.constexpr,  # noqa: N803
    n_hidden_states_p2: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_V

    # We don't instantiate gumbel_max yet, because each program just writes
    # its local max into main memory for a parallel reduction in stage 2.

    offsets_v = block_start + tl.arange(0, BLOCK_SIZE_V)
    mask_v = offsets_v < vocab_size

    offset_seqlen = tl.arange(0, n_hidden_states)
    logits_blk = tl.zeros((BLOCK_SIZE_V, n_hidden_states), dtype=tl.float32)

    # Compute a block of logits logits_blk
    for d_start in range(0, hidden_size, BLOCK_SIZE_D):
        offsets_h = d_start + tl.arange(0, BLOCK_SIZE_D)
        mask_h = offsets_h < hidden_size

        w_offsets = offsets_v[:, None] * hidden_size + offsets_h[None, :]
        w_blk = tl.load(
            weights_ptr + w_offsets,
            mask=mask_v[:, None] & mask_h[None, :],
        )

        hidden_states_blk = tl.load(
            hidden_states_ptr + offset_seqlen[None, :] + n_hidden_states * offsets_h[:, None],
            mask=mask_h[:, None],
        )
        logits_blk = tl.dot(w_blk, hidden_states_blk, acc=logits_blk)

    logits_blk = logits_blk / temperature  # [Vblk, n_hidden_states]

    # Process samples in batches to limit memory usage
    samples_n_batches: tl.constexpr = triton.cdiv(num_samples, N_SAMPLES_BSZ)
    for batch_idx in range(samples_n_batches):
        # Calculate how many samples in this batch
        batch_start = batch_idx * N_SAMPLES_BSZ
        batch_end = min(batch_start + N_SAMPLES_BSZ, num_samples)
        actual_batch_size = batch_end - batch_start

        # Note: Creating appropriately sized tensors is tricky because
        # tl.arange() only accepts tl.constexpr that are powers of 2.
        noise_size: tl.constexpr = BLOCK_SIZE_V * n_hidden_states_p2 * N_SAMPLES_BSZ
        noise_offsets = tl.arange(0, noise_size).reshape(
            (N_SAMPLES_BSZ, BLOCK_SIZE_V, n_hidden_states_p2)
        )
        # Note: Each program needs a different seed, otherwise they
        # all create the same noise, leading to sampling artifacts.
        # Also vary seed by batch_idx to ensure different noise per batch
        unif_noise = tl.rand(seed + pid + batch_idx * 1000000, noise_offsets)
        gumbel_noise = -tl.log(-tl.log(unif_noise))

        gumbel_max, gumbel_max_idx_local = tl.max(
            logits_blk + gumbel_noise, axis=1, return_indices=True
        )  # [batch_size_p2, n_hidden_states_p2]
        gumbel_max_idx_global = gumbel_max_idx_local + block_start

        # Output offset for this batch
        out_blk_start = pid * n_hidden_states * num_samples + batch_start

        # Note: It makes a difference if indices are row-major or column-major
        # Note: The stride needs to match the non-padded shape!
        out_offsets = (
            tl.arange(0, N_SAMPLES_BSZ)[:, None]
            + num_samples * tl.arange(0, n_hidden_states_p2)[None, :]
        )
        out_mask = tl.arange(0, N_SAMPLES_BSZ)[:, None] < actual_batch_size
        tl.store(
            max_out_ptr + out_blk_start + out_offsets,
            gumbel_max,
            mask=out_mask,
        )
        tl.store(
            max_out_idx_ptr + out_blk_start + out_offsets,
            gumbel_max_idx_global,
            mask=out_mask,
        )
