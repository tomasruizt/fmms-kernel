# import os

# os.environ["TRITON_INTERPRET"] = "1"
import math
from dataclasses import dataclass
from typing import Callable, Protocol

import torch
import triton
import triton.language as tl


def sample(
    weights: torch.Tensor,  # [V, D]
    hidden_states: torch.Tensor,  # [D, n_hidden_states]
    num_samples: int,
    temperature: float,
    return_probs: bool = False,
    seed: int = None,
):
    if seed is not None:
        torch.manual_seed(seed)
    logits = weights @ hidden_states  # [V, n_hidden_states]
    probs = (logits / temperature).softmax(dim=0)  # [V, n_hidden_states]
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


MIN_BLOCK_SIZE_V = 32


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
        return (
            triton.cdiv(V, meta["BLOCK_SIZE_V"]),
            triton.cdiv(n_hidden_states, meta["BLOCK_SIZE_H"]),
        )

    fused_mm_sample_triton_kernel[grid](
        weights_ptr=weights,
        hidden_states_t_ptr=hidden_states.T.contiguous(),
        max_out_ptr=maxs,
        max_out_idx_ptr=maxs_idx,
        vocab_size=V,
        hidden_size=D,
        n_hidden_states=n_hidden_states,
        num_samples=num_samples,
        temperature=temperature,
        seed=seed,
    )

    # 2nd stage: reduction
    idxs = maxs.max(axis=0).indices
    samples = maxs_idx.gather(dim=0, index=idxs[None, :])
    return samples.squeeze(0)  # [n_hidden_states, num_samples]


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_V": MIN_BLOCK_SIZE_V,
                "BLOCK_SIZE_D": 32,
                "BLOCK_SIZE_H": 32,
                "BLOCK_SIZE_NSAMPLES": 4,
                "GROUP_SIZE_V": 4,
            },
            maxnreg=255,
        )
    ],
    key=["vocab_size", "hidden_size", "n_hidden_states", "num_samples"],
    cache_results=True,
)
@triton.jit
def fused_mm_sample_triton_kernel(
    weights_ptr,  # [V, D]
    hidden_states_t_ptr,  # [n_hidden_states, D] (transposed)
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
    BLOCK_SIZE_H: tl.constexpr,  # noqa: N803
    BLOCK_SIZE_NSAMPLES: tl.constexpr,  # noqa: N803
    GROUP_SIZE_V: tl.constexpr,  # noqa: N803
):
    # Compute a different program ordering to exploit L2 cache, as suggested in
    # https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
    pid_v = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_v, pid_h = tl.swizzle2d(pid_v, pid_h, vocab_size, n_hidden_states, GROUP_SIZE_V)
    v_start = pid_v * BLOCK_SIZE_V
    h_start = pid_h * BLOCK_SIZE_H

    # We don't instantiate gumbel_max yet, because each program just writes
    # its local max into main memory for a parallel reduction in stage 2.

    offsets_v = v_start + tl.arange(0, BLOCK_SIZE_V)
    mask_v = offsets_v < vocab_size

    # H-tile within n_hidden_states
    offsets_h = h_start + tl.arange(0, BLOCK_SIZE_H)
    mask_h = offsets_h < n_hidden_states
    logits_blk = tl.zeros((BLOCK_SIZE_V, BLOCK_SIZE_H), dtype=tl.float32)

    # Compute a block of logits logits_blk
    for d_start in range(0, hidden_size, BLOCK_SIZE_D):
        offsets_d = d_start + tl.arange(0, BLOCK_SIZE_D)
        mask_d = offsets_d < hidden_size

        w_offsets = offsets_v[:, None] * hidden_size + offsets_d[None, :]
        w_blk = tl.load(
            weights_ptr + w_offsets,
            mask=mask_v[:, None] & mask_d[None, :],
        )

        # load hidden_states tile [BLOCK_SIZE_H, BLOCK_SIZE_D]
        hidden_states_blk = tl.load(
            hidden_states_t_ptr + offsets_d[None, :] + hidden_size * offsets_h[:, None],
            mask=mask_h[:, None] & mask_d[None, :],
        )
        logits_blk = tl.dot(w_blk, hidden_states_blk.T, acc=logits_blk)

    # Later we will take max over logits + noise, but rows outside the mask
    # should not be considered. Setting them to -inf achieves this.
    logits_blk = tl.where(mask_v[:, None], logits_blk, -float("inf"))
    logits_blk = logits_blk / temperature  # [Vblk, n_hidden_states]

    # Process samples in batches to limit memory usage
    samples_n_batches: tl.constexpr = triton.cdiv(num_samples, BLOCK_SIZE_NSAMPLES)
    for batch_idx in range(samples_n_batches):
        # Calculate how many samples in this batch
        batch_start = batch_idx * BLOCK_SIZE_NSAMPLES
        batch_end = min(batch_start + BLOCK_SIZE_NSAMPLES, num_samples)
        actual_batch_size = batch_end - batch_start

        # Note: Creating appropriately sized tensors is tricky because
        # tl.arange() only accepts tl.constexpr that are powers of 2.
        noise_size: tl.constexpr = BLOCK_SIZE_V * BLOCK_SIZE_H * BLOCK_SIZE_NSAMPLES
        noise_offsets = tl.arange(0, noise_size).reshape(
            (BLOCK_SIZE_NSAMPLES, BLOCK_SIZE_V, BLOCK_SIZE_H)
        )
        # Note: Each tile (v, h) and batch of samples needs a different seed,
        # otherwise they all create the same noise, leading to sampling artifacts.
        unif_noise = tl.rand(
            seed + pid_v * 100 + pid_h * 1_000 + batch_idx * 10_000,
            noise_offsets,
        )
        gumbel_noise = -tl.log(-tl.log(unif_noise))

        gumbel_max, gumbel_max_idx_local = tl.max(
            logits_blk + gumbel_noise, axis=1, return_indices=True
        )
        gumbel_max_idx_global = gumbel_max_idx_local + v_start

        # Output offset for this batch
        out_blk_start = pid_v * n_hidden_states * num_samples + batch_start

        # Note: It makes a difference if indices are row-major or column-major
        # Note: The stride needs to match the non-padded shape!
        out_offsets = (
            tl.arange(0, BLOCK_SIZE_NSAMPLES)[:, None]
            + num_samples * (h_start + tl.arange(0, BLOCK_SIZE_H))[None, :]
        )
        mask_nsamples = tl.arange(0, BLOCK_SIZE_NSAMPLES) < actual_batch_size
        out_mask = mask_nsamples[:, None] & mask_h[None, :]
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


class Sampler(Protocol):
    def prepare(self) -> "Sampler":
        raise NotImplementedError()

    def sample(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError()


@dataclass
class SimpleSampler(Sampler):
    fn: Callable[..., torch.Tensor]

    def prepare(self) -> "SimpleSampler":
        return self

    def sample(self, **kwargs) -> torch.Tensor:
        return self.fn(**kwargs)


@dataclass
class JLSampler(Sampler):
    weights: torch.Tensor  # [V, D]
    k: int
    prepared: bool = False

    def prepare(self) -> "JLSampler":
        D = self.weights.shape[1]  # noqa: N806
        self.rand_mat = torch.randn(
            (D, self.k),
            dtype=self.weights.dtype,
            device=self.weights.device,
        ) / math.sqrt(self.k)
        self.w_p = self.weights @ self.rand_mat  # [V, k]
        self.prepared = True
        self.weights = None  # not needed anymore
        return self

    @torch.compile
    def sample(
        self,
        hidden_states: torch.Tensor,  # [D, n_hidden_states]
        temperature: float,
        num_samples: int,
        seed: int = None,
        weights: torch.Tensor = None,  # ignored
    ):
        """
        Sampling using low-dimensional random projections (Johnson-Lindenstrauss lemma).
        """
        if not self.prepared:
            raise ValueError("Sampler not prepared. Call .prepare() first.")
        if seed is not None:
            torch.manual_seed(seed)
        h_p = self.rand_mat.T @ hidden_states  # [k, n_hidden_states]
        logits_p = self.w_p @ h_p  # [V, n_hidden_states]
        probs = (logits_p / temperature).softmax(dim=0)  # [V, n_hidden_states]
        samples = torch.multinomial(probs.T, num_samples=num_samples, replacement=True)
        return samples
