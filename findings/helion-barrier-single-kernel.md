# Merging stage 2 into the Helion kernel with `hl.barrier()`

## Problem

The FMMS Helion kernel uses a two-stage approach split across GPU and host:

1. **Stage 1 (GPU kernel)**: Each `(V, H)` tile computes logits + Gumbel noise, writes its local max/argmax to `tile_maxs[tile_v.id, tile_h]` and `tile_max_idxs[tile_v.id, tile_h]`.
2. **Stage 2 (Python)**: Reduces across tiles with `tile_maxs.argmax(dim=0)` + `tile_max_idxs.gather(...)`.

Stage 2 requires two separate PyTorch kernel launches (argmax, gather) plus the overhead of returning control to the Python wrapper between stages. For V=128K with `BLOCK_SIZE_V=128`, there are 1000 tiles; the stage-2 argmax costs ~0.1-0.5ms. For V=256K the cost rises to ~1ms.

## Observation

Helion supports `hl.barrier()`, a grid-wide synchronization primitive. The [split-k matmul example](https://helionlang.com/examples/split_k_barrier.html) demonstrates the pattern: two `hl.tile` loops separated by `hl.barrier()` inside a single kernel. Stage 1 writes partial results to a temporary buffer, the barrier ensures all tiles complete, then stage 2 reduces the partials.

The split-k example:

```python
@helion.kernel(static_shapes=True, dot_precision="ieee")
def split_k_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(16, 512, 64))
    block_k = helion.next_power_of_2(helion.cdiv(k, split_k))
    tmp = torch.zeros((m, n, split_k), device=a.device, dtype=a.dtype)
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)

    for tile_m, tile_n, tile_k_outer in hl.tile([m, n, k], block_size=[None, None, block_k]):
        acc = hl.zeros([tile_m, tile_n], device=a.device, dtype=a.dtype)
        for tile_k_inner in hl.tile(tile_k_outer.begin, tile_k_outer.end):
            acc = torch.addmm(acc, a[tile_m, tile_k_inner], b[tile_k_inner, tile_n])
        tmp[tile_m, tile_n, tile_k_outer.id] = acc

    hl.barrier()

    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = torch.sum(tmp[tile_m, tile_n, :], dim=-1)

    return out
```

## Proposed change

Merge both stages into a single Helion kernel using `hl.barrier()`:

```python
@helion.kernel(...)
def fused_sample_helion_kernel(
    weights: torch.Tensor,       # [V, D]
    hidden_states: torch.Tensor, # [D, H]
    out_idxs: torch.Tensor,     # [H] — final sampled token indices
    temperature: float,
    seed: int,
):
    V, D = weights.size()
    H = hidden_states.size(1)
    n_tiles_v = helion.cdiv(V, BLOCK_SIZE_V)

    tile_maxs = torch.full(
        (n_tiles_v, H), float("-inf"), device=weights.device, dtype=torch.float32
    )
    tile_max_idxs = torch.empty(
        (n_tiles_v, H), dtype=torch.long, device=weights.device
    )

    # Stage 1: per-tile matmul + Gumbel-max (parallel over V and H)
    for tile_v, tile_h in hl.tile([V, H], block_size=[BLOCK_SIZE_V, BLOCK_SIZE_H]):
        logits_blk = hl.zeros([tile_v, tile_h], dtype=torch.float32)
        for tile_d in hl.tile(D):
            mm = torch.matmul(weights[tile_v, tile_d], hidden_states[tile_d, tile_h])
            logits_blk = logits_blk + mm
        logits_blk = logits_blk / temperature

        unif_noise = hl.rand([tile_v, tile_h], seed=seed)
        gumbel_noise = -(-unif_noise.log()).log()
        summed = logits_blk + gumbel_noise

        tile_maxs[tile_v.id, tile_h] = hl.reduce(torch.max, summed, dim=0)
        tile_max_idxs[tile_v.id, tile_h] = torch.argmax(summed, dim=0)

    hl.barrier()

    # Stage 2: reduce across V-tiles to find global argmax (parallel over H)
    for tile_h in hl.tile(H):
        best_tile = torch.argmax(tile_maxs[:, tile_h], dim=0)
        out_idxs[tile_h] = tile_max_idxs[best_tile, tile_h]
```

The Python wrapper simplifies to just calling the kernel and collecting `out_idxs` — no more buffer allocation or host-side reduction.

## Expected benefits

| Aspect | Current (2-stage) | With `hl.barrier()` |
|--------|-------------------|---------------------|
| Kernel launches per sample | 1 (kernel) + 2 (argmax, gather) | 1 total |
| Stage 2 data locality | Cold read from DRAM | Hot in L2 cache |
| Stage 2 cost (V=128K) | ~0.1-0.5ms | ~microseconds |
| Python wrapper complexity | Buffer alloc + reduction loop | Just call kernel |

## Open questions

1. **Tensor allocation inside kernel**: The split-k example allocates `torch.zeros(...)` inside the kernel body. CLAUDE.md notes a `TensorOperationInWrapper` warning for tensor ops outside `hl.tile` loops. Need to verify whether this is just a warning or blocks compilation.

2. **Stage 2 argmax over `:` slice**: The split-k example reduces with `torch.sum(tmp[tile_m, tile_n, :], dim=-1)`. Our stage 2 needs `torch.argmax(tile_maxs[:, tile_h], dim=0)` followed by indexing into `tile_max_idxs`. This is more complex than a simple sum — need to verify Helion generates correct code for this pattern.

3. **`hl.barrier()` version support**: The example is in official Helion docs, but we should confirm the installed version supports it. Check with `python -c "import helion.language as hl; print(hasattr(hl, 'barrier'))"`.

4. **Interaction with 2D tiling**: Stage 1 tiles both V and H, producing `tile_maxs` indexed by `tile_v.id`. The barrier must wait for all `(V, H)` tiles before stage 2 runs. The split-k example has a 3D stage-1 grid and a 2D stage-2 grid, so mixed grid dimensions across barrier should be fine.

## Relevant code

- `src/fused_mm_sampling/helion_impl.py` — current 2-stage implementation
- Helion split-k barrier example: https://helionlang.com/examples/split_k_barrier.html
