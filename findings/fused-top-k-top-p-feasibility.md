# Fused top-k and top-p sampling in the FMMS kernel

## Context

The FMMS kernel fuses matrix multiplication with categorical sampling via the
Gumbel-max trick: `sample = argmax_i(logits_i + Gumbel_i)`. Argmax is a
streaming reducible operation — each V-tile computes its local max, a cheap
reduction finds the global winner, and the full `[V, H]` logit matrix is never
materialized. This is the core insight that makes fusion possible.

However, the kernel currently only supports vanilla categorical sampling
(temperature-scaled). Many inference workloads require **top-k** and/or
**top-p** (nucleus) sampling. This document analyzes whether these can be
fused into the kernel.

## FlashInfer's rejection sampling

FlashInfer ([blog post](https://flashinfer.ai/2025/03/10/sampling.html))
replaces sorting-based top-k/top-p with **rejection sampling**:

1. Sample a token via inverse transform sampling (CDF without sorting).
2. Check if it satisfies the top-k or top-p constraint.
3. If not, update a pivot threshold and resample.

A dual-pivot variant guarantees convergence in O(log(1/ε)) rounds. This avoids
full sorting and multiple kernel launches.

**Key limitation for fusion:** each rejection round scans the full probability
distribution. In a fused kernel that would mean re-running the matmul per
round — a non-starter. FlashInfer's approach operates on pre-computed logits.

### `filter_apply_order`: sequential vs joint

`top_k_top_p_sampling_from_logits` in
`.venv/.../flashinfer/sampling.py` (lines 1127–1154) supports two strategies
via the `filter_apply_order` parameter:

**`"top_k_first"` (default):** sequential — three kernel launches.

```python
masked_logits = top_k_mask_logits(logits, top_k)   # 1. mask all but top-k
probs = torch.softmax(masked_logits, dim=-1)        # 2. softmax on survivors
return top_p_sampling_from_probs(probs, top_p, ...)  # 3. rejection-sample with top-p
```

Top-p is evaluated on the **renormalized** distribution after top-k filtering.

**`"joint"`:** single fused CUDA kernel (`TopKTopPSamplingFromProbKernel` in
`sampling.cuh`, ~line 1199). Uses dual-pivot binary search rejection sampling
and accepts a candidate only when **both** constraints are satisfied
simultaneously on the **original** distribution:

```cpp
if (aggregate_gt_pivot_0.count < k && aggregate_gt_pivot_0.value < p) {
    break;  // accept: both top-k and top-p satisfied jointly
}
```

### Sequential vs joint: when results differ

The two strategies can produce different candidate sets because renormalization
after top-k inflates surviving tokens' probabilities.

**Example:** 5 tokens, top_k=3, top_p=0.35

| Token | Original prob |
|-------|---------------|
| A     | 0.30          |
| B     | 0.25          |
| C     | 0.24          |
| D     | 0.11          |
| E     | 0.10          |

**Sequential** (top-k first):

1. Top-k=3: keep {A=0.30, B=0.25, C=0.24}, discard D, E.
2. Renormalize: A=0.380, B=0.316, C=0.304.
3. Top-p=0.35: cumsum A=0.380 ≥ 0.35 → only **{A}** survives.
   Result: deterministic.

**Joint** (both at once on original probs):

- Top-p=0.35: cumsum A=0.30 < 0.35, A+B=0.55 ≥ 0.35 → **{A, B}**.
- Top-k=3: {A, B, C}.
- Intersection: **{A, B}**.
  Result: still stochastic.

Renormalization inflated A from 0.30 to 0.38, crossing the 0.35 threshold
alone. On the original distribution A doesn't reach 0.35, so B is also needed.
In general, **sequential can produce a smaller candidate set** than joint.

## Top-k: feasible via tile-local top-k + merge

Top-k restricts sampling to the k tokens with the highest logits. The key
observation: the tile-local max can be generalized to a **tile-local top-k**.

### Single-pass algorithm

1. **Stage 1 (fused with matmul):** Each V-tile computes logits as now, but
   outputs its **local top-k** values + indices instead of just the top-1.
   Finding top-k within a tile of `BLOCK_SIZE_V` (128–256) elements for small
   k is cheap — partial sort within registers, or `tl.sort` on the tile
   followed by a slice.

2. **Stage 2 (merge):** Merge all per-tile top-k lists into a global top-k.
   This reduces `num_tiles × k` candidates — still tiny compared to the full
   vocab.

3. **Stage 3 (sample):** Apply Gumbel noise to the k surviving candidates and
   take argmax. Trivially cheap.

### Cost analysis

- **Intermediate storage:** Grows from `num_tiles × H` to
  `num_tiles × k × H`. For V=128K, BLOCK_SIZE_V=256, k=50, H=1: 512 tiles ×
  50 = 25,600 elements. Negligible vs materializing 128K logits.
- **Matmul cost:** Unchanged (dominant cost).
- **Within-tile partial sort:** O(BLOCK_SIZE_V × k) comparisons in registers.
  Small relative to the matmul.

### Engineering challenges

- **No built-in top-k in Triton.** Options:
  - `tl.sort` on the tile logits, then take the first k values. `tl.sort`
    exists in Triton and works on tile-sized data (bitonic sort, O(n log² n)).
  - Manual insertion sort maintaining k running maxima. Better for small k
    but awkward to express in Triton.
- **Stage 2 merge.** The current stage 2 is a single `max(axis=0)` in PyTorch.
  For top-k, it becomes a merge of sorted lists — still simple but requires
  a custom reduction (sort `num_tiles × k` candidates, take top k, then
  Gumbel-sample).

## Top-p: not directly fusible

Top-p (nucleus sampling) keeps the smallest set of tokens whose cumulative
probability mass exceeds p. This requires three global operations:

1. **Global softmax** — needs the max logit across ALL tiles for numerical
   stability. Requires a full pass before any probabilities can be computed.
2. **Global sorting** — must rank tokens by probability across the full vocab.
3. **Cumulative sum** — accumulate mass from highest to lowest until exceeding
   p.

None of these decompose into independent tile-local work. Fusion would require
multiple passes over the weight matrix (or materializing logits), defeating the
purpose.

## Min-p: partially feasible but needs global max

Min-p filters tokens whose probability is below `min_p × max_probability`.
This requires knowing the global max probability, which in turn requires:

1. A full pass to find the max logit (for softmax stability).
2. Computing `softmax(max_logit)` to get the reference probability.
3. A second pass to filter tokens below the threshold.

The need for two passes makes direct fusion impractical, though the approach is
simpler than top-p since no sorting or cumulative sum is needed.

## Hybrid approach: fused top-k + post-kernel top-p

The most practical path for supporting both:

1. **Fuse top-k into the kernel** (as described above) with a conservatively
   large k (e.g. k=256).
2. **Apply top-p on the k survivors** outside the kernel. On 256 elements,
   softmax + sort + cumsum + resample is trivially fast.

This mirrors how vLLM combines the two filters. vLLM's FlashInfer path
(`vllm/v1/sample/ops/topk_topp_sampler.py`, line 377) calls
`flashinfer.sampling.top_k_top_p_sampling_from_logits(logits, k, p)` without
passing `filter_apply_order=`, so it uses the default `"top_k_first"`
(sequential). vLLM's native PyTorch path (`apply_top_k_top_p()`, lines
243–283) also applies top-k first: sort once, mask by k-th value, then
softmax + cumsum for top-p.

The fusion benefit is that the full logit matrix is still never materialized;
only `num_tiles × k × H` intermediate values are stored.

## Summary

| Strategy   | Fusible? | Difficulty | Notes                                              |
|------------|----------|------------|----------------------------------------------------|
| **Top-k**  | Yes      | Medium     | Local top-k per tile + merge. `tl.sort` on tiles.  |
| **Top-p**  | No       | —          | Requires global softmax + sorted cumsum.           |
| **Top-k + top-p** | Partially | Medium | Fuse top-k, apply top-p on k survivors post-kernel. |
| **Min-p**  | No       | —          | Needs global max logit as threshold reference.     |

The dominant cost in all cases is the matmul, which is unchanged. The overhead
of within-tile top-k and the merge reduction is small for practical k values
(k ≤ 64 covers nearly all real workloads).
