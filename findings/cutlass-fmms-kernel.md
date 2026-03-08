# CUTLASS FMMS Kernel

## Goal

Fuse the matmul and Gumbel-max sampling into a single CUTLASS kernel using the Epilogue Visitor Tree (EVT) pattern.
Instead of GEMM → full logits buffer → separate sampling kernel, the CUTLASS epilogue applies temperature scaling, Gumbel noise, and argmax reduction as GEMM tiles are computed, avoiding the intermediate `[V, H]` write/read.

## Implementation plan

Incremental steps, each producing a commit. Start simple, grow complexity.

### Step 1: CUTLASS 3.x matmul ✅

Rewrite the GEMM from CUTLASS 2.x (`cutlass::gemm::device::Gemm`, SM80) to CUTLASS 3.x (`GemmUniversal` with CuTe layouts and TMA, SM90).
The current 2.x kernel works but cannot use Sm90EVT epilogues.
The 3.x API is fundamentally different (collective mainloop, tile schedulers, CuTe tensor layouts), so this is a full rewrite of the GEMM setup.
Keep the existing `gumbel_argmax_kernel` as the sampling stage for now.

**Done.** Key implementation details:
- Uses `CollectiveBuilder` pattern (example 48/49 from CUTLASS) for automatic mainloop/epilogue configuration.
- `AlignmentC=AlignmentD=1` to support arbitrary H (including H=1 for decode). The builder auto-selects a thread-level (non-TMA) epilogue for stores.
- Strides constructed manually since `cutlass::make_cute_packed_stride` is not available in the pip-installed CUTLASS version. The stride types are rank-3 `Stride<int64_t, Int<1>, int64_t>` from `TagToStrideA/B/C` in `cutlass/detail/layout.hpp`.
- Python wrapper requires SM90+ and uses `compute_90a,code=sm_90a` gencode for TMA/WGMMA instructions.
- Also fixed the `cudaMemcpy` GPU-CPU sync (see "Known issues" below).

### Step 2: Run on Modal ✅

Verify the 3.x matmul compiles and runs on Modal H100 (CUDA 13.0, SM90) using the existing speed-test make target.
The `fused-cutlass` provider is already registered in `core.py`, so no Python changes needed.
Fix any compilation issues that arise from the CUDA 13.0 / SM90 environment.

**Done.** Compiled and ran on Modal H100 (CUDA 13.0, SM90). Results (V=151,936, D=4,096, H=4):
- fused-triton: 0.426 ms (median)
- fused-cutlass: 0.482 ms (median, ~13% slower)

Added `nvidia-cutlass` and `cuda-bench` to the Modal image deps in `utils.py`.

### Step 3: Simple EVT epilogue (add 1) ✅

Wire up the Sm90EVT plumbing with a trivial epilogue that adds 1 to every accumulator element:

```cpp
using CustomEVT = Sm90EVT<
    Sm90Compute<cutlass::plus, ElementD, ElementCompute, RoundStyle>,
    Sm90ScalarBroadcast<ElementCompute>,
    Sm90AccFetch
>;
```

This validates the EVT infrastructure works end-to-end without any custom visitors.

**Done.** Key implementation details:
- EVTs require `TmaWarpSpecialized` epilogue schedule (not `EpilogueScheduleAuto`).
- TMA stores need 16B alignment: `AlignmentC=AlignmentD=4` for float32 (4 * 4 bytes). H is padded to a multiple of 4 in the Python wrapper.
- The `EpilogueTileAuto` selection must produce a tile where `EPI_TILE_M >= MMA_TILE_M`. With float32 alignment 4, the auto tile (64x32) is too small for TileShape M=128. Fixed by using `TileShape_EVT = Shape<_64, _64, _64>` for the EVT path.
- EVT thread args follow the recursive tree structure: `{scalar_broadcast_args, acc_fetch_args, compute_args}`.
- Verified on Modal H100: max_err=0.0024 for V=151,936, D=4,096, H=4 (bf16 accumulation error).

### Step 4: Column reduction epilogue (max across V)

Replace the add-1 epilogue with `Sm90ColReduction` using `cutlass::maximum<float>`.
This reduces across the V/M dimension and outputs one value per H/N column.
Validates the reduction dimension is correct before adding argmax complexity.

### Step 5: Full FMMS epilogue

Build the custom `VisitorRowArgmax` visitor. Split into sub-steps:

**5a: Argmax reduction (no noise, no temperature).** Track `(value, index)` pairs in the reduction. Validate indices are correct against a reference matmul + argmax.

**5b: Add temperature scaling.** Pass `inv_temperature` as a scalar broadcast. Verify logits are scaled correctly.

**5c: Add Gumbel noise with Philox RNG.** Derive global `(v, h)` coordinates from the epilogue's coordinate tensors to seed the RNG deterministically. Use the existing chi-squared distribution tests to prove sampling correctness.

### Step 6: Profile with NCU

Run Nsight Compute on the fused kernel. Key metrics:
- Memory throughput (should be near roofline, decode matmul is memory-bound)
- Verify zero intermediate `[V, H]` buffer writes
- Compare against the Triton kernel (1.42ms baseline)

Iterate on the kernel based on NCU feedback.

### Known issues to fix early

- ~~`cudaMemcpy` on line 186 of `fmms_cutlass_kernel.cu` reads temperature GPU→CPU. This is a sync point. Pass as a device pointer or kernel argument instead.~~ **Fixed.** The sampling kernel now takes `const float* temperature_ptr` and reads via `__ldg()` on the GPU. No CPU-GPU synchronization.

### Fallback

If the custom visitor proves too fragile or hard to maintain, keep the 2-kernel approach (SM90 GEMM + separate sampling kernel).
The 2-kernel version is already within 7% of Triton and much simpler.

## Reference

- EVT tutorial: https://research.colfax-intl.com/epilogue_visitor_tree/
- Sm90EVT header: `epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp`
- Alignment requirement: bf16 needs `AlignmentA=AlignmentB=8` (16 bytes) for `cp_async`
