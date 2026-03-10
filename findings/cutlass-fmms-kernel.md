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

### Step 4: Row reduction epilogue (max across V) ✅

Replace the add-1 epilogue with `Sm90RowReduction` using `cutlass::maximum<float>`.
This reduces across the M/V dimension and outputs one max value per N/H column.
Validates the reduction mechanism before adding argmax complexity.

**Done.** Key implementation details:
- CUTLASS naming: "Row reduction" = reduce across M (rows), output shape [N]. "Col reduction" = reduce across N (columns), output shape [M]. We need row reduction since M=V is the dimension to reduce.
- EVT tree: `Sm90Compute<Identity>` as root (stores full logits to D), with inner `Sm90EVT<Sm90RowReduction<maximum>, Sm90AccFetch>`. The reduction is a pass-through that stores to `ptr_row` while forwarding values to the root.
- `Sm90RowReduction` requires workspace for a reduction buffer and tile counters (coordinating partial reductions across CTA tiles). The GEMM adapter handles workspace allocation automatically.
- Reduction identity for `maximum` is `-infinity` (`-std::numeric_limits<float>::infinity()`).
- Stride for row reduction output: `Stride<_0, _1, _0>` (M=0 reduced, N=1 stride-1, L=0).
- Thread args follow the tree: `{inner_evt_args{acc_fetch_args, row_reduction_args}, identity_compute_args}`.
- Verified on Modal H100: max_err=0.0014 for V=151,936, D=4,096, H=4.

### Step 5: Full FMMS epilogue

Build the custom `VisitorRowArgmax` visitor. Split into sub-steps:

**5a: Argmax reduction (no noise, no temperature).** ✅ Track `(value, index)` pairs in the reduction. Validate indices are correct against a reference matmul + argmax.

**Done.** Key implementation details:
- Custom `Sm90RowArgmax` EVT visitor modeled after `Sm90RowReduction`. Tracks `ValIdx {float val; int32_t idx}` pairs (8 bytes = uint64_t for warp shuffle).
- 2-stage approach: CUTLASS GEMM with EVT epilogue does per-CTA-tile argmax (no intermediate [V,H] logits buffer). Python reduces across V-tiles.
- Critical bug fix: `visit()` receives fragment values in accumulator (source) layout, but `args.tCcD` provides coordinates in destination (output) layout. These layouts differ on SM90. Fix: create a separate coordinate tensor with `sm90_partition_for_epilogue<true>(identity_tensor, ...)` to get accumulator-layout M coordinates. Keep `args.tCcD` for bounds checking only (bounds checking doesn't need correct M coordinates).
- Per-tile results written to explicit output arrays via atomic-counter-gated `end()` callback.
- Warp shuffle reduction uses `__shfl_xor_sync` / `__shfl_down_sync` with `uint64_t` reinterpretation of `ValIdx`.
- Verified on Modal H100: all tests pass (V=256/1024/151936, H=1/4/7).

**5b: Add temperature scaling.** ✅ Pass `inv_temperature` as a parameter to the argmax visitor. Verify logits are scaled correctly before argmax.

**Done.** Key implementation details:
- Added `inv_temperature` field to `Sm90RowArgmax::Arguments` and `Params` with default `1.0f` (backward compatible with existing no-temperature tests).
- Scaling applied in `visit()`: `float value = static_cast<float>(frg_input[i]) * params.inv_temperature`.
- Simpler than using `Sm90ScalarBroadcast` in the EVT tree (no extra EVT node, no restructuring).
- Python wrapper computes `inv_temperature = 1.0 / temperature` and passes it to the C++ function.
- Verified on Modal H100: all tests pass (V=256/1024/151936, H=1/4/7, T=0.5/1.0).

**5c: Add Gumbel noise with Philox RNG.** Derive global `(v, h)` coordinates from the epilogue's coordinate tensors to seed the RNG deterministically. Use the existing chi-squared distribution tests to prove sampling correctness.

**Done.** Key implementation details:
- Lightweight Philox-2x32-10 RNG implemented directly in `sm90_row_argmax.hpp` (avoids expensive `curand_init` per element). Uses `(seed, global_m, global_n)` as counter/key inputs.
- `gumbel_noise()` function: generates uniform float via Philox, clamps to `[1e-10, 1)`, applies double-log transform.
- Gumbel noise is always applied (not optional). The seed controls RNG determinism.
- Global N coordinate computed from `tCcAcc_mn` (accumulator-layout coordinates), same approach as global M.
- New `fused-cutlass-evt` provider registered in `core.py`, uses `fused_mm_sample_cutlass_evt()` which runs the EVT GEMM once per sample.
- Sampling correctness validated via chi-squared distribution tests on Modal H100 (`test_evt --test sampling-evt`).
- `inv_temperature` passed as a 0-d GPU tensor (device pointer). The visitor reads it via `__ldg()`, cached in a register in the `ConsumerStoreCallbacks` constructor. No GPU-CPU synchronization.
- Philox RNG uses CUDA's built-in `curand_Philox4x32_10` (from `curand_philox4x32_x.h`) for correctness.
- JIT cache invalidation: header content hash is included in the extension name (`fmms_cutlass_{hash}`) so changes to `.hpp` files trigger recompilation (torch only tracks files listed in `sources=`).

**Performance (V=151,936, D=4,096, H=4, Modal H100):**
- fused-triton: 0.428 ms
- fused-cutlass (2-kernel): 0.490 ms
- fused-cutlass-evt: 0.626 ms

The EVT path is 0.136ms slower than the 2-kernel path despite eliminating the sampling kernel. Root causes:
1. **D output write is unavoidable.** `void` as ElementD causes compile errors in the TmaWarpSpecialized epilogue builder (`sm90_builder.inl` tries to compute smem layout for void element type, hitting a `% 0` divide). The dummy `[V, H]` logits buffer is still written.
2. **EVT epilogue overhead.** The visitor's reduction workspace writes, warp shuffles, and per-tile gmem copies add latency inside the epilogue pipeline. The separate `gumbel_argmax_kernel` is simpler and only adds ~0.008ms to the GEMM.

**Tile shape investigation:** Originally the EVT used `TileShape_EVT` (64x64x64) because `TmaWarpSpecialized` hard-caps epilogue M at 64. Switching to `TmaWarpSpecializedCooperative` allows the full 128x128x64 tile (cooperative supports epilogue M=128). This improved performance from 0.690ms to 0.626ms. The previous 0.690ms result was from a stale JIT cache that loaded the old 64x64 kernel.

The EVT approach cannot beat the 2-kernel path until the D output write is eliminated. CUTLASS 4.3.5 documents `void` D support but the implementation has a bug with `TmaWarpSpecialized`/`TmaWarpSpecializedCooperative` + EVT. Possible workarounds: (a) use `Sm90NoSmemWarpSpecialized` epilogue schedule, (b) patch the builder to skip smem layout computation for void elements, (c) upgrade CUTLASS.

### Step 6: Integrate EVT epilogue into the `fused-cutlass` provider

Blocked on eliminating the D output write. Until then, the 2-kernel path is faster. Keep the EVT code for future use when the D store can be disabled.

### Known issues to fix early

- ~~`cudaMemcpy` on line 186 of `fmms_cutlass_kernel.cu` reads temperature GPU→CPU. This is a sync point. Pass as a device pointer or kernel argument instead.~~ **Fixed.** The sampling kernel now takes `const float* temperature_ptr` and reads via `__ldg()` on the GPU. No CPU-GPU synchronization.

### Fallback

If the custom visitor proves too fragile or hard to maintain, keep the 2-kernel approach (SM90 GEMM + separate sampling kernel).
The 2-kernel version is already within 7% of Triton and much simpler.

## Reference

- EVT tutorial: https://research.colfax-intl.com/epilogue_visitor_tree/
- Sm90EVT header: `epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp`
- Alignment requirement: bf16 needs `AlignmentA=AlignmentB=8` (16 bytes) for `cp_async`
