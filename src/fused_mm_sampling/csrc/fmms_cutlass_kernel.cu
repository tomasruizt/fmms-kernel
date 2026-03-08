// Fused Matrix-Multiply & Sampling (FMMS) — CUTLASS 3.x implementation (SM90)
//
// Uses CUTLASS 3.x GemmUniversal with collective builders and TMA for the
// matmul, followed by a tiled Gumbel-max sampling kernel.
//
// Two-stage approach:
//   Stage 1a: CUTLASS 3.x GEMM (logits = W @ hs.T)
//   Stage 1b: Custom kernel (scale by temperature, add Gumbel noise, tiled argmax)
//   Stage 2:  Python-side reduction across V-tiles
//
// Requires: SM90+ (H100), CUTLASS 3.x headers (pip install nvidia-cutlass)

#include <limits>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>

// CUTLASS 3.x includes
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
// EVT (Epilogue Visitor Tree) includes
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/thread/activation.h"  // Identity<T>

using namespace cute;

// ──────────────────────── constants ────────────────────────

static constexpr int TILE_V = 128;
static constexpr int SAMPLING_THREADS = 256;

// ──────────────────────── CUTLASS 3.x GEMM type aliases ────────────────────────

// GEMM: [V, D] x [D, H] -> [V, H]   (M=V, N=H, K=D)
// A = weights:       [V, D] row-major,   bfloat16
// B = hidden_states: [H, D] row-major = [D, H] col-major, bfloat16
// D = logits:        [V, H] row-major,   float32

using ElementA           = cutlass::bfloat16_t;
using ElementB           = cutlass::bfloat16_t;
using ElementC           = float;
using ElementD           = float;
using ElementAccumulator = float;
using ElementCompute     = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// 16B alignment for bf16 TMA loads (8 elements x 2 bytes)
static constexpr int AlignmentA = 8;
static constexpr int AlignmentB = 8;
// Alignment of 1 to support arbitrary H (including H=1 for decode)
static constexpr int AlignmentC = 1;
static constexpr int AlignmentD = 1;

using TileShape    = Shape<_128, _128, _64>;
using ClusterShape = Shape<_1, _1, _1>;

// Build the epilogue collective (D = alpha * Acc + beta * C)
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

// Build the mainloop collective (auto stage count, carves out epilogue smem)
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

// ──────────────────── EVT GEMM: D = acc + scalar ────────────────────
//
// Epilogue Visitor Tree that adds a scalar to every accumulator element.
// Used to validate the EVT infrastructure before building the full FMMS epilogue.
//
// EVTs require TMA warp-specialized epilogue, which needs 16B-aligned stores.
// For float32 output, AlignmentD >= 4 (4 * 4 bytes = 16 bytes).
// H must be padded to a multiple of 4 in the Python wrapper.

// EVT requires TMA warp-specialized epilogue, which requires 16B-aligned stores.
// For float32 output: AlignmentD >= 4 (4 * 4B = 16B).
// The epilogue tile auto-selection must produce a tile where EPI_TILE_M >= MMA_TILE_M.
// With float32 and alignment 4, the auto tile is too small for TileShape M=128.
// Use a smaller CTA tile (64x64x64) for the EVT path to avoid this mismatch.
static constexpr int AlignmentC_EVT = 4;
static constexpr int AlignmentD_EVT = 4;
static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

using TileShape_EVT = Shape<_64, _64, _64>;

// D = acc + scalar
using CustomEVT = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<cutlass::plus, ElementD, ElementCompute, RoundStyle>,
    cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementCompute>,  // scalar value
    cutlass::epilogue::fusion::Sm90AccFetch                          // accumulator
>;

using CollectiveEpilogue_EVT = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_EVT, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC_EVT,
    ElementD, LayoutD, AlignmentD_EVT,
    cutlass::epilogue::TmaWarpSpecialized,
    CustomEVT
>::CollectiveOp;

using CollectiveMainloop_EVT = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape_EVT, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue_EVT::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel_EVT = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int>,
    CollectiveMainloop_EVT,
    CollectiveEpilogue_EVT
>;

using Gemm_EVT = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_EVT>;

using StrideC_EVT = typename Gemm_EVT::GemmKernel::StrideC;
using StrideD_EVT = typename Gemm_EVT::GemmKernel::StrideD;


// ──────────────────── EVT GEMM: D = acc, row_max = max(acc, dim=M) ────────────────────
//
// Row reduction: reduces across M (V dimension) using maximum, producing one
// value per N (H) column.  The D store writes the full logits matrix so we can
// cross-check the reduction result.
//
// EVT tree:
//   root: Sm90Compute<Identity>  →  stores to D (pass-through)
//     └── Sm90EVT
//           ├── Sm90RowReduction<maximum>  →  stores to ptr_row
//           └── Sm90AccFetch               →  provides accumulator values

using RowReduceEVT = cutlass::epilogue::fusion::Sm90EVT<
    // root: identity — stores to D, passes values through unchanged
    cutlass::epilogue::fusion::Sm90Compute<
        cutlass::epilogue::thread::Identity, ElementD, ElementCompute, RoundStyle>,
    // child 0: inner EVT with row reduction + acc fetch
    cutlass::epilogue::fusion::Sm90EVT<
        cutlass::epilogue::fusion::Sm90RowReduction<
            cutlass::maximum,       // register-level reduce
            cutlass::maximum,       // warp-level shuffle reduce
            cutlass::maximum,       // global memory reduce
            0,                      // stages (must be 0)
            TileShape_EVT,
            ElementD,               // output element type
            ElementCompute,         // compute element type
            RoundStyle,
            Stride<_0, _1, _0>,     // StrideMNL: M=0 (reduced), N=1, L=0
            AlignmentD_EVT          // alignment (4 for float32 → 16B)
        >,
        cutlass::epilogue::fusion::Sm90AccFetch
    >
>;

using CollectiveEpilogue_RowReduce = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_EVT, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC_EVT,
    ElementD, LayoutD, AlignmentD_EVT,
    cutlass::epilogue::TmaWarpSpecialized,
    RowReduceEVT
>::CollectiveOp;

using CollectiveMainloop_RowReduce = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape_EVT, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue_RowReduce::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel_RowReduce = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int>,
    CollectiveMainloop_RowReduce,
    CollectiveEpilogue_RowReduce
>;

using Gemm_RowReduce = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_RowReduce>;

using StrideC_RowReduce = typename Gemm_RowReduce::GemmKernel::StrideC;
using StrideD_RowReduce = typename Gemm_RowReduce::GemmKernel::StrideD;


// ──────────────────── Gumbel noise + tiled argmax kernel ────────────────────

// Operates on the full logits [V, H] buffer. Each thread block handles one
// (tile_id, h_idx) pair, reducing TILE_V rows to a single (max_val, max_idx).
//
// Grid: (n_tiles_v, H)   Block: (SAMPLING_THREADS,)

__global__ void gumbel_argmax_kernel(
    const float* __restrict__ logits,       // [V, H], row-major
    float*       __restrict__ max_out,      // [n_tiles_v, H, num_samples]
    int64_t*     __restrict__ max_out_idx,  // [n_tiles_v, H, num_samples]
    const float* __restrict__ temperature_ptr,  // 0-d tensor on GPU (no CPU-GPU sync)
    int V,
    int H,
    int num_samples,
    unsigned long long base_seed
) {
    const float inv_temperature = 1.0f / __ldg(temperature_ptr);
    const int tile_id = blockIdx.x;
    const int h_idx = blockIdx.y;
    if (h_idx >= H) return;

    const int v_offset = tile_id * TILE_V;
    const int tile_v_size = min(TILE_V, V - v_offset);
    const int tid = threadIdx.x;

    for (int s = 0; s < num_samples; s++) {
        float best_val = -INFINITY;
        int best_global_idx = -1;

        for (int v = tid; v < tile_v_size; v += SAMPLING_THREADS) {
            float logit = logits[(v_offset + v) * H + h_idx] * inv_temperature;

            // Gumbel noise
            unsigned long long seq = (unsigned long long)tile_id * 100000ULL
                + (unsigned long long)h_idx * 1000ULL
                + (unsigned long long)s * 10ULL
                + (unsigned long long)v;
            curandStatePhilox4_32_10_t state;
            curand_init(base_seed, seq, 0, &state);
            float u = curand_uniform(&state);
            u = fmaxf(u, 1e-10f);
            float gumbel = -logf(-logf(u));
            float noisy_logit = logit + gumbel;

            if (noisy_logit > best_val) {
                best_val = noisy_logit;
                best_global_idx = v_offset + v;
            }
        }

        // Block-level reduction via shared memory
        __shared__ float smem_vals[SAMPLING_THREADS];
        __shared__ int smem_idxs[SAMPLING_THREADS];
        smem_vals[tid] = best_val;
        smem_idxs[tid] = best_global_idx;
        __syncthreads();

        // Tree reduction
        for (int stride = SAMPLING_THREADS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                if (smem_vals[tid + stride] > smem_vals[tid]) {
                    smem_vals[tid] = smem_vals[tid + stride];
                    smem_idxs[tid] = smem_idxs[tid + stride];
                }
            }
            __syncthreads();
        }

        // Thread 0 writes the result
        if (tid == 0) {
            int64_t out_offset = ((int64_t)tile_id * H + h_idx) * num_samples + s;
            max_out[out_offset] = smem_vals[0];
            max_out_idx[out_offset] = smem_idxs[0];
        }
        __syncthreads();
    }
}


// ──────────────────────── host wrapper ────────────────────────

void fmms_cutlass_stage1(
    torch::Tensor weights,        // [V, D] bfloat16
    torch::Tensor hidden_states,  // [H, D] bfloat16
    torch::Tensor max_out,        // [n_tiles_v, H, num_samples] float32
    torch::Tensor max_out_idx,    // [n_tiles_v, H, num_samples] int64
    torch::Tensor temperature,    // 0-d float32
    int64_t seed
) {
    TORCH_CHECK(weights.is_cuda(), "weights must be CUDA");
    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be CUDA");
    TORCH_CHECK(weights.dtype() == torch::kBFloat16, "weights must be bfloat16");
    TORCH_CHECK(hidden_states.dtype() == torch::kBFloat16, "hidden_states must be bfloat16");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(hidden_states.is_contiguous(), "hidden_states must be contiguous");
    TORCH_CHECK(temperature.is_cuda(), "temperature must be CUDA");

    int V = weights.size(0);
    int D = weights.size(1);
    int H = hidden_states.size(0);
    TORCH_CHECK(hidden_states.size(1) == D,
        "hidden_states dim 1 (", hidden_states.size(1), ") != weights dim 1 (", D, ")");

    int num_samples = max_out.size(2);
    int n_tiles_v = (V + TILE_V - 1) / TILE_V;

    auto stream = at::cuda::getCurrentCUDAStream();

    // ── Stage 1a: CUTLASS 3.x GEMM ──
    // logits = weights [V, D] @ hidden_states.T [D, H] -> [V, H]
    auto logits = torch::empty({V, H}, torch::dtype(torch::kFloat32).device(weights.device()));

    // Construct packed strides manually (make_cute_packed_stride not available
    // in this CUTLASS version). Stride types are rank-3: [primary, secondary, batch].
    // Int<1> elements are compile-time constants (stride-1 dimension).
    StrideA stride_A{int64_t(D), {}, int64_t(0)};   // A [V, D] RowMajor: V_stride=D, D_stride=1
    StrideB stride_B{int64_t(D), {}, int64_t(0)};   // B [H, D] ColMajor: H_stride=D, D_stride=1
    StrideC stride_C{int64_t(H), {}, int64_t(0)};   // C [V, H] RowMajor: V_stride=H, H_stride=1
    StrideD stride_D{int64_t(H), {}, int64_t(0)};   // D [V, H] RowMajor: V_stride=H, H_stride=1

    int device_id = weights.get_device();
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device_id;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {V, H, D},                                     // problem shape (M, N, K)
        {                                               // mainloop args
            reinterpret_cast<ElementA const*>(weights.data_ptr<at::BFloat16>()),
            stride_A,
            reinterpret_cast<ElementB const*>(hidden_states.data_ptr<at::BFloat16>()),
            stride_B
        },
        {                                               // epilogue args
            {},                                         // thread args (set below)
            reinterpret_cast<ElementC const*>(logits.data_ptr<float>()),  // C ptr (unused, beta=0)
            stride_C,
            reinterpret_cast<ElementD*>(logits.data_ptr<float>()),        // D ptr (output)
            stride_D
        },
        hw_info
    };
    arguments.epilogue.thread.alpha = 1.0f;
    arguments.epilogue.thread.beta = 0.0f;

    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS 3.x GEMM cannot be implemented: ",
        cutlass::cutlassGetStatusString(status));

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    auto workspace = torch::empty(
        {static_cast<int64_t>(workspace_size)},
        torch::dtype(torch::kUInt8).device(weights.device()));

    status = gemm_op.initialize(arguments, workspace.data_ptr(), stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS 3.x GEMM initialization failed: ",
        cutlass::cutlassGetStatusString(status));

    status = gemm_op.run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS 3.x GEMM execution failed: ",
        cutlass::cutlassGetStatusString(status));

    // ── Stage 1b: Gumbel noise + tiled argmax ──
    dim3 grid(n_tiles_v, H);
    dim3 block(SAMPLING_THREADS);
    gumbel_argmax_kernel<<<grid, block, 0, stream>>>(
        logits.data_ptr<float>(),
        max_out.data_ptr<float>(),
        max_out_idx.data_ptr<int64_t>(),
        temperature.data_ptr<float>(),
        V,
        H,
        num_samples,
        static_cast<unsigned long long>(seed)
    );
}


// ──────────────────────── EVT test: D = acc + 1 ────────────────────────

torch::Tensor test_evt_add1(
    torch::Tensor weights,        // [V, D] bfloat16
    torch::Tensor hidden_states   // [H, D] bfloat16
) {
    TORCH_CHECK(weights.is_cuda(), "weights must be CUDA");
    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be CUDA");
    TORCH_CHECK(weights.dtype() == torch::kBFloat16, "weights must be bfloat16");
    TORCH_CHECK(hidden_states.dtype() == torch::kBFloat16, "hidden_states must be bfloat16");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(hidden_states.is_contiguous(), "hidden_states must be contiguous");

    int V = weights.size(0);
    int D = weights.size(1);
    int H = hidden_states.size(0);
    TORCH_CHECK(hidden_states.size(1) == D,
        "hidden_states dim 1 (", hidden_states.size(1), ") != weights dim 1 (", D, ")");

    // EVT requires H padded to multiple of AlignmentD_EVT (4 for float32)
    int H_padded = ((H + AlignmentD_EVT - 1) / AlignmentD_EVT) * AlignmentD_EVT;
    bool needs_padding = (H_padded != H);

    // Pad hidden_states if needed (zero-pad doesn't affect matmul results for original columns)
    torch::Tensor hs_padded = hidden_states;
    if (needs_padding) {
        hs_padded = torch::zeros({H_padded, D}, hidden_states.options());
        hs_padded.narrow(0, 0, H).copy_(hidden_states);
    }

    auto stream = at::cuda::getCurrentCUDAStream();

    auto logits = torch::empty({V, H_padded}, torch::dtype(torch::kFloat32).device(weights.device()));

    // Reuse StrideA/StrideB from the non-EVT GEMM (same layout for A and B)
    StrideA stride_A{int64_t(D), {}, int64_t(0)};
    StrideA stride_B{int64_t(D), {}, int64_t(0)};
    // EVT strides may differ from non-EVT due to different alignment
    StrideC_EVT stride_C{int64_t(H_padded), {}, int64_t(0)};
    StrideD_EVT stride_D{int64_t(H_padded), {}, int64_t(0)};

    int device_id = weights.get_device();
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device_id;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

    typename Gemm_EVT::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {V, H_padded, D},
        {
            reinterpret_cast<ElementA const*>(weights.data_ptr<at::BFloat16>()),
            stride_A,
            reinterpret_cast<ElementB const*>(hs_padded.data_ptr<at::BFloat16>()),
            stride_B
        },
        {
            {},  // thread args (EVT args, set below)
            nullptr,  // C ptr (unused)
            stride_C,
            reinterpret_cast<ElementD*>(logits.data_ptr<float>()),
            stride_D
        },
        hw_info
    };

    // EVT thread args: {scalar_broadcast_args, acc_fetch_args, compute_args}
    // Sm90ScalarBroadcast args: {{scalar_value}, {scalar_ptr}, {stride}}
    arguments.epilogue.thread = {
        {{1.0f}},  // Sm90ScalarBroadcast: add 1.0f
        {},        // Sm90AccFetch: no args
        {}         // Sm90Compute<plus>: no args
    };

    Gemm_EVT gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS EVT GEMM cannot be implemented: ",
        cutlass::cutlassGetStatusString(status));

    size_t workspace_size = Gemm_EVT::get_workspace_size(arguments);
    auto workspace = torch::empty(
        {static_cast<int64_t>(workspace_size)},
        torch::dtype(torch::kUInt8).device(weights.device()));

    status = gemm_op.initialize(arguments, workspace.data_ptr(), stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS EVT GEMM initialization failed: ",
        cutlass::cutlassGetStatusString(status));

    status = gemm_op.run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS EVT GEMM execution failed: ",
        cutlass::cutlassGetStatusString(status));

    // Slice back to original H if we padded
    if (needs_padding) {
        return logits.narrow(1, 0, H).contiguous();
    }
    return logits;
}


// ──────────────────────── EVT test: row reduce (max across V) ────────────────────────

std::vector<torch::Tensor> test_evt_row_reduce(
    torch::Tensor weights,        // [V, D] bfloat16
    torch::Tensor hidden_states   // [H, D] bfloat16
) {
    TORCH_CHECK(weights.is_cuda(), "weights must be CUDA");
    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be CUDA");
    TORCH_CHECK(weights.dtype() == torch::kBFloat16, "weights must be bfloat16");
    TORCH_CHECK(hidden_states.dtype() == torch::kBFloat16, "hidden_states must be bfloat16");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(hidden_states.is_contiguous(), "hidden_states must be contiguous");

    int V = weights.size(0);
    int D = weights.size(1);
    int H = hidden_states.size(0);
    TORCH_CHECK(hidden_states.size(1) == D,
        "hidden_states dim 1 (", hidden_states.size(1), ") != weights dim 1 (", D, ")");

    // EVT requires H padded to multiple of AlignmentD_EVT (4 for float32)
    int H_padded = ((H + AlignmentD_EVT - 1) / AlignmentD_EVT) * AlignmentD_EVT;
    bool needs_padding = (H_padded != H);

    torch::Tensor hs_padded = hidden_states;
    if (needs_padding) {
        hs_padded = torch::zeros({H_padded, D}, hidden_states.options());
        hs_padded.narrow(0, 0, H).copy_(hidden_states);
    }

    auto stream = at::cuda::getCurrentCUDAStream();

    // D output: full logits [V, H_padded]
    auto logits = torch::empty({V, H_padded}, torch::dtype(torch::kFloat32).device(weights.device()));
    // Row reduction output: max across V, shape [H_padded]
    auto row_max = torch::empty({H_padded}, torch::dtype(torch::kFloat32).device(weights.device()));

    StrideA stride_A{int64_t(D), {}, int64_t(0)};
    StrideA stride_B{int64_t(D), {}, int64_t(0)};
    StrideC_RowReduce stride_C{int64_t(H_padded), {}, int64_t(0)};
    StrideD_RowReduce stride_D{int64_t(H_padded), {}, int64_t(0)};

    int device_id = weights.get_device();
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device_id;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

    typename Gemm_RowReduce::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {V, H_padded, D},
        {
            reinterpret_cast<ElementA const*>(weights.data_ptr<at::BFloat16>()),
            stride_A,
            reinterpret_cast<ElementB const*>(hs_padded.data_ptr<at::BFloat16>()),
            stride_B
        },
        {
            {},  // thread args (EVT args, set below)
            nullptr,  // C ptr (unused)
            stride_C,
            reinterpret_cast<ElementD*>(logits.data_ptr<float>()),
            stride_D
        },
        hw_info
    };

    // EVT thread args follow the tree structure:
    //   outer: (Child0_args, NodeOp_args)
    //   Child0 is inner EVT: (AccFetch_args, RowReduction_args)
    arguments.epilogue.thread = {
        {   // Child 0: inner Sm90EVT
            {},  // AccFetch: no args
            {    // RowReduction: {ptr_row, reduction_identity, dRow}
                reinterpret_cast<ElementD*>(row_max.data_ptr<float>()),
                -std::numeric_limits<ElementCompute>::infinity(),
                {}   // Stride<_0, _1, _0>{}
            }
        },
        {}   // NodeOp: Compute<Identity> — no args
    };

    Gemm_RowReduce gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS RowReduce GEMM cannot be implemented: ",
        cutlass::cutlassGetStatusString(status));

    size_t workspace_size = Gemm_RowReduce::get_workspace_size(arguments);
    auto workspace = torch::empty(
        {static_cast<int64_t>(workspace_size)},
        torch::dtype(torch::kUInt8).device(weights.device()));

    status = gemm_op.initialize(arguments, workspace.data_ptr(), stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS RowReduce GEMM initialization failed: ",
        cutlass::cutlassGetStatusString(status));

    status = gemm_op.run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS RowReduce GEMM execution failed: ",
        cutlass::cutlassGetStatusString(status));

    // Slice back to original H if we padded
    if (needs_padding) {
        return {logits.narrow(1, 0, H).contiguous(), row_max.narrow(0, 0, H).contiguous()};
    }
    return {logits, row_max};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fmms_cutlass_stage1", &fmms_cutlass_stage1,
          "FMMS Stage 1 using CUTLASS 3.x GEMM + Gumbel argmax (SM90)");
    m.def("test_evt_add1", &test_evt_add1,
          "Test EVT epilogue: returns matmul(W, hs.T) + 1.0 (SM90)");
    m.def("test_evt_row_reduce", &test_evt_row_reduce,
          "Test EVT epilogue: returns (logits, max_per_column) where max is across V (SM90)");
}
