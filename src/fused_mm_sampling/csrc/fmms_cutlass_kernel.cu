// Fused Matrix-Multiply & Sampling (FMMS) — CUTLASS implementation
//
// Single CUTLASS GEMM for the full matmul, then a tiled Gumbel-max
// sampling kernel on the resulting logits.
//
// Two-stage approach (same as Triton/CUDA variants):
//   Stage 1a: CUTLASS GEMM — logits = W @ hs.T  (single launch)
//   Stage 1b: Custom kernel — scale by temperature, add Gumbel noise, tiled argmax
//   Stage 2:  Python-side reduction across V-tiles
//
// Requires: CUTLASS 3.x headers (pip install nvidia-cutlass or set CUTLASS_PATH)

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>

// CUTLASS includes
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/epilogue/thread/linear_combination.h>

// ──────────────────────── constants ────────────────────────

static constexpr int TILE_V = 128;   // V-tile size for sampling (matches other implementations)
static constexpr int SAMPLING_THREADS = 256;

// ──────────────────────── CUTLASS GEMM type aliases ────────────────────────

// GEMM: [V, D] x [D, H] -> [V, H]
// A = weights: [V, D], row-major, bfloat16
// B = hidden_states^T: [D, H], column-major (i.e. hidden_states [H, D] row-major)
// C = logits: [V, H], row-major, float32

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = float;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;     // W [V, D]
using LayoutB = cutlass::layout::ColumnMajor;   // hs [H, D] row-major = [D, H] col-major
using LayoutC = cutlass::layout::RowMajor;      // logits [V, H]

// Alignment of 8 elements (16 bytes for bf16) for efficient cp_async in the
// multistage pipeline. This constrains the K dimension (D) to be divisible by 8,
// which holds for all LLM hidden sizes (2048, 4096, 8192, ...).
static constexpr int kAlignmentA = 8;
static constexpr int kAlignmentB = 8;
static constexpr int kEpilogueElements = 1;

using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,          // A matrix
    ElementB, LayoutB,          // B matrix
    ElementC, LayoutC,          // C matrix
    ElementAccumulator,         // Accumulator
    cutlass::arch::OpClassTensorOp,  // Use tensor cores
    cutlass::arch::Sm80,        // Target SM80 (works on 80, 86, 89, 90)
    cutlass::gemm::GemmShape<64, 64, 32>,    // Threadblock tile
    cutlass::gemm::GemmShape<32, 32, 32>,    // Warp tile
    cutlass::gemm::GemmShape<16, 8, 16>,     // MMA instruction (bf16 tensor core)
    cutlass::epilogue::thread::LinearCombination<
        ElementC,               // Output type
        kEpilogueElements,      // Elements per access (1 for arbitrary H)
        ElementAccumulator,     // Accumulator type
        ElementAccumulator      // Compute type
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,                          // Pipeline stages
    kAlignmentA,                // A alignment
    kAlignmentB                 // B alignment
>;


// ──────────────────── Gumbel noise + tiled argmax kernel ────────────────────

// Operates on the full logits [V, H] buffer. Each thread block handles one
// (tile_id, h_idx) pair, reducing TILE_V rows to a single (max_val, max_idx).
//
// Grid: (n_tiles_v, H)   Block: (SAMPLING_THREADS,)

__global__ void gumbel_argmax_kernel(
    const float* __restrict__ logits,       // [V, H], row-major
    float*       __restrict__ max_out,      // [n_tiles_v, H, num_samples]
    int64_t*     __restrict__ max_out_idx,  // [n_tiles_v, H, num_samples]
    float inv_temperature,
    int V,
    int H,                                  // number of hidden states (stride)
    int num_samples,
    unsigned long long base_seed
) {
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

    // Read temperature on GPU (avoid CPU-GPU sync)
    float inv_temperature;
    {
        float temp_val;
        cudaMemcpy(&temp_val, temperature.data_ptr<float>(), sizeof(float), cudaMemcpyDeviceToHost);
        inv_temperature = 1.0f / temp_val;
    }

    auto stream = at::cuda::getCurrentCUDAStream();

    // ── Stage 1a: Single CUTLASS GEMM for the full matmul ──
    // logits = weights [V, D] @ hidden_states.T [D, H] -> [V, H]
    auto logits = torch::empty({V, H}, torch::dtype(torch::kFloat32).device(weights.device()));

    typename Gemm::Arguments args(
        {V, H, D},                                  // Problem size (M, N, K)
        {reinterpret_cast<ElementA*>(
            weights.data_ptr<at::BFloat16>()),
         D},                                         // A: ptr + stride (row-major ld=D)
        {reinterpret_cast<ElementB*>(
            hidden_states.data_ptr<at::BFloat16>()),
         D},                                         // B: ptr + stride (col-major ld=D)
        {reinterpret_cast<ElementC*>(
            logits.data_ptr<float>()),
         H},                                         // C: ptr + stride
        {reinterpret_cast<ElementC*>(
            logits.data_ptr<float>()),
         H},                                         // D: ptr + stride (same as C)
        {1.0f, 0.0f}                                 // alpha=1, beta=0
    );

    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS GEMM cannot be implemented for these dimensions: ",
        cutlass::cutlassGetStatusString(status));

    status = gemm_op.initialize(args, nullptr, stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS GEMM initialization failed: ",
        cutlass::cutlassGetStatusString(status));

    status = gemm_op(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS GEMM execution failed: ",
        cutlass::cutlassGetStatusString(status));

    // ── Stage 1b: Gumbel noise + tiled argmax ──
    dim3 grid(n_tiles_v, H);
    dim3 block(SAMPLING_THREADS);
    gumbel_argmax_kernel<<<grid, block, 0, stream>>>(
        logits.data_ptr<float>(),
        max_out.data_ptr<float>(),
        max_out_idx.data_ptr<int64_t>(),
        inv_temperature,
        V,
        H,
        num_samples,
        static_cast<unsigned long long>(seed)
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fmms_cutlass_stage1", &fmms_cutlass_stage1,
          "FMMS Stage 1 using CUTLASS GEMM + Gumbel argmax (CUDA)");
}
