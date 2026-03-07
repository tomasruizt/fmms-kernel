// Fused Matrix-Multiply & Sampling (FMMS) — CUTLASS implementation
//
// Uses CUTLASS GEMM with tensor cores for the matmul, then a custom
// Gumbel-max sampling kernel on the resulting logits tile.
//
// Two-stage approach (same as Triton/CUDA variants):
//   Stage 1a: CUTLASS GEMM — logits_tile = W_tile @ hs.T  (per V-tile)
//   Stage 1b: Custom kernel — scale by temperature, add Gumbel noise, argmax
//   Stage 2:  Python-side reduction across V-tiles
//
// Requires: CUTLASS 3.x headers (pip install nvidia-cutlass or set CUTLASS_PATH)

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>

// CUTLASS includes
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/epilogue/thread/linear_combination.h>

// ──────────────────────── constants ────────────────────────

static constexpr int TILE_V = 128;   // V-tile size (matches other implementations)
static constexpr int SAMPLING_THREADS = 256;

// ──────────────────────── CUTLASS GEMM type aliases ────────────────────────

// GEMM: [TILE_V, D] x [D, H] -> [TILE_V, H]
// A = weights_tile: [TILE_V, D], row-major, bfloat16
// B = hidden_states^T: [D, H], column-major (i.e. hidden_states [H, D] row-major)
// C = logits_tile: [TILE_V, H], row-major, float32

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = float;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;     // W_tile [TILE_V, D]
using LayoutB = cutlass::layout::ColumnMajor;   // hs [H, D] row-major = [D, H] col-major
using LayoutC = cutlass::layout::RowMajor;      // logits [TILE_V, H]

// Use the default GEMM configuration for SM80+ (tensor cores)
// ThreadblockShape, WarpShape, InstructionShape are tuned for bf16 on Ampere/Hopper
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
        128 / cutlass::sizeof_bits<ElementC>::value,  // Elements per access
        ElementAccumulator,     // Accumulator type
        ElementAccumulator      // Compute type
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3                           // Pipeline stages
>;


// ──────────────────── Gumbel noise + argmax kernel ────────────────────

// After CUTLASS GEMM produces logits_tile [TILE_V, H], this kernel:
//   1. Scales by 1/temperature
//   2. Adds Gumbel noise: -log(-log(u)), u ~ Uniform(0,1)
//   3. Computes argmax across the V-tile dimension
//   4. Writes (max_val, max_idx) for each (h, sample)
//
// Grid: (H,)   Block: (SAMPLING_THREADS,)
// Each thread block handles one hidden state across all V in the tile.

__global__ void gumbel_argmax_kernel(
    const float* __restrict__ logits_tile,  // [tile_v_size, H], row-major
    float*       __restrict__ max_out,      // [H, num_samples]
    int64_t*     __restrict__ max_out_idx,  // [H, num_samples]
    float inv_temperature,
    int tile_v_size,                        // actual rows in this tile (<= TILE_V)
    int H,                                  // number of hidden states (stride)
    int num_samples,
    int v_offset,                           // global vocab offset for this tile
    unsigned long long base_seed,
    int tile_id                             // for unique RNG sequences
) {
    const int h_idx = blockIdx.x;
    if (h_idx >= H) return;

    const int tid = threadIdx.x;

    // Each sample is processed sequentially (num_samples is typically 1)
    for (int s = 0; s < num_samples; s++) {
        float best_val = -INFINITY;
        int best_global_idx = -1;

        // Each thread processes multiple V rows in a strided loop
        for (int v = tid; v < tile_v_size; v += SAMPLING_THREADS) {
            float logit = logits_tile[v * H + h_idx] * inv_temperature;

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
            int64_t out_offset = (int64_t)h_idx * num_samples + s;
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

    // Allocate workspace for one tile's logits
    auto logits_tile = torch::empty({TILE_V, H}, torch::dtype(torch::kFloat32).device(weights.device()));

    auto stream = at::cuda::getCurrentCUDAStream();

    for (int tile_id = 0; tile_id < n_tiles_v; tile_id++) {
        int v_offset = tile_id * TILE_V;
        int tile_v_size = std::min(TILE_V, V - v_offset);

        // ── Stage 1a: CUTLASS GEMM ──
        // A = weights_tile [tile_v_size, D] row-major (pointer offset into weights)
        // B = hidden_states [H, D] row-major = [D, H] column-major
        // C = logits_tile [tile_v_size, H] row-major

        typename Gemm::Arguments args(
            {tile_v_size, H, D},                    // Problem size (M, N, K)
            {reinterpret_cast<ElementA*>(
                weights.data_ptr<at::BFloat16>()) + (int64_t)v_offset * D,
             D},                                     // A: ptr + stride (row-major ld=D)
            {reinterpret_cast<ElementB*>(
                hidden_states.data_ptr<at::BFloat16>()),
             D},                                     // B: ptr + stride (col-major ld=D)
            {reinterpret_cast<ElementC*>(
                logits_tile.data_ptr<float>()),
             H},                                     // C: ptr + stride
            {reinterpret_cast<ElementC*>(
                logits_tile.data_ptr<float>()),
             H},                                     // D: ptr + stride (same as C)
            {1.0f, 0.0f}                             // alpha=1, beta=0
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

        // ── Stage 1b: Gumbel noise + argmax ──
        // max_out layout: [n_tiles_v, H, num_samples]
        // For this tile: offset = tile_id * H * num_samples
        float* tile_max_out = max_out.data_ptr<float>()
            + (int64_t)tile_id * H * num_samples;
        int64_t* tile_max_out_idx = max_out_idx.data_ptr<int64_t>()
            + (int64_t)tile_id * H * num_samples;

        dim3 grid(H);
        dim3 block(SAMPLING_THREADS);
        gumbel_argmax_kernel<<<grid, block, 0, stream>>>(
            logits_tile.data_ptr<float>(),
            tile_max_out,
            tile_max_out_idx,
            inv_temperature,
            tile_v_size,
            H,
            num_samples,
            v_offset,
            static_cast<unsigned long long>(seed),
            tile_id
        );
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fmms_cutlass_stage1", &fmms_cutlass_stage1,
          "FMMS Stage 1 using CUTLASS GEMM + Gumbel argmax (CUDA)");
}
