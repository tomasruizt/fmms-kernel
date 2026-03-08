// Custom EVT visitor: per-tile row argmax reduction across the M (vocab) dimension.
//
// Each CTA tile reduces its local M rows to a single (value, index) pair per
// N (H) column. Per-tile results are written to explicit output arrays
// (ptr_tile_vals, ptr_tile_idxs) indexed as [tile_m * H + global_n].
// The cross-tile reduction is done by the host (Python) after the GEMM.
//
// Internally, CTA tiles write partial results to a gmem workspace buffer.
// The last CTA (determined by atomic counters) copies per-tile results
// from the workspace to the explicit output arrays.
//
// The reduction element is ValIdx {float val; int32_t idx;} (8 bytes = uint64_t),
// which fits naturally into the warp shuffle mechanism.

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/workspace.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"

// ──────────────────── Philox RNG for Gumbel noise ────────────────────
//
// Uses curand_Philox4x32_10 directly (10 rounds of Philox) to generate
// one uniform random float per (seed, m, n) triple. This avoids
// curand_init() which does expensive skip-ahead computation per call.

#include <curand_philox4x32_x.h>

__device__ __forceinline__ float gumbel_noise(uint64_t seed, uint32_t m, uint32_t n) {
    uint4 counter = {m, n, 0, 0};
    uint2 key = {static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)};
    uint4 result = curand_Philox4x32_10(counter, key);
    // Convert uint32 to float in (0, 1]: use top 24 bits (float has 24-bit mantissa)
    float u = (result.x >> 8) * (1.0f / 16777216.0f);
    u = fmaxf(u, 1e-10f);
    return -__logf(-__logf(u));
}

namespace cutlass::epilogue::fusion {

using namespace cute;
using namespace detail;

// ──────────────────── Value-Index pair for argmax reduction ────────────────────

struct ValIdx {
    float val;
    int32_t idx;
};

static_assert(sizeof(ValIdx) == sizeof(uint64_t), "ValIdx must be 8 bytes for shuffle");

// Reduction functor: keeps the element with larger val (argmax semantics).
template <class T>
struct ArgmaxReduce;

template <>
struct ArgmaxReduce<ValIdx> {
    CUTLASS_DEVICE ValIdx operator()(ValIdx const& a, ValIdx const& b) const {
        return (b.val > a.val) ? b : a;
    }
};

template <int N>
struct ArgmaxReduce<Array<ValIdx, N>> {
    CUTLASS_DEVICE Array<ValIdx, N> operator()(
            Array<ValIdx, N> const& a, Array<ValIdx, N> const& b) const {
        Array<ValIdx, N> result;
        ArgmaxReduce<ValIdx> scalar_reduce{};
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            result[i] = scalar_reduce(a[i], b[i]);
        }
        return result;
    }
};


// ──────────────────── Sm90RowArgmax visitor (per-tile output) ────────────────────
//
// Produces per-CTA-tile (max_value, argmax_index) pairs written to explicit
// output arrays. The last CTA copies from the workspace to the output.
// No cross-tile reduction — the host reduces across tiles.

template <
    int Stages,
    class CtaTileShapeMNK,
    class ElementOutput,   // int32_t — the argmax index output type
    FloatRoundStyle RoundStyle,
    class StrideMNL = Stride<_0, _1, _0>,
    int Alignment = 128 / sizeof_bits_v<ElementOutput>,
    bool EnableNullptr = true
>
struct Sm90RowArgmax {
private:
    static_assert(Stages == 0, "Smem usage not supported yet");
    static_assert(Alignment * sizeof_bits_v<ElementOutput> % 128 == 0, "sub-16B alignment not supported yet");

    using ElementCompute = ValIdx;

    CUTLASS_HOST_DEVICE
    static ValIdx identity() {
        return {-std::numeric_limits<float>::infinity(), -1};
    }

public:
    struct SharedStorage { };

    struct Arguments {
        void* ptr_row = nullptr;       // unused (kept for EVT API compat)
        StrideMNL dRow = {};
        float* ptr_tile_vals = nullptr;   // [n_tiles_m, H] per-tile max values
        int32_t* ptr_tile_idxs = nullptr; // [n_tiles_m, H] per-tile argmax indices
        int H = 0;                        // number of valid N columns
        const float* inv_temperature_ptr = nullptr;  // device pointer: 1/temperature
        uint64_t seed = 0;               // Philox RNG seed for Gumbel noise
    };

    struct Params {
        void* ptr_row = nullptr;
        StrideMNL dRow = {};
        ValIdx* reduction_buffer = nullptr;
        float* ptr_tile_vals = nullptr;
        int32_t* ptr_tile_idxs = nullptr;
        int H = 0;
        const float* inv_temperature_ptr = nullptr;
        uint64_t seed = 0;
    };

    template <class ProblemShape>
    static constexpr Params
    to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
        ValIdx* reduction_buffer = reinterpret_cast<ValIdx*>(workspace);
        return {args.ptr_row, args.dRow, reduction_buffer,
                args.ptr_tile_vals, args.ptr_tile_idxs, args.H, args.inv_temperature_ptr, args.seed};
    }

    template <class ProblemShape>
    static bool
    can_implement(ProblemShape const& problem_shape, Arguments const& args) {
        return true;
    }

    template <class ProblemShape>
    static size_t
    get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
        auto problem_shape_mnkl = append<4>(problem_shape, 1);
        auto [M, N, K, L] = problem_shape_mnkl;
        auto [tile_M, tile_N, tile_K] = CtaTileShapeMNK{};
        // Reduction buffer: one ValIdx per (tile_m, n_col, tile_n, batch)
        return product(ceil_div(make_shape(size<>(M), size<>(N), L), make_shape(tile_M, tile_N)))
            * tile_N * sizeof(ValIdx);
    }

    template <class ProblemShape>
    static cutlass::Status
    initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
        CudaHostAdapter* cuda_adapter = nullptr) {
        return cutlass::Status::kSuccess;  // no initialization needed
    }

    CUTLASS_DEVICE bool
    is_producer_load_needed() const { return false; }

    CUTLASS_DEVICE bool
    is_C_load_needed() const { return false; }

    CUTLASS_HOST_DEVICE
    Sm90RowArgmax() { }

    CUTLASS_HOST_DEVICE
    Sm90RowArgmax(Params const& params, SharedStorage const& shared_storage)
        : params(params) { }

    Params params;

    template <class... Args>
    CUTLASS_DEVICE auto
    get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
        return EmptyProducerLoadCallbacks{};
    }

    template <class ArgsTuple>
    struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
        CUTLASS_DEVICE
        ConsumerStoreCallbacks(ArgsTuple&& args_tuple, Params const& params)
            : args_tuple(cute::forward<ArgsTuple>(args_tuple)),
              params(params),
              inv_temperature(__ldg(params.inv_temperature_ptr)) {}

        ArgsTuple args_tuple;
        Params const& params;
        float inv_temperature;  // cached in register from device pointer

        // visit(): accumulate (value, global_m_index) pairs into register reduction tensor.
        template <typename ElementAccumulator, typename... ElementInputs, int FragmentSize>
        CUTLASS_DEVICE auto
        visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
              Array<ElementInputs, FragmentSize> const&... frg_inputs) {
            auto& [ref_src, tCrRow, tCcRow, gRow_l, cRow, gBuf_ml, sBuf_layout,
                lane_layout_MN, lane_mn, warp_layout_MN, warp_mn,
                tile_coord_mnkl, residue_cRow, residue_tCcRow, epi_tile, tiled_copy, thread_idx,
                tCcAcc] = args_tuple;
            Tensor tCrRow_mn = tCrRow(_,_,_,epi_m,epi_n);
            Tensor tCcRow_mn = tCcRow(_,_,_,epi_m,epi_n);
            // Accumulator-layout coordinates for correct M index tracking
            Tensor tCcAcc_mn = tCcAcc(_,_,_,epi_m,epi_n);

            using ReduceInput = ArgmaxReduce<ValIdx>;
            ReduceInput reduce_input{};

            auto frg_input = cute::get<0>(cute::make_tuple(frg_inputs...));

            auto [m, n, k, l] = tile_coord_mnkl;
            constexpr int cta_tile_M = decltype(get<0>(CtaTileShapeMNK{}))::value;
            int32_t m_offset = static_cast<int32_t>(m) * cta_tile_M;

            constexpr int cta_tile_N = decltype(get<1>(CtaTileShapeMNK{}))::value;
            int32_t n_offset = static_cast<int32_t>(n) * cta_tile_N;

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < FragmentSize; ++i) {
                int coord_idx = epi_v * FragmentSize + i;
                if (elem_less(tCcRow_mn(coord_idx), residue_tCcRow)) {
                    float value = static_cast<float>(frg_input[i]) * inv_temperature;
                    // Use accumulator-layout coordinates for M and N indices
                    int32_t local_m = static_cast<int32_t>(get<0>(tCcAcc_mn(coord_idx)));
                    int32_t global_m = m_offset + local_m;
                    // Add Gumbel noise for sampling
                    int32_t local_n = static_cast<int32_t>(get<1>(tCcAcc_mn(coord_idx)));
                    int32_t global_n = n_offset + local_n;
                    value += gumbel_noise(params.seed,
                        static_cast<uint32_t>(global_m), static_cast<uint32_t>(global_n));
                    ValIdx candidate = {value, global_m};
                    ValIdx& current = tCrRow_mn(coord_idx);
                    current = reduce_input(current, candidate);
                }
            }

            return frg_input;  // pass through unchanged
        }

        // reduce(): warp shuffle + write per-tile result to gmem workspace.
        // The last CTA (via atomic counter) will copy results to output arrays in end().
        template <class STensor, class SyncFn, class VTensor>
        CUTLASS_DEVICE void
        reduce(STensor&& smem_buffer, SyncFn const& sync_fn, int epi_m, int epi_n,
               bool is_last_iteration, VTensor visit_results) {
            if (not is_last_iteration) {
                return;
            }

            auto& [ref_src, tCrRow, tCcRow, gRow_l, cRow, gBuf_ml, sBuf_layout,
                lane_layout_MN, lane_mn, warp_layout_MN, warp_mn,
                tile_coord_mnkl, residue_cRow, residue_tCcRow, epi_tile, tiled_copy, thread_idx,
                tCcAcc] = args_tuple;
            auto [m, n, k, l] = tile_coord_mnkl;
            constexpr bool ReferenceSrc = decltype(ref_src)::value;

            // fully OOB CTA in partially OOB cluster
            if (not elem_less(cRow(_0{},_0{}), residue_cRow)) {
                return;
            }

            int lane_m = get<0>(lane_mn);
            [[maybe_unused]] bool is_reduced_lane = lane_m == 0;

            //
            // 1. Warp shuffle reduction
            //
            using FragmentShuffle = Array<ValIdx, sizeof(uint64_t) / sizeof(ValIdx)>;
            static_assert(sizeof(FragmentShuffle) == sizeof(uint64_t));
            Tensor tCrRow_frg = recast<FragmentShuffle>(filter(tCrRow));
            using ReduceShuffle = ArgmaxReduce<FragmentShuffle>;
            ReduceShuffle reduce_shuffle{};

            auto FrgSizePerLaneM = size(tCrRow_frg) / size<0>(lane_layout_MN);
            constexpr bool SwapShuffle = FrgSizePerLaneM > 0;

            if constexpr (SwapShuffle) {
                Tensor tCrRow_frg_ = logical_divide(tCrRow_frg, FrgSizePerLaneM);
                CUTLASS_PRAGMA_UNROLL
                for (int m = size<1>(tCrRow_frg_) / 2; m > 0; m /= 2) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int r = 0; r < m; ++r) {
                        auto frg_A = tCrRow_frg_(_,r);
                        auto frg_B = tCrRow_frg_(_,r + m);
                        CUTLASS_PRAGMA_UNROLL
                        for (int v = 0; v < size(frg_A); ++v) {
                            if (not (lane_m & m)) {
                                cutlass::swap(frg_A(v), frg_B(v));
                            }
                            uint64_t frg_shfl = reinterpret_cast<uint64_t&>(frg_A(v));
                            frg_shfl = __shfl_xor_sync(0xFFFFFFFF, frg_shfl, lane_layout_MN(m, _0{}));
                            frg_A(v) = reduce_shuffle(frg_B(v), reinterpret_cast<FragmentShuffle&>(frg_shfl));
                        }
                    }
                }
            }
            else {
                CUTLASS_PRAGMA_UNROLL
                for (int reduction_rows = size<0>(lane_layout_MN) / 2; reduction_rows > 0; reduction_rows /= 2) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int frg_idx = 0; frg_idx < size(tCrRow_frg); ++frg_idx) {
                        uint64_t frg_shfl = reinterpret_cast<uint64_t&>(tCrRow_frg(frg_idx));
                        frg_shfl = __shfl_down_sync(0xFFFFFFFF, frg_shfl, lane_layout_MN(reduction_rows, _0{}));
                        tCrRow_frg(frg_idx) = reduce_shuffle(
                            tCrRow_frg(frg_idx), reinterpret_cast<FragmentShuffle&>(frg_shfl));
                    }
                }
            }

            //
            // 2. Write per-tile result to gmem workspace
            //
            if constexpr (decltype(size<0>(warp_layout_MN))::value <= 1) {
                // One warp in M: write directly to gmem workspace
                Tensor tCgBuf = sm90_partition_for_epilogue<ReferenceSrc>(
                    gBuf_ml(_,_,m,l), epi_tile, tiled_copy, thread_idx);

                if constexpr (SwapShuffle) {
                    Tensor tCrRow_flt = filter(tCrRow);
                    Tensor tCgBuf_flt = filter(tCgBuf);
                    auto FltFrgSizePerLaneM = size(tCrRow_flt) / size<0>(lane_layout_MN);
                    Tensor tCgBuf_flt_ = logical_divide(tCgBuf_flt, FltFrgSizePerLaneM);
                    Tensor tCrRow_flt_ = logical_divide(tCrRow_flt, FltFrgSizePerLaneM);
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < size(tCrRow_flt_(_,_0{})); ++i) {
                        ValIdx volatile& dst = reinterpret_cast<ValIdx volatile&>(tCgBuf_flt_(i, lane_m));
                        dst = tCrRow_flt_(i, _0{});
                    }
                }
                else {
                    if (is_reduced_lane) {
                        Tensor tCgBuf_flt = filter_zeros(tCgBuf);
                        Tensor tCrRow_flt = filter_zeros(tCrRow);
                        CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < size(tCrRow_flt); ++i) {
                            ValIdx volatile& dst = reinterpret_cast<ValIdx volatile&>(tCgBuf_flt(i));
                            dst = tCrRow_flt(i);
                        }
                    }
                }
                sync_fn();
            }
            //
            // Multiple warps in M: smem reduction then write to gmem
            //
            else {
                Tensor sBuf = make_tensor(
                    make_smem_ptr<ValIdx>(raw_pointer_cast(smem_buffer.data())), sBuf_layout);
                sync_fn();

                Tensor tCsBuf = sm90_partition_for_epilogue<ReferenceSrc>(
                    sBuf(_,_,get<0>(warp_mn)), epi_tile, tiled_copy, thread_idx);

                if constexpr (SwapShuffle) {
                    Tensor tCrRow_flt = filter(tCrRow);
                    Tensor tCsBuf_flt = filter(tCsBuf);
                    auto FltFrgSizePerLaneM = size(tCrRow_flt) / size<0>(lane_layout_MN);
                    Tensor tCsBuf_flt_ = logical_divide(tCsBuf_flt, FltFrgSizePerLaneM);
                    Tensor tCrRow_flt_ = logical_divide(tCrRow_flt, FltFrgSizePerLaneM);
                    copy_aligned(tCrRow_flt_(_,_0{}), tCsBuf_flt_(_,lane_m));
                }
                else {
                    if (is_reduced_lane) {
                        copy_aligned(tCrRow, tCsBuf);
                    }
                }
                sync_fn();

                // Smem tree reduction across warps
                using ReduceSmem = ArgmaxReduce<ValIdx>;
                ReduceSmem reduce_smem{};

                constexpr int RowNum = decltype(size<0>(warp_layout_MN))::value;
                Tensor sBuf_flat = filter_zeros(sBuf);
                constexpr int FragsPerRow = decltype(size<1>(sBuf_flat))::value;

                using VectorGmem = uint64_t volatile;
                Tensor gBuf_flat = filter(gBuf_ml(_,_,m,l));
                Tensor gBuf_vec = recast<VectorGmem>(gBuf_flat);
                CUTLASS_PRAGMA_UNROLL
                for (int frg_idx = thread_idx; frg_idx < FragsPerRow; frg_idx += size(tiled_copy)) {
                    Array<ValIdx, RowNum> frg_smem;
                    CUTLASS_PRAGMA_UNROLL
                    for (int row = 0; row < RowNum; ++row) {
                        frg_smem[row] = sBuf_flat(row * FragsPerRow + frg_idx);
                    }
                    CUTLASS_PRAGMA_UNROLL
                    for (int stride = RowNum / 2; stride > 0; stride /= 2) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int row = 0; row < stride; ++row) {
                            frg_smem[row] = reduce_smem(frg_smem[row], frg_smem[row + stride]);
                        }
                    }
                    gBuf_vec(frg_idx) = reinterpret_cast<uint64_t&>(frg_smem[0]);
                }
                sync_fn();
            }

            //
            // 3. Copy this CTA's result from workspace to output arrays.
            //    Each CTA writes only its own tile_m slot — no coordination needed.
            //
            CUTLASS_PRAGMA_NO_UNROLL
            for (int n_col = thread_idx; n_col < size<1>(gBuf_ml); n_col += size(tiled_copy)) {
                if (n_col >= params.H) continue;  // skip padded columns
                ValIdx vi = gBuf_ml(_0{}, n_col, m, l);
                int out_idx = static_cast<int>(m) * params.H + n_col;
                params.ptr_tile_vals[out_idx] = vi.val;
                params.ptr_tile_idxs[out_idx] = vi.idx;
            }
        }
    };

    template <
        bool ReferenceSrc,
        class... Args
    >
    CUTLASS_DEVICE auto
    get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
        Layout ref_layout_MN = [&] () {
            auto mn_shape = shape(typename decltype(args.tiled_copy)::Tiler_MN{});
            if constexpr (ReferenceSrc) { return right_inverse(args.tiled_copy.get_layoutS_TV()).with_shape(mn_shape); }
            else                        { return right_inverse(args.tiled_copy.get_layoutD_TV()).with_shape(mn_shape); }
        }();

        // Lane layout for shuffle reduction
        using _W = Int<decltype(args.tiled_copy)::TiledNumThr::value / NumThreadsPerWarp>;
        Layout tv2lane = Layout<Shape<Int<NumThreadsPerWarp>,_W,_1>,Stride<_1,_0,_0>>{};
        Layout ref2lane = composition(tv2lane, ref_layout_MN);
        Layout lane_layout_MN = make_layout(filter(get<0>(ref2lane)), filter(get<1>(ref2lane)));
        Layout inv_lane_layout_MN = right_inverse(lane_layout_MN);
        int lane_idx = canonical_lane_idx();
        auto lane_mn = idx2crd(inv_lane_layout_MN(lane_idx), shape(lane_layout_MN));

        // Warp layout for smem reduction
        Layout tv2warp = Layout<Shape<Int<NumThreadsPerWarp>,_W,_1>,Stride<_0,_1,_0>>{};
        Layout ref2warp = composition(tv2warp, ref_layout_MN);
        Layout warp_layout_MN = make_layout(filter(get<0>(ref2warp)), filter(get<1>(ref2warp)));
        Layout inv_warp_layout_MN = right_inverse(warp_layout_MN);
        int warp_idx = args.thread_idx / NumThreadsPerWarp;
        auto warp_mn = idx2crd(inv_warp_layout_MN(warp_idx), shape(warp_layout_MN));

        // Partition output gmem and register tensors
        auto [tile_M, tile_N, tile_K] = args.tile_shape_mnk;
        auto [M, N, K, L] = args.problem_shape_mnkl;
        auto [m, n, k, l] = args.tile_coord_mnkl;

        // Output tensor (ptr_row, used for CuTe partitioning)
        Tensor mRow = make_tensor(
            make_gmem_ptr<ElementOutput>(params.ptr_row),
            make_shape(M, N, L), params.dRow);
        Tensor gRow_l = local_tile(mRow, take<0,2>(args.tile_shape_mnk), make_coord(m,n,_));

        // Register tensor for partial (value, index) pairs
        Tensor tCgRow = sm90_partition_for_epilogue<ReferenceSrc>(
            gRow_l(_,_,l), args.epi_tile, args.tiled_copy, args.thread_idx);
        Tensor tCrRow = make_tensor_like<ValIdx>(tCgRow);

        fill(tCrRow, identity());

        // Accumulator-layout coordinate tensor for M index tracking.
        // visit() receives fragments in accumulator (source) layout, so we need
        // ReferenceSrc=true coordinates to correctly map fragment elements to M indices.
        // args.tCcD may use destination layout (ReferenceSrc=false), which would give
        // wrong M coordinates for argmax tracking.
        auto cAcc = make_identity_tensor(make_shape(tile_M, tile_N));
        auto tCcAcc = sm90_partition_for_epilogue<true>(
            cAcc, args.epi_tile, args.tiled_copy, args.thread_idx);

        // Gmem workspace buffer for per-tile results
        Layout gBuf_layout = make_layout(take<0,2>(args.tile_shape_mnk), make_stride(_0{}, _1{}));
        auto block_shape = ceil_div(make_shape(M,N,L), shape(gBuf_layout));
        Layout block_layout = make_layout(block_shape,
            make_stride(get<1>(block_shape), _1{}, get<0>(block_shape) * get<1>(block_shape)));
        Layout mBuf_layout = blocked_product(gBuf_layout, block_layout);
        Tensor mBuf = make_tensor(make_gmem_ptr(params.reduction_buffer), mBuf_layout);
        Tensor gBuf_ml = local_tile(mBuf, take<0,2>(args.tile_shape_mnk), make_coord(_,n,_));

        // Smem layout for inter-warp reduction
        Layout sBuf_layout = blocked_product(gBuf_layout,
            make_layout(make_shape(_1{},_1{},size<0>(warp_layout_MN))));

        auto args_tuple = make_tuple(
            bool_constant<ReferenceSrc>{}, cute::move(tCrRow), args.tCcD, gRow_l, args.cD, gBuf_ml, sBuf_layout,
            lane_layout_MN, lane_mn, warp_layout_MN, warp_mn,
            args.tile_coord_mnkl, args.residue_cD, args.residue_tCcD, args.epi_tile, args.tiled_copy, args.thread_idx,
            cute::move(tCcAcc));
        return ConsumerStoreCallbacks<decltype(args_tuple)>(cute::move(args_tuple), params);
    }
};

} // namespace cutlass::epilogue::fusion
