#include "kittens.cuh"

using namespace kittens;

namespace mxfp8_quantize {

struct config {
    static constexpr int Mb = 128;
    static constexpr int Nb = 128;

    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = Mb;
};

struct globals {
    using x_bf16_tile = st_bf<config::Mb, config::Nb, false>;
    using x_fp8_tile = st_fp8e4m3<config::Mb, config::Nb, false>;
    using x_sc_tile = st_fp8e8m0<32, 16, false>;

    using x_bf16_gl = gl<bf16, 1, 1, -1, -1, x_bf16_tile>;
    using x_fp8_gl = gl<fp8e4m3, 1, 1, -1, -1, x_fp8_tile>;
    using x_sc_gl = gl<fp8e8m0, -1, -1, 32, 16, x_sc_tile>;

    x_bf16_gl x_bf16;      // (M, N) bf16 input
    x_fp8_gl x_fp8;        // (M, N) MXFP8 output
    x_sc_gl x_sc;          // (M // 128, N // 128, 32, 16) fp8e8m0 block scales
    x_fp8_gl x_fp8_t;      // (N, M) MXFP8 output, transposed
    x_sc_gl x_sc_t;        // (N // 128, M // 128, 32, 16) fp8e8m0 block scales, transposed

    __host__ inline dim3 grid() const {
        return dim3(x_bf16.cols() / config::Nb, x_bf16.rows() / config::Mb);
    }
    __host__ inline int dynamic_shared_memory() const {
        return config::Mb * config::Nb * sizeof(bf16)
             + config::Mb * config::Nb * sizeof(fp8e4m3) + 32 * 16 * sizeof(fp8e8m0)
             + 1024;
    }
};

template <int NUM_THREADS, bool RETURN_NORMAL, bool RETURN_TRANSPOSED>
__device__ __forceinline__ void mxfp8_quantize_tile(
    const globals::x_bf16_tile &x_bf16_smem,
    const globals::x_fp8_tile &x_fp8_smem,
    const globals::x_sc_tile &x_sc_smem,
    const globals::x_fp8_tile &xt_fp8_smem,
    const globals::x_sc_tile &xt_sc_smem,
    const int tid,
    const int barrier_id
) {
    constexpr int TILE_SIZE = 128;
    constexpr int K_BLOCK_SIZE = 32;
    static_assert(RETURN_NORMAL || RETURN_TRANSPOSED, "At least one output pair must be requested");
    static_assert(TILE_SIZE % NUM_THREADS == 0 && NUM_THREADS <= TILE_SIZE);

    constexpr int ROWS_PER_THREAD = TILE_SIZE / NUM_THREADS;
    constexpr int NUM_K_BLOCKS = TILE_SIZE / K_BLOCK_SIZE;
    constexpr int PACKED_PER_K_BLOCK = K_BLOCK_SIZE / 2;

    const uint32_t bf16_src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&x_bf16_smem));

    #pragma unroll
    for (int pass = 0; pass < 2; pass++) {
        const bool transposed = pass == 0;
        if ((transposed && !RETURN_TRANSPOSED) || (!transposed && !RETURN_NORMAL)) continue;

        const uint32_t fp8_dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(transposed ? &xt_fp8_smem : &x_fp8_smem));
        const uint32_t sc_dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(transposed ? &xt_sc_smem : &x_sc_smem));
        bf16_2 x_bf16_reg[ROWS_PER_THREAD][NUM_K_BLOCKS][PACKED_PER_K_BLOCK];

        // The transposed pass reads other threads' rows, so it must sync BEFORE staging: this
        // orders the caller's row zeroing and output-reuse guard ahead of this pass. The normal
        // pass reads only the calling thread's own rows and instead syncs between staging and
        // its in-place writes (below).
        if (transposed)
            group<NUM_THREADS / WARP_THREADS>::sync(barrier_id);

        // Load input matrix from shared memory (custom swizzling): packed rows, or columns as
        // vertical bf16 pairs when transposed. A thread's output row is a tile row, or a tile
        // column when transposed (remapped so the 32 lanes of a warp cover 32 distinct banks,
        // making the scalar column loads conflict-free).
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; r++) {
            int own_row = transposed ? (tid % 64) * 2 + tid / 64 + r * (NUM_THREADS / 64) : tid + r * NUM_THREADS;
            #pragma unroll
            for (int i = 0; i < NUM_K_BLOCKS; i++) {
                int k_block_idx = (i + tid/8) % NUM_K_BLOCKS; // 8 SMEM banks per K-block
                #pragma unroll
                for (int j = 0; j < PACKED_PER_K_BLOCK; j++) {
                    int elem = k_block_idx*K_BLOCK_SIZE + (tid*4 + j*2) % K_BLOCK_SIZE;
                    if (transposed) {
                        move<bf16>::lds(x_bf16_reg[r][i][j].x, bf16_src_addr + (elem*TILE_SIZE + own_row) * sizeof(bf16));
                        move<bf16>::lds(x_bf16_reg[r][i][j].y, bf16_src_addr + ((elem+1)*TILE_SIZE + own_row) * sizeof(bf16));
                    } else {
                        move<bf16_2>::lds(x_bf16_reg[r][i][j], bf16_src_addr + (own_row*TILE_SIZE + elem) * sizeof(bf16));
                    }
                }
            }
        }
        if (!transposed)
            group<NUM_THREADS / WARP_THREADS>::sync(barrier_id); // all staging done: in-place writes may begin

        // Perform MXFP8 quantization
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; r++) {
            int own_row = transposed ? (tid % 64) * 2 + tid / 64 + r * (NUM_THREADS / 64) : tid + r * NUM_THREADS;
            uint32_t sc_word = 0;
            #pragma unroll
            for (int i = 0; i < NUM_K_BLOCKS; i++) {
                int k_block_idx = (i + tid/8) % NUM_K_BLOCKS; // 8 SMEM banks per K-block

                // Calculate absolute maximum
                bf16_2 amax = __habs2(x_bf16_reg[r][i][0]);
                #pragma unroll
                for (int j = 1; j < PACKED_PER_K_BLOCK; j++)
                    amax = __hmax2(amax, __habs2(x_bf16_reg[r][i][j]));

                // Compute the e8m0 scale, rounding towards positive infinity and saturating to finite
                // (https://arxiv.org/pdf/2506.08027), and its exact reciprocal 2^(127 - exponent).
                // cvt.rp keeps everything in registers; __nv_cvt_float_to_e8m0 would spill to the stack.
                float scale = max(__bfloat162float(__hmax(amax.x, amax.y)) * 0.002232142857f, 0.000000000001f); // in theory lower clamp is not needed
                uint16_t sc;
                asm volatile("{cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;}" : "=h"(sc) : "f"(scale), "f"(scale));
                sc_word |= static_cast<uint32_t>(sc & 0xFF) << (k_block_idx * 8);
                const float scale_inv = __uint_as_float((254u - (sc & 0xFF)) << 23);

                // Quantize (round-to-nearest-even, saturating to finite) and store 4 packed values
                // per store. cvt packs operand b into the low half, so the value order is reversed.
                #pragma unroll
                for (int j = 0; j < PACKED_PER_K_BLOCK; j += 2) {
                    int elem = k_block_idx*K_BLOCK_SIZE + (tid*4 + j*2) % K_BLOCK_SIZE;
                    const float2 v01 = __bfloat1622float2(x_bf16_reg[r][i][j]);
                    const float2 v23 = __bfloat1622float2(x_bf16_reg[r][i][j+1]);
                    uint16_t lo, hi;
                    asm volatile("{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}" : "=h"(lo) : "f"(v01.x * scale_inv), "f"(v01.y * scale_inv));
                    asm volatile("{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}" : "=h"(hi) : "f"(v23.x * scale_inv), "f"(v23.y * scale_inv));
                    const uint32_t packed = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16); // bytes elem..elem+3 of the output row
                    move<int>::sts(fp8_dst_addr + own_row*TILE_SIZE + elem, std::bit_cast<int>(packed));
                }
            }

            // Store the scales to shared memory. Each thread will access 1 bank, so no need to swizzle,
            // but we do have to follow this complicated layout pattern made by NVIDIA:
            // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
            move<int>::sts(sc_dst_addr + (own_row % 32) * 16 + (own_row / 32) * 4, std::bit_cast<int>(sc_word));
        }
    }
}

template <bool RETURN_NORMAL = true, bool RETURN_TRANSPOSED = false>
__device__ inline void kernel(const globals &G) {
    // Allocate shared memory: the input tile plus one transposed-output scratch after it (the
    // transposed pair cannot be quantized in place: it needs the bf16 tile intact and has nowhere
    // to alias)
    extern __shared__ int __shm[];
    const uint64_t smem_base_addr = (reinterpret_cast<uint64_t>(&__shm[0]) + 1023) & ~uint64_t(1023);
    auto &x_bf16_smem = *reinterpret_cast<globals::x_bf16_tile *>(smem_base_addr);
    struct alignas(128) quant_scratch {
        globals::x_fp8_tile fp8;
        globals::x_sc_tile sc;
    };
    auto &t_scratch = *reinterpret_cast<quant_scratch *>(smem_base_addr + sizeof(globals::x_bf16_tile));

    // Calculate indices
    const int tid = threadIdx.x;
    const int row = blockIdx.y;
    const int col = blockIdx.x;

    // Initialize mbarrier and initiate TMA load
    __shared__ semaphore inputs_arrived;
    if (tid == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        tma::expect(inputs_arrived, x_bf16_smem);
        tma::load_async(x_bf16_smem, G.x_bf16, {row, col}, inputs_arrived);
    }

    // Wait for the TMA load to complete
    __syncthreads();
    wait(inputs_arrived, 0);

    // The full 128x128 tile is now in shared memory. Run the ThunderKittens MXFP8 tile quantizer
    // (transposed pass into scratch first, normal pass in place) and store the requested pairs
    globals::x_fp8_tile &fp8_tile = *reinterpret_cast<globals::x_fp8_tile *>(&x_bf16_smem);
    globals::x_sc_tile &sc_tile = *reinterpret_cast<globals::x_sc_tile *>(
        reinterpret_cast<uint64_t>(&x_bf16_smem) + sizeof(globals::x_fp8_tile));
    mxfp8_quantize_tile<config::Mb, RETURN_NORMAL, RETURN_TRANSPOSED>(x_bf16_smem, fp8_tile, sc_tile, t_scratch.fp8, t_scratch.sc, tid, 1);
    group<config::Mb / WARP_THREADS>::sync(1);

    // Store to global memory
    if (tid == 0) {
        if constexpr (RETURN_TRANSPOSED) {
            tma::store_async(G.x_fp8_t, t_scratch.fp8, {col, row});
            tma::store_async(G.x_sc_t, t_scratch.sc, {col, row, 0, 0});
        }
        if constexpr (RETURN_NORMAL) {
            tma::store_async(G.x_fp8, fp8_tile, {row, col});
            tma::store_async(G.x_sc, sc_tile, {row, col, 0, 0});
        }
    }
}

} // namespace mxfp8_quantize

#include "pyutils/torchutils.cuh"

void mxfp8_quantize_entrypoint(
    const at::Tensor &x_bf16,
    at::Tensor &x_fp8,
    at::Tensor &x_sc,
    at::Tensor &x_fp8_t,
    at::Tensor &x_sc_t,
    const bool return_normal,
    const bool return_transposed
) {
    using C = mxfp8_quantize::config;
    using G = mxfp8_quantize::globals;

    TORCH_CHECK(return_normal || return_transposed, "At least one output pair must be requested");

    G g {
        .x_bf16 = kittens::py::tensor_to_gl<G::x_bf16_gl>(x_bf16),
        .x_fp8 = kittens::py::tensor_to_gl<G::x_fp8_gl>(x_fp8),
        .x_sc = kittens::py::tensor_to_gl<G::x_sc_gl>(x_sc),
        .x_fp8_t = kittens::py::tensor_to_gl<G::x_fp8_gl>(x_fp8_t),
        .x_sc_t = kittens::py::tensor_to_gl<G::x_sc_gl>(x_sc_t)
    };
    if (return_normal && return_transposed)
        kittens::py::launch_kernel<C, G, mxfp8_quantize::kernel<true, true>>(g);
    else if (return_normal)
        kittens::py::launch_kernel<C, G, mxfp8_quantize::kernel<true, false>>(g);
    else
        kittens::py::launch_kernel<C, G, mxfp8_quantize::kernel<false, true>>(g);
}

PYBIND11_MODULE(_C, m) {
    m.def("mxfp8_quantize", &mxfp8_quantize_entrypoint, "MXFP8-quantize a bf16 matrix into normal and/or transposed fp8e4m3 payloads with swizzled e8m0 block scales",
          pybind11::arg("x_bf16"), pybind11::arg("x_fp8"), pybind11::arg("x_sc"),
          pybind11::arg("x_fp8_t"), pybind11::arg("x_sc_t"),
          pybind11::arg("return_normal") = true, pybind11::arg("return_transposed") = false);
}
