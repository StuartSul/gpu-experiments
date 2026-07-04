#include "kittens.cuh"

using namespace kittens;

namespace mxfp8_quantize {

struct config {
    static constexpr int Mb = 128;
    static constexpr int Nb = 128;
    static constexpr int NUM_GROUPS = 2;
    static constexpr int PIPE_DEPTH = 6;

    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 148;
    static constexpr int NUM_THREADS = NUM_GROUPS * Mb;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1024;

    static_assert(PIPE_DEPTH * Mb * Nb * sizeof(bf16)
                + NUM_GROUPS * (Mb * Nb * sizeof(fp8e4m3) + 32 * 16 * sizeof(fp8e8m0))
                + 1024 <= DYNAMIC_SHARED_MEMORY);
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
};

template <bool RETURN_NORMAL, bool RETURN_TRANSPOSED>
__device__ __forceinline__ void mxfp8_quantize_tile(
    const globals::x_bf16_tile &x_bf16_tile,
    const globals::x_fp8_tile &x_fp8_tile,
    const globals::x_sc_tile &x_sc_tile,
    const globals::x_fp8_tile &x_fp8_t_tile,
    const globals::x_sc_tile &x_sc_t_tile,
    const int tid,
    const int barrier_id
) {
    constexpr int TILE_SIZE = 128;
    constexpr int K_BLOCK_SIZE = 32;
    static_assert(RETURN_NORMAL || RETURN_TRANSPOSED, "At least one output pair must be requested");

    // Excess threads have no tile rows to handle. Caller must ensure that this is called by tids 0-127
    if (tid >= TILE_SIZE) return;

    constexpr int ROWS_PER_THREAD = TILE_SIZE / TILE_SIZE;
    constexpr int NUM_K_BLOCKS = TILE_SIZE / K_BLOCK_SIZE; // 4
    constexpr int PACKED_PER_K_BLOCK = K_BLOCK_SIZE / 2;   // 16

    const uint32_t bf16_src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&x_bf16_tile));

    #pragma unroll
    for (int pass = 0; pass < 2; pass++) {
        const bool transposed = pass == 0;
        if ((transposed && !RETURN_TRANSPOSED) || (!transposed && !RETURN_NORMAL)) continue;
        const uint32_t fp8_dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(transposed ? &x_fp8_t_tile : &x_fp8_tile));
        const uint32_t sc_dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(transposed ? &x_sc_t_tile : &x_sc_tile));
        bf16_2 x_bf16_reg[ROWS_PER_THREAD][NUM_K_BLOCKS][PACKED_PER_K_BLOCK];

        // Wait for previous iteration's smem read to finish
        if (transposed)
            group<TILE_SIZE / WARP_THREADS>::sync(barrier_id);

        // Load input matrix from shared memory w/ custom swizzling
        #pragma unroll
        for (int i = 0; i < ROWS_PER_THREAD; i++) {
            int row = transposed ? (tid % 64) * 2 + tid / 64 + i * (TILE_SIZE / 64) : tid + i * TILE_SIZE;
            #pragma unroll
            for (int j = 0; j < NUM_K_BLOCKS; j++) {
                int k_block_idx = (j + tid/8) % NUM_K_BLOCKS;
                #pragma unroll
                for (int k = 0; k < PACKED_PER_K_BLOCK; k++) {
                    int col = k_block_idx*K_BLOCK_SIZE + (tid*4 + k*2) % K_BLOCK_SIZE;
                    if (transposed) {
                        move<bf16>::lds(x_bf16_reg[i][j][k].x, bf16_src_addr + (col*TILE_SIZE + row) * sizeof(bf16));
                        move<bf16>::lds(x_bf16_reg[i][j][k].y, bf16_src_addr + ((col+1)*TILE_SIZE + row) * sizeof(bf16));
                    } else {
                        move<bf16_2>::lds(x_bf16_reg[i][j][k], bf16_src_addr + (row*TILE_SIZE + col) * sizeof(bf16));
                    }
                }
            }
        }
        if (!transposed)
            group<TILE_SIZE / WARP_THREADS>::sync(barrier_id); // in-place writes may begin

        // Perform MXFP8 quantization
        #pragma unroll
        for (int i = 0; i < ROWS_PER_THREAD; i++) {
            int row = transposed ? (tid % 64) * 2 + tid / 64 + i * (TILE_SIZE / 64) : tid + i * TILE_SIZE;
            uint32_t scale_word = 0;
            #pragma unroll
            for (int j = 0; j < NUM_K_BLOCKS; j++) {
                int k_block_idx = (j + tid/8) % NUM_K_BLOCKS;

                bf16_2 amax = __habs2(x_bf16_reg[i][j][0]);
                #pragma unroll
                for (int k = 1; k < PACKED_PER_K_BLOCK; k++)
                    amax = __hmax2(amax, __habs2(x_bf16_reg[i][j][k]));

                // Compute the e8m0 scale, rounding towards positive infinity and saturating to finite (https://arxiv.org/pdf/2506.08027)
                float scale = max(__bfloat162float(__hmax(amax.x, amax.y)) * 0.002232142857f, 0.000000000001f);
                uint16_t sc;
                asm volatile("{cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;}" : "=h"(sc) : "f"(scale), "f"(scale));
                scale_word |= static_cast<uint32_t>(sc & 0xFF) << (k_block_idx * 8);
                const float scale_inv = __uint_as_float((254u - (sc & 0xFF)) << 23); // directly build float32 reciprocal without division

                // Quantize (round-to-nearest-even, saturating to finite)
                #pragma unroll
                for (int k = 0; k < PACKED_PER_K_BLOCK; k += 2) {
                    int col = k_block_idx*K_BLOCK_SIZE + (tid*4 + k*2) % K_BLOCK_SIZE;
                    const float2 v01 = __bfloat1622float2(x_bf16_reg[i][j][k]);
                    const float2 v23 = __bfloat1622float2(x_bf16_reg[i][j][k+1]);
                    uint16_t lo, hi;
                    asm volatile("{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}" : "=h"(lo) : "f"(v01.x * scale_inv), "f"(v01.y * scale_inv));
                    asm volatile("{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}" : "=h"(hi) : "f"(v23.x * scale_inv), "f"(v23.y * scale_inv));
                    const uint32_t packed = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
                    move<int>::sts(fp8_dst_addr + row*TILE_SIZE + col, std::bit_cast<int>(packed));
                }
            }

            // Store the scales to shared memory. Each thread will access 1 bank, so no need to swizzle,
            // but we do have to follow this complicated layout pattern made by NVIDIA:
            // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
            move<int>::sts(sc_dst_addr + (row % 32) * 16 + (row / 32) * 4, std::bit_cast<int>(scale_word));
        }
    }
}

template <bool RETURN_NORMAL, bool RETURN_TRANSPOSED>
__device__ inline void mxfp8_quantize_kernel(const globals &G) {
    // Allocate shared memory
    extern __shared__ int __shm[];
    const uint64_t smem_base_addr = (reinterpret_cast<uint64_t>(&__shm[0]) + 1023) & ~uint64_t(1023);
    auto &x_bf16_tiles = *reinterpret_cast<globals::x_bf16_tile (*)[config::PIPE_DEPTH]>(smem_base_addr);
    auto &x_fp8_t_tiles = *reinterpret_cast<globals::x_fp8_tile (*)[config::NUM_GROUPS]>(smem_base_addr + config::PIPE_DEPTH * sizeof(globals::x_bf16_tile));
    auto &x_sc_t_tiles = *reinterpret_cast<globals::x_sc_tile (*)[config::NUM_GROUPS]>(reinterpret_cast<uint64_t>(&x_fp8_t_tiles) + config::NUM_GROUPS * sizeof(globals::x_fp8_tile));

    // Calculate indices
    const int tid = threadIdx.x;
    const int group_id = tid / config::Mb;
    const int tid_in_group = tid % config::Mb;
    const int row_blocks = static_cast<int>(G.x_bf16.rows()) / config::Mb;
    const int col_blocks = static_cast<int>(G.x_bf16.cols()) / config::Nb;
    const int num_tiles = row_blocks * col_blocks;
    const int num_iters = max(0, (num_tiles - static_cast<int>(blockIdx.x) + static_cast<int>(gridDim.x) - 1) / static_cast<int>(gridDim.x));
    const int num_iters_per_group = max(0, (num_iters - group_id + config::NUM_GROUPS - 1) / config::NUM_GROUPS);
    uint32_t phasebits = 0;

    // Initialize mbarriers
    __shared__ semaphore inputs_arrived[config::PIPE_DEPTH];
    if (tid == 0) {
        #pragma unroll
        for (int stage = 0; stage < config::PIPE_DEPTH; ++stage)
            init_semaphore(inputs_arrived[stage], 0, 1);
    }
    __syncthreads();

    if (tid_in_group == 0) {
        if (num_iters_per_group > 0) {
            const int linear_idx = static_cast<int>(blockIdx.x) + group_id * static_cast<int>(gridDim.x);
            tma::expect(inputs_arrived[group_id], x_bf16_tiles[group_id]);
            tma::load_async(x_bf16_tiles[group_id], G.x_bf16, {linear_idx / col_blocks, linear_idx % col_blocks}, inputs_arrived[group_id]);
        }
        if (num_iters_per_group > 1) {
            const int linear_idx = static_cast<int>(blockIdx.x) + (config::NUM_GROUPS + group_id) * static_cast<int>(gridDim.x);
            tma::expect(inputs_arrived[config::NUM_GROUPS + group_id], x_bf16_tiles[config::NUM_GROUPS + group_id]);
            tma::load_async(x_bf16_tiles[config::NUM_GROUPS + group_id], G.x_bf16, {linear_idx / col_blocks, linear_idx % col_blocks}, inputs_arrived[config::NUM_GROUPS + group_id]);
        }
    }

    for (int group_iter = 0; group_iter < num_iters_per_group; ++group_iter) {
        const int iter = group_iter * config::NUM_GROUPS + group_id;
        const int linear_idx = static_cast<int>(blockIdx.x) + iter * static_cast<int>(gridDim.x);
        const int stage = iter % config::PIPE_DEPTH;
        const int row = linear_idx / col_blocks;
        const int col = linear_idx % col_blocks;

        wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
        update_phasebit<0>(phasebits, stage);

        if (RETURN_TRANSPOSED && tid_in_group == 0)
            tma::store_async_read_wait<RETURN_NORMAL ? 2 : 0>();

        auto &x_fp8_tile = *reinterpret_cast<globals::x_fp8_tile *>(&x_bf16_tiles[stage]);
        auto &x_sc_tile = *reinterpret_cast<globals::x_sc_tile *>(reinterpret_cast<uint64_t>(&x_bf16_tiles[stage]) + sizeof(globals::x_fp8_tile));
        mxfp8_quantize_tile<RETURN_NORMAL, RETURN_TRANSPOSED>(x_bf16_tiles[stage], x_fp8_tile, x_sc_tile, x_fp8_t_tiles[group_id], x_sc_t_tiles[group_id], tid_in_group, 1 + group_id);
        group<config::Mb / WARP_THREADS>::sync(1 + group_id);

        if (tid_in_group == 0) {
            if constexpr (RETURN_TRANSPOSED) {
                tma::store_async(G.x_fp8_t, x_fp8_t_tiles[group_id], {col, row});
                tma::store_async(G.x_sc_t, x_sc_t_tiles[group_id], {col, row, 0, 0});
            }
            if constexpr (RETURN_NORMAL) {
                tma::store_async(G.x_fp8, x_fp8_tile, {row, col});
                tma::store_async(G.x_sc, x_sc_tile, {row, col, 0, 0});
            }
            if (group_iter + 2 < num_iters_per_group) {
                constexpr int TMA_STORES_PER_TILE = (RETURN_NORMAL ? 2 : 0) + (RETURN_TRANSPOSED ? 2 : 0);
                tma::store_async_read_wait<TMA_STORES_PER_TILE>();
                const int prefetch_iter = (group_iter + 2) * config::NUM_GROUPS + group_id;
                const int prefetch_linear_idx = static_cast<int>(blockIdx.x) + prefetch_iter * static_cast<int>(gridDim.x);
                const int prefetch_stage = prefetch_iter % config::PIPE_DEPTH;
                tma::expect(inputs_arrived[prefetch_stage], x_bf16_tiles[prefetch_stage]);
                tma::load_async(x_bf16_tiles[prefetch_stage], G.x_bf16, {prefetch_linear_idx / col_blocks, prefetch_linear_idx % col_blocks}, inputs_arrived[prefetch_stage]);
            }
        }
    }
}

} // namespace mxfp8_quantize

#include "pyutils/torchutils.cuh"

void mxfp8_quantize_entrypoint(
    const at::Tensor &x_bf16,
    const at::Tensor &x_fp8,
    const at::Tensor &x_sc,
    const at::Tensor &x_fp8_t,
    const at::Tensor &x_sc_t,
    const bool return_normal,
    const bool return_transposed
) {
    using C = mxfp8_quantize::config;
    using G = mxfp8_quantize::globals;

    G g {
        .x_bf16 = kittens::py::tensor_to_gl<G::x_bf16_gl>(x_bf16),
        .x_fp8 = kittens::py::tensor_to_gl<G::x_fp8_gl>(x_fp8),
        .x_sc = kittens::py::tensor_to_gl<G::x_sc_gl>(x_sc),
        .x_fp8_t = kittens::py::tensor_to_gl<G::x_fp8_gl>(x_fp8_t),
        .x_sc_t = kittens::py::tensor_to_gl<G::x_sc_gl>(x_sc_t)
    };

    if (return_normal && return_transposed)
        kittens::py::launch_kernel<C, G, mxfp8_quantize::mxfp8_quantize_kernel<true, true>>(g);
    else if (return_normal)
        kittens::py::launch_kernel<C, G, mxfp8_quantize::mxfp8_quantize_kernel<true, false>>(g);
    else
        kittens::py::launch_kernel<C, G, mxfp8_quantize::mxfp8_quantize_kernel<false, true>>(g);
}

PYBIND11_MODULE(_C, m) {
    m.def("mxfp8_quantize", &mxfp8_quantize_entrypoint, "",
          pybind11::arg("x_bf16"),
          pybind11::arg("x_fp8"), pybind11::arg("x_sc"),
          pybind11::arg("x_fp8_t"), pybind11::arg("x_sc_t"),
          pybind11::arg("return_normal"), pybind11::arg("return_transposed"));
}
