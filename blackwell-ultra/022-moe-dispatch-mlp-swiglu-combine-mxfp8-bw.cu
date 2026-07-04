#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros.h>

using namespace kittens;

struct scheduler {

struct config {
    static constexpr int EXPERT_PADDING = 256; // row alignment for contiguous grouped GEMM
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 1024;
    static constexpr int NUM_WARPS = NUM_THREADS / WARP_THREADS;
};

struct globals {
    using topk_gl = gl<int, 1, -1, -1, -1>;
    using index_gl = gl<int, 1, 1, 1, -1>;

    topk_gl topk;                        // (world_size, num_local_tokens, topk)
    index_gl schedule_peer_rank;         // (schedule_capacity,) must be initialized to -1
    index_gl schedule_peer_token_idx;    // (schedule_capacity,) original_token_idx * topk + k
    index_gl num_tokens;                 // (1,) total padded token count, must be zero-initialized
    index_gl tokens_per_expert;          // (num_local_experts,) padded per-expert token counts
    index_gl tokens_per_expert_and_peer; // (num_local_experts * world_size,) per-(local_expert, peer_rank) token counts, must be zero-initialized

    int rank;                            // this (destination) rank
};

// Stage 1: Count the number of tokens routed from each peer rank to each local expert
static __device__ __forceinline__ void count_kernel(const globals &G) {
    const int world_size = G.topk.depth();
    const int num_local_tokens = G.topk.rows();
    const int topk = G.topk.cols();
    const int rank_stride = num_local_tokens * topk;
    const int num_global_tokens = world_size * rank_stride;
    const int num_local_experts = G.tokens_per_expert.cols();
    const int first_expert = G.rank * num_local_experts;
    const int last_expert = first_expert + num_local_experts;

    extern __shared__ int tokens_per_expert_and_peer[]; // (num_local_experts, world_size)
    for (int i = threadIdx.x; i < G.tokens_per_expert_and_peer.cols(); i += blockDim.x)
        tokens_per_expert_and_peer[i] = 0;
    __syncthreads();

    const int grid_stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_global_tokens; idx += grid_stride) {
        const int peer_rank = idx / rank_stride;
        const int peer_token_idx = idx - peer_rank * rank_stride;
        const int expert_idx = G.topk[{peer_rank, peer_token_idx / topk, peer_token_idx % topk}];
        if (expert_idx >= first_expert && expert_idx < last_expert)
            atomicAdd(&tokens_per_expert_and_peer[(expert_idx - first_expert) * world_size + peer_rank], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < G.tokens_per_expert_and_peer.cols(); i += blockDim.x)
        if (tokens_per_expert_and_peer[i] != 0)
            atomicAdd(&G.tokens_per_expert_and_peer[{i}], tokens_per_expert_and_peer[i]);
}

// Stage 2: Pad each expert's total token count by EXPERT_PADDING and accumulate the total count
static __device__ __forceinline__ void pad_kernel(const globals &G) {
    const int local_expert = blockIdx.x;
    const int world_size = G.topk.depth();
    int num_tokens = 0;
    for (int peer_rank = 0; peer_rank < world_size; ++peer_rank)
        num_tokens += G.tokens_per_expert_and_peer[{local_expert * world_size + peer_rank}];
    const int padded_num_tokens = (num_tokens + config::EXPERT_PADDING - 1) / config::EXPERT_PADDING * config::EXPERT_PADDING;
    G.tokens_per_expert[{local_expert}] = padded_num_tokens;
    atomicAdd(&G.num_tokens[{0}], padded_num_tokens);
}

// Stage 3: Schedule each token into its expert's 256-padded segment
static __device__ __forceinline__ void schedule_kernel(const globals &G) {
    const int world_size = G.topk.depth();
    const int num_local_tokens = G.topk.rows();
    const int topk = G.topk.cols();
    const int rank_stride = num_local_tokens * topk;
    const int num_local_experts = G.tokens_per_expert.cols();
    const int first_expert = G.rank * num_local_experts;

    // The schedule capacity is for a realistic worst case, but not the absolute one
    if (G.num_tokens[{0}] > G.schedule_peer_rank.cols()) asm volatile("{trap;}");

    extern __shared__ int tokens_per_peer_rank[]; // (world_size,) this expert's per-peer-rank counts
    __shared__ int cumulative_tokens_from_peer_rank[config::NUM_WARPS];

    for (int idx = blockIdx.x; idx < num_local_experts * world_size; idx += gridDim.x) {
        const int local_expert = idx / world_size;
        const int peer_rank = idx % world_size;

        // Base row of this expert's padded segment
        int expert_base = 0;
        for (int expert_idx = 0; expert_idx < local_expert; ++expert_idx) 
            expert_base += G.tokens_per_expert[{expert_idx}];

        for (int rank = threadIdx.x; rank < world_size; rank += blockDim.x)
            tokens_per_peer_rank[rank] = G.tokens_per_expert_and_peer[{local_expert * world_size + rank}];
        __syncthreads();

        // Step 1. Count the number of tokens routed from this peer rank to this expert
        int _tokens_from_peer_rank = 0;
        for (int peer_token_idx = threadIdx.x; peer_token_idx < rank_stride; peer_token_idx += blockDim.x) {
            const int expert_idx = G.topk[{peer_rank, peer_token_idx / topk, peer_token_idx % topk}];
            _tokens_from_peer_rank += (expert_idx - first_expert == local_expert) ? 1 : 0;
        }
        // Step 2. Cumulative sum within a warp: thread i's `inclusive` will have the sum from thread 0 to thread i
        int inclusive = _tokens_from_peer_rank;
        for (int offset = 1; offset < WARP_THREADS; offset *= 2) {
            const int n = __shfl_up_sync(0xffffffff, inclusive, offset);
            if (warp::laneid() >= offset) inclusive += n;
        }
        if (warp::laneid() == WARP_THREADS - 1) cumulative_tokens_from_peer_rank[warpid()] = inclusive;
        __syncthreads();
        // Step 3: Cumulative sum across warps
        if (warpid() == 0) {
            int warp_total = (warp::laneid() < config::NUM_WARPS) ? cumulative_tokens_from_peer_rank[warp::laneid()] : 0;
            for (int offset = 1; offset < WARP_THREADS; offset *= 2) {
                const int n = __shfl_up_sync(0xffffffff, warp_total, offset);
                if (warp::laneid() >= offset) warp_total += n;
            }
            if (warp::laneid() < config::NUM_WARPS) cumulative_tokens_from_peer_rank[warp::laneid()] = warp_total;
        }
        __syncthreads();
        int j = (warpid() == 0 ? 0 : cumulative_tokens_from_peer_rank[warpid() - 1]) + inclusive - _tokens_from_peer_rank;

        for (int peer_token_idx = threadIdx.x; peer_token_idx < rank_stride; peer_token_idx += blockDim.x) {
            const int orig_token_idx = peer_token_idx / topk;
            const int expert_idx = G.topk[{peer_rank, orig_token_idx, peer_token_idx % topk}];
            if (expert_idx - first_expert == local_expert) {
                int dst_token_idx = expert_base;
                for (int rank = 0; rank < world_size; ++rank) {
                    const int num_tokens = tokens_per_peer_rank[rank];
                    dst_token_idx += min(num_tokens, j);
                    dst_token_idx += (rank < peer_rank && num_tokens > j) ? 1 : 0;
                }
                G.schedule_peer_rank[{dst_token_idx}] = peer_rank;
                G.schedule_peer_token_idx[{dst_token_idx}] = peer_token_idx; // original_token_idx * topk + k
                ++j;
            }
        }
        __syncthreads(); // before the next iteration reuses cumulative_tokens_from_peer_rank
    }
}

static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> schedule(
    const at::Tensor &topk_all,
    const int num_local_experts,
    const int schedule_capacity,
    const int rank
) {
    const int world_size = static_cast<int>(topk_all.size(0));

    at::Tensor schedule_peer_rank = at::empty({schedule_capacity}, topk_all.options().dtype(at::kInt));
    at::Tensor schedule_peer_token_idx = at::empty({schedule_capacity}, topk_all.options().dtype(at::kInt));
    at::Tensor num_tokens = at::zeros({1}, topk_all.options().dtype(at::kInt));
    at::Tensor tokens_per_expert = at::empty({num_local_experts}, topk_all.options().dtype(at::kInt));
    at::Tensor tokens_per_expert_and_peer = at::zeros({num_local_experts * world_size}, topk_all.options().dtype(at::kInt));
    schedule_peer_rank.fill_(-1);

    globals G {
        .topk = kittens::py::tensor_to_gl<globals::topk_gl>(topk_all),
        .schedule_peer_rank = kittens::py::tensor_to_gl<globals::index_gl>(schedule_peer_rank),
        .schedule_peer_token_idx = kittens::py::tensor_to_gl<globals::index_gl>(schedule_peer_token_idx),
        .num_tokens = kittens::py::tensor_to_gl<globals::index_gl>(num_tokens),
        .tokens_per_expert = kittens::py::tensor_to_gl<globals::index_gl>(tokens_per_expert),
        .tokens_per_expert_and_peer =kittens::py::tensor_to_gl<globals::index_gl>(tokens_per_expert_and_peer),
        .rank = rank,
    };

    auto stream = at::cuda::getCurrentCUDAStream();
    kittens::py::global_kernel<config, globals, scheduler::count_kernel>
        <<<(G.topk.numel() + config::NUM_THREADS - 1) / config::NUM_THREADS, config::NUM_THREADS, num_local_experts * world_size * sizeof(int), stream>>>(G);
    kittens::py::global_kernel<config, globals, scheduler::pad_kernel>
        <<<num_local_experts, 1, 0, stream>>>(G);
    kittens::py::global_kernel<config, globals, scheduler::schedule_kernel>
        <<<num_local_experts * world_size, config::NUM_THREADS, world_size * sizeof(int), stream>>>(G);

    return {schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert};
}

}; // struct scheduler

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

    // Excess threads have no tile rows to handle. Caller must ensure that this is called by threads 0-127
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
    auto &x_bf16_tile = *reinterpret_cast<globals::x_bf16_tile *>(smem_base_addr);
    auto &x_fp8_tile = *reinterpret_cast<globals::x_fp8_tile *>(&x_bf16_tile);
    auto &x_sc_tile = *reinterpret_cast<globals::x_sc_tile *>(reinterpret_cast<uint64_t>(&x_bf16_tile) + sizeof(globals::x_fp8_tile));
    auto &x_fp8_t_tile = *reinterpret_cast<globals::x_fp8_tile *>(reinterpret_cast<uint64_t>(&x_bf16_tile) + sizeof(globals::x_bf16_tile));
    auto &x_sc_t_tile = *reinterpret_cast<globals::x_sc_tile *>(reinterpret_cast<uint64_t>(&x_fp8_t_tile) + sizeof(globals::x_fp8_tile));

    // Calculate indices
    const int tid = threadIdx.x;
    const int row = blockIdx.y;
    const int col = blockIdx.x;

    // Initialize mbarrier and initiate TMA load
    __shared__ semaphore inputs_arrived;
    if (tid == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        tma::expect(inputs_arrived, x_bf16_tile);
        tma::load_async(x_bf16_tile, G.x_bf16, {row, col}, inputs_arrived);
    }

    // Wait for the TMA load to complete
    __syncthreads();
    wait(inputs_arrived, 0);

    // Quantize
    mxfp8_quantize_tile<RETURN_NORMAL, RETURN_TRANSPOSED>(x_bf16_tile, x_fp8_tile, x_sc_tile, x_fp8_t_tile, x_sc_t_tile, tid, 1);
    __syncthreads();

    // Store to global memory
    if (tid == 0) {
        if constexpr (RETURN_TRANSPOSED) {
            tma::store_async(G.x_fp8_t, x_fp8_t_tile, {col, row});
            tma::store_async(G.x_sc_t, x_sc_t_tile, {col, row, 0, 0});
        }
        if constexpr (RETURN_NORMAL) {
            tma::store_async(G.x_fp8, x_fp8_tile, {row, col});
            tma::store_async(G.x_sc, x_sc_tile, {row, col, 0, 0});
        }
    }
}

} // namespace mxfp8_quantize

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

template <int NUM_DEVICES>
struct dispatch_mlp_swiglu_combiner {

struct config {
    // Grouped GEMM
    static constexpr int MLP_Mb = 256;
    static constexpr int MLP_Nb = 256;
    static constexpr int MLP_Kb = 128;
    static constexpr int SUPERGROUP_SIZE = 8;
    static constexpr int LOAD_PIPE_DEPTH = 6;
    static constexpr int EPI_PIPE_DEPTH = 16;
    static constexpr int NUM_D_TILES = 4;

    // MXFP8 quantize
    static constexpr int QUANT_Mb = 128;
    static constexpr int QUANT_Nb = 128;
    static constexpr int QUANT_NUM_GROUPS = 2;
    static constexpr int QUANT_PIPE_DEPTH = 6;
    static constexpr int QUANT_TILES_PER_CTA = 12;

    // Fused SwiGLU + MXFP8 quantize
    static constexpr int SWIGLU_Mb = 128;
    static constexpr int SWIGLU_Nb = 128;
    static constexpr int SWIGLU_PIPE_DEPTH = 3;

    // Dispatch/Combine
    static constexpr int DISPATCH_COMBINE_Mb = 64;
    static constexpr int DISPATCH_COMBINE_Nb = 256;
    static constexpr int DISPATCH_COMBINE_PIPE_DEPTH = 7;
    
    // Kernel launch
    static constexpr int CLC_PIPE_DEPTH = 1;
    static constexpr int CLC_DRAIN_PIPE_DEPTH = 8; // roughly a good number, but variance is low
    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int NUM_CONSUMERS = 1;
    static constexpr int NUM_PRODUCERS = 1;
    static constexpr int NUM_WARPS = (NUM_CONSUMERS + NUM_PRODUCERS) * WARPGROUP_WARPS; // 8
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS; // 256
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1024;
};

struct globals {
    // Grouped GEMM
    using fp8_tile = st_fp8e4m3<config::MLP_Mb / 2, config::MLP_Kb>;
    using sc_tile = st_fp8e8m0<32, 16, false>;
    using d_tile = st_bf<config::MLP_Mb / 2, config::MLP_Nb / config::EPI_PIPE_DEPTH>;

    // MXFP8 quantize
    using q_bf16_tile = mxfp8_quantize::globals::x_bf16_tile; // st_bf<128, 128, false>
    using q_fp8_tile = mxfp8_quantize::globals::x_fp8_tile;   // st_fp8e4m3<128, 128, false>

    // Fused SwiGLU
    using swiglu_tile = st_bf<config::SWIGLU_Mb, config::SWIGLU_Nb>;

    // Dispatch/Combine
    using token_vec = sv_bf<config::DISPATCH_COMBINE_Nb>;

    // Global layouts
    using activation_gl = gl<bf16, 1, 1, -1, -1, token_vec, q_bf16_tile, d_tile, swiglu_tile>;
    using activation_pgl = pgl<activation_gl, NUM_DEVICES, false>;
    using activation_fp8_gl = gl<fp8e4m3, 1, 1, -1, -1, fp8_tile, q_fp8_tile>;
    using weight_fp8_gl = gl<fp8e4m3, 1, -1, -1, -1, fp8_tile>;
    using sc_gl = gl<fp8e8m0, -1, -1, 32, 16, sc_tile>;
    using index_gl = gl<int, 1, 1, 1, -1>;

    activation_gl x_shared;              // (num_local_tokens, H)
    activation_gl x_routed;              // (macrobatch_size, H)
    activation_fp8_gl x_fp8_shared;      // (num_local_tokens, H)
    activation_fp8_gl x_fp8_routed;      // (macrobatch_size, H)
    sc_gl x_sc_shared;                   // (num_local_tokens / 128, H / 128, 32, 16)
    sc_gl x_sc_routed;                   // (macrobatch_size / 128, H / 128, 32, 16)
    activation_gl gate_shared;           // (num_local_tokens, I)
    activation_gl gate_routed;           // (macrobatch_size, I)
    activation_gl up_shared;             // (num_local_tokens, I)
    activation_gl up_routed;             // (macrobatch_size, I)
    activation_fp8_gl hidden_fp8_shared; // (num_local_tokens, I)
    activation_fp8_gl hidden_fp8_routed; // (macrobatch_size, I)
    sc_gl hidden_sc_shared;              // (num_local_tokens / 128, I / 128, 32, 16)
    sc_gl hidden_sc_routed;              // (macrobatch_size / 128, I / 128, 32, 16)
    activation_gl y_shared;              // (num_local_tokens, H)
    activation_gl y_routed;              // (macrobatch_size, H)

    activation_pgl x_routed_send_buffer; // (num_local_tokens, H)
    activation_pgl y_routed_recv_buffer; // (num_local_tokens * topk, H)

    weight_fp8_gl w_shared_gate;         // (I, H)
    weight_fp8_gl w_routed_gate;         // (num_local_experts, I, H)
    weight_fp8_gl w_shared_up;           // (I, H)
    weight_fp8_gl w_routed_up;           // (num_local_experts, I, H)
    weight_fp8_gl w_shared_down;         // (H, I)
    weight_fp8_gl w_routed_down;         // (num_local_experts, H, I)
    sc_gl w_shared_gate_sc;              // (I / 128, H / 128, 32, 16)
    sc_gl w_routed_gate_sc;              // (num_local_experts * I / 128, H / 128, 32, 16)
    sc_gl w_shared_up_sc;                // (I / 128, H / 128, 32, 16)
    sc_gl w_routed_up_sc;                // (num_local_experts * I / 128, H / 128, 32, 16)
    sc_gl w_shared_down_sc;              // (H / 128, I / 128, 32, 16)
    sc_gl w_routed_down_sc;              // (num_local_experts * H / 128, I / 128, 32, 16)

    index_gl schedule_peer_rank;         // (schedule_capacity,)
    index_gl schedule_peer_token_idx;    // (schedule_capacity,)
    index_gl num_tokens;                 // (1,)
    index_gl tokens_per_expert;          // (num_local_experts,)

    index_gl mlp_swiglu_counter;         // (shared_gate_up_tasks + routed_gate_up_tasks + shared_row_blocks + routed_row_blocks,)
    index_gl quantize_counter;           // (1 + num_minibatches,), [0] is shared, [1 + g] is global minibatch g
    index_gl dispatch_row_counter;       // (schedule_capacity / DISPATCH_COMBINE_Mb,)
    index_gl dispatch_tile_counter;      // (schedule_capacity / DISPATCH_COMBINE_Mb * (H / DISPATCH_COMBINE_Nb),)
    index_gl combine_counter;            // (num_minibatches,)

    const int topk;
    const int num_comm_sms;
    const int macrobatch_size;
    const int minibatch_size;
    const int quantize_tile_granularity; // 1: wait per dispatched (64, 256) tile, 0: wait per fully-dispatched 64-row group

    __host__ inline dim3 grid() const {
        const int num_minibatches = (schedule_peer_rank.cols() + minibatch_size - 1) / minibatch_size; // across all macrobatches
        const int shared_row_blocks = x_shared.rows() / config::MLP_Mb;
        const int minibatch_routed_row_blocks = minibatch_size / config::MLP_Mb;
        const int shared_quantize_tiles = (x_shared.rows() / config::QUANT_Mb) * (x_shared.cols() / config::QUANT_Nb);
        const int minibatch_routed_quantize_tiles = (minibatch_size / config::QUANT_Mb) * (x_routed.cols() / config::QUANT_Nb);
        const int shared_quantize_tasks = (shared_quantize_tiles + config::CLUSTER_SIZE * config::QUANT_TILES_PER_CTA - 1) / (config::CLUSTER_SIZE * config::QUANT_TILES_PER_CTA);
        const int minibatch_routed_quantize_tasks = (minibatch_routed_quantize_tiles + config::CLUSTER_SIZE * config::QUANT_TILES_PER_CTA - 1) / (config::CLUSTER_SIZE * config::QUANT_TILES_PER_CTA);
        const int shared_gate_up_tasks = shared_row_blocks * (w_shared_gate.rows() / config::MLP_Nb);
        const int minibatch_routed_gate_up_tasks = minibatch_routed_row_blocks * (w_routed_gate.rows() / config::MLP_Nb);
        const int shared_swiglu_tiles = (hidden_fp8_shared.rows() / config::SWIGLU_Mb) * (hidden_fp8_shared.cols() / config::SWIGLU_Nb);
        const int minibatch_routed_swiglu_tiles = (minibatch_size / config::SWIGLU_Mb) * (hidden_fp8_routed.cols() / config::SWIGLU_Nb);
        const int shared_swiglu_tasks = (shared_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_PIPE_DEPTH);
        const int minibatch_routed_swiglu_tasks = (minibatch_routed_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_PIPE_DEPTH);
        const int shared_down_tasks = shared_row_blocks * (w_shared_down.rows() / config::MLP_Nb);
        const int minibatch_routed_down_tasks = minibatch_routed_row_blocks * (w_routed_down.rows() / config::MLP_Nb);
        const int shared_tasks = shared_quantize_tasks + 2 * shared_gate_up_tasks + shared_swiglu_tasks + shared_down_tasks;
        const int minibatch_tasks = minibatch_routed_quantize_tasks + 2 * minibatch_routed_gate_up_tasks + minibatch_routed_swiglu_tasks + minibatch_routed_down_tasks;
        return dim3(config::CLUSTER_SIZE * (shared_tasks + num_minibatches * minibatch_tasks) + num_comm_sms);
    }
};

template <bool IS_DISPATCH>
static __device__ __forceinline__ void dispatch_combine_kernel(
    const globals &G,
    semaphore (&inputs_arrived)[config::DISPATCH_COMBINE_PIPE_DEPTH],
    uint32_t &bitfield,
    int macrobatch_idx,
    int task_idx,
    uint64_t smem_base_addr
) {
    auto &token_vecs = *reinterpret_cast<typename globals::token_vec (*)[config::DISPATCH_COMBINE_PIPE_DEPTH][config::DISPATCH_COMBINE_Mb]>(smem_base_addr);

    const int tid = threadIdx.x;
    const bool is_worker = tid < config::DISPATCH_COMBINE_Mb; // only these threads move tokens, but all threads join the barriers and waits

    const int col_blocks = G.x_routed.cols() / config::DISPATCH_COMBINE_Nb;
    const int first_tile_idx = task_idx * config::DISPATCH_COMBINE_PIPE_DEPTH;

    const int num_tokens = G.num_tokens[{0}];
    const int macrobatch_offset = macrobatch_idx * G.macrobatch_size;
    const int num_macrobatch_tokens = min(G.macrobatch_size, num_tokens - macrobatch_offset);
    const int num_valid_tiles = min(config::DISPATCH_COMBINE_PIPE_DEPTH, num_macrobatch_tokens / config::DISPATCH_COMBINE_Mb * col_blocks - first_tile_idx); // because we pad to 256
    if (num_valid_tiles <= 0) return;

    const int first_row_idx = first_tile_idx / col_blocks * config::DISPATCH_COMBINE_Mb + tid;
    const int first_col_block_idx = first_tile_idx % col_blocks;

    int row_idx[config::DISPATCH_COMBINE_PIPE_DEPTH], col_block_idx[config::DISPATCH_COMBINE_PIPE_DEPTH], peer_rank[config::DISPATCH_COMBINE_PIPE_DEPTH], 
        peer_token_idx[config::DISPATCH_COMBINE_PIPE_DEPTH], num_valid[config::DISPATCH_COMBINE_PIPE_DEPTH];
    #pragma unroll
    for (int stage = 0, row = first_row_idx, col = first_col_block_idx; stage < config::DISPATCH_COMBINE_PIPE_DEPTH; ++stage) {
        const bool is_valid_tile = stage < num_valid_tiles;
        row_idx[stage] = row;
        col_block_idx[stage] = col;
        peer_rank[stage] = is_valid_tile && is_worker ? G.schedule_peer_rank[{macrobatch_offset + row}] : -1;
        peer_token_idx[stage] = is_valid_tile && is_worker ? G.schedule_peer_token_idx[{macrobatch_offset + row}] : -1;
        num_valid[stage] = !is_valid_tile ? 0
                         : (stage == 0 || col == 0) ? __syncthreads_count(peer_rank[stage] >= 0)
                         : num_valid[stage - 1];
        if (++col == col_blocks) { col = 0; row += config::DISPATCH_COMBINE_Mb; }
    }

    if (tid == 0) {
        if (!IS_DISPATCH) {
            // Wait until the routed down GEMMs have fully written every minibatch this task reads
            const int first_global_minibatch_idx = (macrobatch_offset + first_row_idx) / G.minibatch_size;
            const int last_global_minibatch_idx = (macrobatch_offset + (first_tile_idx + num_valid_tiles - 1) / col_blocks * config::DISPATCH_COMBINE_Mb) / G.minibatch_size;
            for (int global_minibatch_idx = first_global_minibatch_idx; global_minibatch_idx <= last_global_minibatch_idx; ++global_minibatch_idx) {
                const int minibatch_rows = min(G.minibatch_size, num_tokens - global_minibatch_idx * G.minibatch_size);
                const int expected = ((minibatch_rows + config::MLP_Mb - 1) / config::MLP_Mb) * (G.y_routed.cols() / config::MLP_Nb) * config::CLUSTER_SIZE;
                int combine_counter;
                while (true) {
                    asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}" : "=r"(combine_counter) : "l"(&G.combine_counter[{global_minibatch_idx}]) : "memory");
                    if (combine_counter >= expected) break;
                    __nanosleep(16);
                }
            }
            asm volatile("{fence.acquire.gpu;}" ::: "memory");
        }
        #pragma unroll
        for (int stage = 0; stage < config::DISPATCH_COMBINE_PIPE_DEPTH; ++stage)
            if (stage < num_valid_tiles)
                tma::expect_bytes(inputs_arrived[stage], num_valid[stage] * sizeof(typename globals::token_vec)); // 0 bytes completes the phase immediately
    }
    __syncthreads();

    #pragma unroll
    for (int stage = 0; stage < config::DISPATCH_COMBINE_PIPE_DEPTH; ++stage) {
        if (peer_rank[stage] >= 0) {
            if constexpr (IS_DISPATCH)
                tma::load_async(token_vecs[stage][tid], G.x_routed_send_buffer[peer_rank[stage]], {peer_token_idx[stage] / G.topk, col_block_idx[stage]}, inputs_arrived[stage]);
            else
                tma::load_async(token_vecs[stage][tid], G.y_routed, {row_idx[stage], col_block_idx[stage]}, inputs_arrived[stage]);
        }
    }

    // Store each tile out as its loads arrive
    #pragma unroll
    for (int stage = 0; stage < config::DISPATCH_COMBINE_PIPE_DEPTH; ++stage) {
        if (stage < num_valid_tiles) {
            wait(inputs_arrived[stage], get_phasebit<0>(bitfield, stage)); // semaphores are reused across tasks
            update_phasebit<0>(bitfield, stage);
            if (peer_rank[stage] >= 0) {
                if constexpr (IS_DISPATCH)
                    tma::store_async(G.x_routed, token_vecs[stage][tid], {row_idx[stage], col_block_idx[stage]});
                else
                    tma::store_async(G.y_routed_recv_buffer[peer_rank[stage]], token_vecs[stage][tid], {peer_token_idx[stage], col_block_idx[stage]});
            }
        }
    }

    if constexpr (IS_DISPATCH) {
        tma::store_async_wait();
        __syncthreads();
        if (tid == 0) {
            #pragma unroll
            for (int stage = 0; stage < config::DISPATCH_COMBINE_PIPE_DEPTH; ++stage) {
                if (stage < num_valid_tiles) {
                    const int row_block_idx = macrobatch_offset / config::DISPATCH_COMBINE_Mb + row_idx[stage] / config::DISPATCH_COMBINE_Mb;
                    asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}" :: "l"(&G.dispatch_row_counter[{row_block_idx}]), "r"(1) : "memory");
                    asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}" :: "l"(&G.dispatch_tile_counter[{row_block_idx * col_blocks + col_block_idx[stage]}]), "r"(1) : "memory");
                }
            }
        }
    } else {
        // The next task on this CTA reuses token_vecs; make sure outgoing stores are done reading shared memory
        tma::store_async_read_wait();
        __syncthreads();
    }
}

template <bool IS_SHARED>
static __device__ __forceinline__ void quantize(
    const globals &g,
    semaphore (&quantize_inputs_arrived)[config::QUANT_PIPE_DEPTH],
    uint32_t &quantize_bitfield,
    int cta_rank,
    int macrobatch_idx,
    int minibatch_idx,
    int task_idx,
    uint64_t smem_base_addr
) {
    auto &x_bf16_tiles = *reinterpret_cast<typename globals::q_bf16_tile (*)[config::QUANT_PIPE_DEPTH]>(smem_base_addr);

    const typename globals::activation_gl &x_gmem = IS_SHARED ? g.x_shared : g.x_routed;
    const typename globals::activation_fp8_gl &x_fp8_gmem = IS_SHARED ? g.x_fp8_shared : g.x_fp8_routed;
    const typename globals::sc_gl &x_sc_gmem = IS_SHARED ? g.x_sc_shared : g.x_sc_routed;

    const int tid = threadIdx.x;
    const int group_id = tid / config::QUANT_Mb;
    const int tid_in_group = tid % config::QUANT_Mb;
    const int global_minibatch_idx = macrobatch_idx * (g.macrobatch_size / g.minibatch_size) + minibatch_idx;
    const int macrobatch_row_block_offset = macrobatch_idx * (g.macrobatch_size / config::QUANT_Mb);

    int num_tokens;
    if constexpr (IS_SHARED) num_tokens = x_gmem.rows();
    else                     num_tokens = g.num_tokens[{0}];

    const int row_blocks = num_tokens / config::QUANT_Mb;
    const int col_blocks = x_gmem.cols() / config::QUANT_Nb;
    const int num_tiles = row_blocks * col_blocks;
    int first_tile_idx, tile_end;
    if constexpr (IS_SHARED) {
        first_tile_idx = task_idx * config::CLUSTER_SIZE * config::QUANT_TILES_PER_CTA + cta_rank * config::QUANT_TILES_PER_CTA;
        tile_end = num_tiles;
    } else {
        const int num_tiles_per_minibatch = (g.minibatch_size / config::QUANT_Mb) * col_blocks;
        const int minibatch_first_tile_idx = global_minibatch_idx * num_tiles_per_minibatch;
        first_tile_idx = minibatch_first_tile_idx + task_idx * config::CLUSTER_SIZE * config::QUANT_TILES_PER_CTA + cta_rank * config::QUANT_TILES_PER_CTA;
        tile_end = min(num_tiles, minibatch_first_tile_idx + num_tiles_per_minibatch);
    }
    const int num_iters = min(config::QUANT_TILES_PER_CTA, tile_end - first_tile_idx);
    const int num_iters_per_group = max(0, (num_iters - group_id + config::QUANT_NUM_GROUPS - 1) / config::QUANT_NUM_GROUPS);
    if (num_iters <= 0) return;

    __syncthreads();

    auto issue_load = [&](int iter) {
        const int linear_idx = first_tile_idx + iter;
        const int stage = iter % config::QUANT_PIPE_DEPTH;
        const int row = linear_idx / col_blocks;
        const int col = linear_idx % col_blocks;
        if constexpr (!IS_SHARED) {
            const int dispatch_col_blocks = g.x_routed.cols() / config::DISPATCH_COMBINE_Nb;
            const int first_row_block = row * (config::QUANT_Mb / config::DISPATCH_COMBINE_Mb);
            #pragma unroll
            for (int i = 0; i < config::QUANT_Mb / config::DISPATCH_COMBINE_Mb; ++i) {
                if (g.quantize_tile_granularity) {
                    const int col_chunk = col * config::QUANT_Nb / config::DISPATCH_COMBINE_Nb;
                    int dispatch_count;
                    while (true) {
                        asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}" : "=r"(dispatch_count) : "l"(&g.dispatch_tile_counter[{(first_row_block + i) * dispatch_col_blocks + col_chunk}]) : "memory");
                        if (dispatch_count >= 1) break;
                        __nanosleep(16);
                    }
                } else {
                    int dispatch_count;
                    while (true) {
                        asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}" : "=r"(dispatch_count) : "l"(&g.dispatch_row_counter[{first_row_block + i}]) : "memory");
                        if (dispatch_count >= dispatch_col_blocks) break;
                        __nanosleep(16);
                    }
                }
            }
            asm volatile("{fence.acquire.gpu;}" ::: "memory");
        }
        tma::expect(quantize_inputs_arrived[stage], x_bf16_tiles[stage]);
        tma::load_async(x_bf16_tiles[stage], x_gmem, {row - macrobatch_row_block_offset, col}, quantize_inputs_arrived[stage]);
    };

    if (tid_in_group == 0) {
        if (num_iters_per_group > 0) issue_load(group_id);
        if (num_iters_per_group > 1) issue_load(config::QUANT_NUM_GROUPS + group_id);
    }

    for (int group_iter = 0; group_iter < num_iters_per_group; ++group_iter) {
        const int iter = group_iter * config::QUANT_NUM_GROUPS + group_id;
        const int linear_idx = first_tile_idx + iter;
        const int stage = iter % config::QUANT_PIPE_DEPTH;
        const int row = linear_idx / col_blocks;
        const int col = linear_idx % col_blocks;

        wait(quantize_inputs_arrived[stage], get_phasebit<0>(quantize_bitfield, stage));
        update_phasebit<0>(quantize_bitfield, stage);

        auto &x_fp8_tile = *reinterpret_cast<typename globals::q_fp8_tile *>(&x_bf16_tiles[stage]);
        auto &x_sc_tile = *reinterpret_cast<typename globals::sc_tile *>(reinterpret_cast<uint64_t>(&x_bf16_tiles[stage]) + sizeof(typename globals::q_fp8_tile));
        mxfp8_quantize::mxfp8_quantize_tile<true, false>(x_bf16_tiles[stage], x_fp8_tile, x_sc_tile, x_fp8_tile, x_sc_tile, tid_in_group, 1 + group_id);
        group<config::QUANT_Mb / WARP_THREADS>::sync(1 + group_id);

        if (tid_in_group == 0) {
            tma::store_async(x_fp8_gmem, x_fp8_tile, {row - macrobatch_row_block_offset, col});
            tma::store_async(x_sc_gmem, x_sc_tile, {row - macrobatch_row_block_offset, col, 0, 0});
            if (group_iter + 2 < num_iters_per_group) {
                tma::store_async_read_wait<2>();
                issue_load((group_iter + 2) * config::QUANT_NUM_GROUPS + group_id);
            }
        }
    }

    // Signal the gate/up GEMMs
    if (tid_in_group == 0) {
        tma::store_async_wait();
        asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}" :: "l"(&g.quantize_counter[{IS_SHARED ? 0 : 1 + global_minibatch_idx}]), "r"(num_iters_per_group) : "memory");
    }
    __syncthreads();
}

template <bool IS_SHARED>
static __device__ __forceinline__ void swiglu(
    const globals &g,
    semaphore (&swiglu_inputs_arrived)[config::SWIGLU_PIPE_DEPTH],
    uint32_t &swiglu_bitfield,
    int cta_rank,
    int macrobatch_idx,
    int minibatch_idx,
    int task_idx,
    uint64_t smem_base_addr
) {
    typename globals::swiglu_tile (&a_smem)[config::SWIGLU_PIPE_DEPTH] = *reinterpret_cast<typename globals::swiglu_tile (*)[config::SWIGLU_PIPE_DEPTH]>(smem_base_addr);
    typename globals::swiglu_tile (&b_smem)[config::SWIGLU_PIPE_DEPTH] = *reinterpret_cast<typename globals::swiglu_tile (*)[config::SWIGLU_PIPE_DEPTH]>(smem_base_addr + sizeof(a_smem));

    const typename globals::activation_gl &gate_gmem           = IS_SHARED ? g.gate_shared : g.gate_routed;
    const typename globals::activation_gl &up_gmem             = IS_SHARED ? g.up_shared : g.up_routed;
    const typename globals::activation_fp8_gl &hidden_fp8_gmem = IS_SHARED ? g.hidden_fp8_shared : g.hidden_fp8_routed;
    const typename globals::sc_gl &hidden_sc_gmem              = IS_SHARED ? g.hidden_sc_shared : g.hidden_sc_routed;
    const typename globals::weight_fp8_gl &w_gate_gmem         = IS_SHARED ? g.w_shared_gate : g.w_routed_gate;

    const int shared_row_blocks = g.x_shared.rows() / config::MLP_Mb;
    const int routed_row_blocks = g.schedule_peer_rank.cols() / config::MLP_Mb;
    const int shared_gate_up_tasks = shared_row_blocks * (g.w_shared_gate.rows() / config::MLP_Nb);
    const int routed_gate_up_tasks = routed_row_blocks * (g.w_routed_gate.rows() / config::MLP_Nb);
    const int gate_counter_offset = IS_SHARED ? 0 : shared_gate_up_tasks;
    const int swiglu_counter_offset = shared_gate_up_tasks + routed_gate_up_tasks + (IS_SHARED ? 0 : shared_row_blocks);
    const int intermediate_col_blocks = w_gate_gmem.rows() / config::MLP_Nb;
    const int global_minibatch_idx = macrobatch_idx * (g.macrobatch_size / g.minibatch_size) + minibatch_idx;
    const int macrobatch_row_block_offset = macrobatch_idx * (g.macrobatch_size / config::SWIGLU_Mb);

    int num_tokens;
    if constexpr (IS_SHARED) num_tokens = g.x_shared.rows();
    else                     num_tokens = g.num_tokens[{0}];

    const int row_blocks = num_tokens / config::SWIGLU_Mb;
    const int col_blocks = hidden_fp8_gmem.cols() / config::SWIGLU_Nb;
    const int num_tiles = row_blocks * col_blocks;
    int first_tile_idx, tile_end;
    if constexpr (IS_SHARED) {
        first_tile_idx = task_idx * config::CLUSTER_SIZE * config::SWIGLU_PIPE_DEPTH + cta_rank * config::SWIGLU_PIPE_DEPTH;
        tile_end = num_tiles;
    } else {
        const int num_tiles_per_minibatch = (g.minibatch_size / config::SWIGLU_Mb) * col_blocks;
        const int minibatch_first_tile_idx = global_minibatch_idx * num_tiles_per_minibatch;
        first_tile_idx = minibatch_first_tile_idx + task_idx * config::CLUSTER_SIZE * config::SWIGLU_PIPE_DEPTH + cta_rank * config::SWIGLU_PIPE_DEPTH;
        tile_end = min(num_tiles, minibatch_first_tile_idx + num_tiles_per_minibatch);
    }
    if (first_tile_idx >= tile_end)
        return;

    int first_row, first_col;
    if (threadIdx.x == 0) {
        first_row = first_tile_idx / col_blocks;
        first_col = first_tile_idx % col_blocks;
        #pragma unroll
        for (int stage = 0; stage < config::SWIGLU_PIPE_DEPTH; ++stage) {
            const int tile_idx = first_tile_idx + stage;
            if (tile_idx < tile_end) {
                // This improves throughput
                int row = first_row;
                int col = first_col + stage;
                if (col >= col_blocks) {
                    ++row;
                    col -= col_blocks;
                }
                tma::expect_bytes(swiglu_inputs_arrived[stage], sizeof(a_smem[stage]) + sizeof(b_smem[stage]));

                const int parent_task_idx = (row / (config::MLP_Mb / config::SWIGLU_Mb)) * intermediate_col_blocks + col / (config::MLP_Nb / config::SWIGLU_Nb);
                while (true) {
                    int gate_up_counter;
                    asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}" : "=r"(gate_up_counter) : "l"(&g.mlp_swiglu_counter[{gate_counter_offset + parent_task_idx}]) : "memory");
                    if (gate_up_counter >= 2 * config::CLUSTER_SIZE) break;
                    __nanosleep(16);
                }
                asm volatile("{fence.acquire.gpu;}" ::: "memory");

                tma::load_async(a_smem[stage], gate_gmem, {row - macrobatch_row_block_offset, col}, swiglu_inputs_arrived[stage]);
                tma::load_async(b_smem[stage], up_gmem,   {row - macrobatch_row_block_offset, col}, swiglu_inputs_arrived[stage]);
            }
        }
    }

    using compute_group = group<config::NUM_WARPS>;
    #pragma unroll
    for (int stage = 0; stage < config::SWIGLU_PIPE_DEPTH; ++stage) {
        const int tile_idx = first_tile_idx + stage;
        if (tile_idx < tile_end) {
            rt_fl<config::SWIGLU_Mb / config::NUM_WARPS, config::SWIGLU_Nb> gate, up, denominator;
            wait(swiglu_inputs_arrived[stage], get_phasebit<0>(swiglu_bitfield, stage));
            update_phasebit<0>(swiglu_bitfield, stage);

            compute_group::load(gate, a_smem[stage]);
            compute_group::load(up, b_smem[stage]);
            compute_group::mul(denominator, gate, -1.4426950408889634f);
            compute_group::exp2(denominator, denominator);
            compute_group::add(denominator, denominator, 1.0f);
            compute_group::div(gate, gate, denominator);
            compute_group::mul(gate, gate, up);

            auto &hidden_bf16_tile = *reinterpret_cast<typename globals::q_bf16_tile *>(&b_smem[stage]);
            auto &hidden_fp8_tile = *reinterpret_cast<typename globals::q_fp8_tile *>(&b_smem[stage]);
            auto &hidden_sc_tile = *reinterpret_cast<typename globals::sc_tile *>(reinterpret_cast<uint64_t>(&b_smem[stage]) + sizeof(typename globals::q_fp8_tile));
            __syncthreads();
            compute_group::store(hidden_bf16_tile, gate);
            __syncthreads();
            mxfp8_quantize::mxfp8_quantize_tile<true, false>(hidden_bf16_tile, hidden_fp8_tile, hidden_sc_tile, hidden_fp8_tile, hidden_sc_tile, threadIdx.x, 1);
            __syncthreads();

            if (threadIdx.x == 0) {
                // This improves throughput
                int row = first_row;
                int col = first_col + stage;
                if (col >= col_blocks) {
                    ++row;
                    col -= col_blocks;
                }
                tma::store_async(hidden_fp8_gmem, hidden_fp8_tile, {row - macrobatch_row_block_offset, col});
                tma::store_async(hidden_sc_gmem, hidden_sc_tile, {row - macrobatch_row_block_offset, col, 0, 0});
            }
        }
    }

    if (threadIdx.x == 0) {
        tma::store_async_wait();
        #pragma unroll
        for (int stage = 0; stage < config::SWIGLU_PIPE_DEPTH; ++stage) {
            const int tile_idx = first_tile_idx + stage;
            if (tile_idx < tile_end) {
                // This improves throughput
                int row = first_row;
                int col = first_col + stage;
                if (col >= col_blocks)
                    ++row;
                asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}" :: "l"(&g.mlp_swiglu_counter[{swiglu_counter_offset + row / (config::MLP_Mb / config::SWIGLU_Mb)}]), "r"(1) : "memory");
            }
        }
    }
}

enum class expert_gemm_kind { GATE, UP, DOWN };

template <bool IS_SHARED>
static __device__ __forceinline__ void expert_grouped_gemm(
    const globals &g,
    semaphore (&gemm_inputs_arrived)[config::LOAD_PIPE_DEPTH],
    semaphore (&gemm_scales_arrived)[config::LOAD_PIPE_DEPTH],
    semaphore (&gemm_inputs_finished)[config::LOAD_PIPE_DEPTH],
    semaphore &gemm_outputs_arrived,
    semaphore &gemm_outputs_finished,
    uint32_t &gemm_bitfield,
    int cta_rank,
    int macrobatch_idx,
    int minibatch_idx,
    int task_idx,
    uint64_t smem_base_addr,
    expert_gemm_kind kind,
    tt<float, config::MLP_Mb / 2, config::MLP_Nb> d_tt,
    full_tt_fp8e8m0<16 * config::LOAD_PIPE_DEPTH> a_sc_tt,
    full_tt_fp8e8m0<32 * config::LOAD_PIPE_DEPTH> b_sc_tt
) {
    typename globals::fp8_tile (&a_smem)[config::LOAD_PIPE_DEPTH]      = *reinterpret_cast<typename globals::fp8_tile (*)[config::LOAD_PIPE_DEPTH]>(smem_base_addr);
    typename globals::fp8_tile (&b_smem)[config::LOAD_PIPE_DEPTH]      = *reinterpret_cast<typename globals::fp8_tile (*)[config::LOAD_PIPE_DEPTH]>(smem_base_addr + sizeof(a_smem));
    typename globals::sc_tile (&a_sc_smem)[config::LOAD_PIPE_DEPTH]    = *reinterpret_cast<typename globals::sc_tile (*)[config::LOAD_PIPE_DEPTH]>(smem_base_addr + sizeof(a_smem) + sizeof(b_smem));
    typename globals::sc_tile (&b_sc_smem)[config::LOAD_PIPE_DEPTH][2] = *reinterpret_cast<typename globals::sc_tile (*)[config::LOAD_PIPE_DEPTH][2]>(smem_base_addr + sizeof(a_smem) + sizeof(b_smem) + sizeof(a_sc_smem));
    typename globals::d_tile (&d_smem)[config::NUM_D_TILES]            = *reinterpret_cast<typename globals::d_tile (*)[config::NUM_D_TILES]>(smem_base_addr + sizeof(a_smem) + sizeof(b_smem) + sizeof(a_sc_smem) + sizeof(b_sc_smem));

    const typename globals::activation_fp8_gl &x_fp8_gmem      = IS_SHARED ? g.x_fp8_shared : g.x_fp8_routed;
    const typename globals::sc_gl             &x_sc_gmem       = IS_SHARED ? g.x_sc_shared : g.x_sc_routed;
    const typename globals::activation_gl     &gate_gmem       = IS_SHARED ? g.gate_shared : g.gate_routed;
    const typename globals::activation_gl     &up_gmem         = IS_SHARED ? g.up_shared : g.up_routed;
    const typename globals::activation_fp8_gl &hidden_fp8_gmem = IS_SHARED ? g.hidden_fp8_shared : g.hidden_fp8_routed;
    const typename globals::sc_gl             &hidden_sc_gmem  = IS_SHARED ? g.hidden_sc_shared : g.hidden_sc_routed;
    const typename globals::activation_gl     &y_gmem          = IS_SHARED ? g.y_shared : g.y_routed;
    const typename globals::weight_fp8_gl     &w_gate_gmem     = IS_SHARED ? g.w_shared_gate : g.w_routed_gate;
    const typename globals::sc_gl             &w_gate_sc_gmem  = IS_SHARED ? g.w_shared_gate_sc : g.w_routed_gate_sc;
    const typename globals::weight_fp8_gl     &w_up_gmem       = IS_SHARED ? g.w_shared_up : g.w_routed_up;
    const typename globals::sc_gl             &w_up_sc_gmem    = IS_SHARED ? g.w_shared_up_sc : g.w_routed_up_sc;
    const typename globals::weight_fp8_gl     &w_down_gmem     = IS_SHARED ? g.w_shared_down : g.w_routed_down;
    const typename globals::sc_gl             &w_down_sc_gmem  = IS_SHARED ? g.w_shared_down_sc : g.w_routed_down_sc;

    const typename globals::activation_fp8_gl &a_gmem    = kind == expert_gemm_kind::DOWN ? hidden_fp8_gmem : x_fp8_gmem;
    const typename globals::sc_gl             &a_sc_gmem = kind == expert_gemm_kind::DOWN ? hidden_sc_gmem : x_sc_gmem;
    const typename globals::weight_fp8_gl     &b_gmem    = kind == expert_gemm_kind::GATE ? w_gate_gmem    : (kind == expert_gemm_kind::UP ? w_up_gmem : w_down_gmem);
    const typename globals::sc_gl             &b_sc_gmem = kind == expert_gemm_kind::GATE ? w_gate_sc_gmem : (kind == expert_gemm_kind::UP ? w_up_sc_gmem : w_down_sc_gmem);
    const typename globals::activation_gl     &d_gmem    = kind == expert_gemm_kind::GATE ? gate_gmem      : (kind == expert_gemm_kind::UP ? up_gmem : y_gmem);

    const int iters_per_task = a_gmem.cols() / config::MLP_Kb;
    const int col_blocks     = b_gmem.rows() / config::MLP_Nb;
    const int shared_row_blocks = g.x_shared.rows() / config::MLP_Mb;
    const int routed_row_blocks = g.schedule_peer_rank.cols() / config::MLP_Mb;
    const int shared_gate_up_tasks = shared_row_blocks * (g.w_shared_gate.rows() / config::MLP_Nb);
    const int routed_gate_up_tasks = routed_row_blocks * (g.w_routed_gate.rows() / config::MLP_Nb);
    const int gate_counter_offset = IS_SHARED ? 0 : shared_gate_up_tasks;
    const int swiglu_counter_offset = shared_gate_up_tasks + routed_gate_up_tasks + (IS_SHARED ? 0 : shared_row_blocks);
    const int global_minibatch_idx = macrobatch_idx * (g.macrobatch_size / g.minibatch_size) + minibatch_idx;
    const int macrobatch_row_block_offset = macrobatch_idx * (g.macrobatch_size / config::MLP_Mb);

    int3 tile_coord = {-1, -1, -1};
    if constexpr (IS_SHARED) {
        const int row_blocks = g.x_shared.rows() / config::MLP_Mb;
        const int num_tasks = row_blocks * col_blocks;
        if (task_idx < num_tasks) {
            const int2 swizzled = get_swizzled_2d_idx<config::SUPERGROUP_SIZE>(row_blocks, col_blocks, task_idx);
            tile_coord = {swizzled.x, swizzled.y, 0};
        }
    } else {
        const int minibatch_routed_row_blocks = g.minibatch_size / config::MLP_Mb;
        const int global_minibatch_routed_first_row_block = global_minibatch_idx * minibatch_routed_row_blocks;
        int global_row_block_offset = 0;
        for (int expert_idx = 0; expert_idx < b_gmem.depth(); ++expert_idx) {
            const int expert_row_blocks = g.tokens_per_expert[{expert_idx}] / config::MLP_Mb;
            const int global_first_row_block = max(global_minibatch_routed_first_row_block, global_row_block_offset);
            const int row_blocks = max(0, min(global_minibatch_routed_first_row_block + minibatch_routed_row_blocks, global_row_block_offset + expert_row_blocks) - global_first_row_block);
            const int num_tasks = row_blocks * col_blocks;
            if (task_idx < num_tasks) {
                const int2 swizzled = get_swizzled_2d_idx<config::SUPERGROUP_SIZE>(row_blocks, col_blocks, task_idx);
                tile_coord = {global_first_row_block + swizzled.x - macrobatch_row_block_offset, swizzled.y, expert_idx};
                break;
            }
            task_idx -= num_tasks;
            global_row_block_offset += expert_row_blocks;
        }
    }
    if (tile_coord.z < 0) return;

    auto wait_for_a_operand = [&]() {
        if (kind == expert_gemm_kind::DOWN) {
            const int expected = (config::MLP_Mb / config::SWIGLU_Mb) * (hidden_fp8_gmem.cols() / config::SWIGLU_Nb);
            int swiglu_counter;
            while (true) {
                asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}" : "=r"(swiglu_counter) : "l"(&g.mlp_swiglu_counter[{swiglu_counter_offset + macrobatch_row_block_offset + tile_coord.x}]) : "memory");
                if (swiglu_counter >= expected) break;
                __nanosleep(16);
            }
            asm volatile("{fence.acquire.gpu;}" ::: "memory");
        } else {
            int expected, counter_idx;
            if constexpr (IS_SHARED) {
                expected = (g.x_shared.rows() / config::QUANT_Mb) * (g.x_shared.cols() / config::QUANT_Nb);
                counter_idx = 0;
            } else {
                const int num_tokens = g.num_tokens[{0}];
                const int minibatch_first_row = global_minibatch_idx * g.minibatch_size;
                const int minibatch_rows = max(0, min(g.minibatch_size, num_tokens - minibatch_first_row));
                expected = (minibatch_rows / config::QUANT_Mb) * (g.x_routed.cols() / config::QUANT_Nb);
                counter_idx = 1 + global_minibatch_idx;
            }
            int quantize_counter;
            while (true) {
                asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}" : "=r"(quantize_counter) : "l"(&g.quantize_counter[{counter_idx}]) : "memory");
                if (quantize_counter >= expected) break;
                __nanosleep(16);
            }
            asm volatile("{fence.acquire.gpu;}" ::: "memory");
        }
    };

    if (warpgroup::groupid() == config::NUM_CONSUMERS) {
        if (warpgroup::warpid() == 3 && warp::elect_leader()) {
            wait_for_a_operand();
            int input_ring = 0;
            for (int idx = 0; idx < iters_per_task; ++idx) {
                wait(gemm_inputs_finished[input_ring], get_phasebit<1>(gemm_bitfield, input_ring));
                tma::cluster::load_async(a_smem[input_ring], a_gmem, {tile_coord.x * 2 + cta_rank, idx},               gemm_inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                tma::cluster::load_async(b_smem[input_ring], b_gmem, {tile_coord.z, tile_coord.y * 2 + cta_rank, idx}, gemm_inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                update_phasebit<1>(gemm_bitfield, input_ring);
                input_ring = ring_advance<config::LOAD_PIPE_DEPTH>(input_ring);
            }
        } else if (warpgroup::warpid() == 2 && warp::elect_leader()) {
            wait_for_a_operand();
            int input_ring = 0;
            for (int idx = 0; idx < iters_per_task; ++idx) {
                wait(gemm_inputs_finished[input_ring], get_phasebit<1>(gemm_bitfield, input_ring));
                tma::cluster::load_async(a_sc_smem[input_ring], a_sc_gmem, {tile_coord.x * 2 + cta_rank, idx, 0, 0}, gemm_scales_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                tma::cluster::load_async(b_sc_smem[input_ring][cta_rank], b_sc_gmem, {tile_coord.z * (b_gmem.rows() / config::QUANT_Mb) + tile_coord.y * 2 + cta_rank, idx, 0, 0}, gemm_scales_arrived[input_ring], (uint16_t)(0b11), 0);
                update_phasebit<1>(gemm_bitfield, input_ring);
                input_ring = ring_advance<config::LOAD_PIPE_DEPTH>(input_ring);
            }
        } else if (cta_rank == 0 && warpgroup::warpid() == 0 && warp::elect_leader()) {
            int input_ring = 0;
            wait(gemm_outputs_finished, get_phasebit<1>(gemm_bitfield, config::LOAD_PIPE_DEPTH));
            update_phasebit<1>(gemm_bitfield, config::LOAD_PIPE_DEPTH);
            tensor_after_thread_sync();
            for (int idx = 0; idx < iters_per_task; ++idx) {
                tma::expect_bytes(gemm_scales_arrived[input_ring], config::CLUSTER_SIZE * 3 * sizeof(typename globals::sc_tile));
                wait(gemm_scales_arrived[input_ring], get_phasebit<0>(gemm_bitfield, input_ring));
                load_mxnv_scale_async2(a_sc_tt.template subtile<full_tt_fp8e8m0<16>>(input_ring * 16), a_sc_smem[input_ring]);
                load_mxnv_scale_async2(b_sc_tt.template subtile<full_tt_fp8e8m0<16>>(input_ring * 32), b_sc_smem[input_ring][0]);
                load_mxnv_scale_async2(b_sc_tt.template subtile<full_tt_fp8e8m0<16>>(input_ring * 32 + 16), b_sc_smem[input_ring][1]);
                tma::expect_bytes(gemm_inputs_arrived[input_ring], config::CLUSTER_SIZE * 2 * sizeof(typename globals::fp8_tile));
                wait(gemm_inputs_arrived[input_ring], get_phasebit<0>(gemm_bitfield, input_ring));
                if (idx == 0) mm2_ABt (d_tt, a_smem[input_ring], b_smem[input_ring],
                                       a_sc_tt.template subtile<full_tt_fp8e8m0<16>>(input_ring * 16),
                                       b_sc_tt.template subtile<full_tt_fp8e8m0<32>>(input_ring * 32),
                                       gemm_inputs_finished[input_ring]);
                else          mma2_ABt(d_tt, a_smem[input_ring], b_smem[input_ring],
                                       a_sc_tt.template subtile<full_tt_fp8e8m0<16>>(input_ring * 16),
                                       b_sc_tt.template subtile<full_tt_fp8e8m0<32>>(input_ring * 32),
                                       gemm_inputs_finished[input_ring]);
                update_phasebit<0>(gemm_bitfield, input_ring);
                input_ring = ring_advance<config::LOAD_PIPE_DEPTH>(input_ring);
            }
            detail::tcgen05::commit<config::CLUSTER_SIZE>(gemm_outputs_arrived);
        }
    } else {
        using epilogue_group = group<WARPGROUP_WARPS>;
        wait(gemm_outputs_arrived, get_phasebit<0>(gemm_bitfield, config::LOAD_PIPE_DEPTH));
        update_phasebit<0>(gemm_bitfield, config::LOAD_PIPE_DEPTH);
        rt_bf<config::MLP_Mb / 8, config::MLP_Nb / config::EPI_PIPE_DEPTH> d_reg[config::EPI_PIPE_DEPTH];
        #pragma unroll
        for (int i = 0; i < config::EPI_PIPE_DEPTH; ++i)
            warpgroup::load_async(d_reg[i], d_tt.template subtile<tt<float, config::MLP_Mb / 2, config::MLP_Nb / config::EPI_PIPE_DEPTH>>(0, config::MLP_Nb / config::EPI_PIPE_DEPTH * i));
        tensor_load_wait();
        warpgroup::sync(1);
        warpgroup::tma::cluster::arrive(gemm_outputs_finished, 0);
        #pragma unroll
        for (int i = 0; i < config::EPI_PIPE_DEPTH; ++i) {
            warpgroup::tma::store_async_read_wait<config::NUM_D_TILES - 1>();
            warpgroup::sync(1);
            warpgroup::store(d_smem[i % config::NUM_D_TILES], d_reg[i]);
            warpgroup::sync(1);
            warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(d_gmem, d_smem[i % config::NUM_D_TILES], {2 * tile_coord.x + cta_rank, config::EPI_PIPE_DEPTH * tile_coord.y + i});
        }
        epilogue_group::sync(4);
        if (epilogue_group::warpid() == 0 && warp::elect_leader()) {
            if (kind != expert_gemm_kind::DOWN) {
                // Up/gate is complete; signal Swiglu
                tma::store_async_wait();
                asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}" :: "l"(&g.mlp_swiglu_counter[{gate_counter_offset + (macrobatch_row_block_offset + tile_coord.x) * col_blocks + tile_coord.y}]), "r"(1) : "memory");
            } else if constexpr (!IS_SHARED) {
                // Routed down is complete; signal combine
                tma::store_async_wait();
                asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}" :: "l"(&g.combine_counter[{global_minibatch_idx}]), "r"(1) : "memory");
            }
        }
    }
}

static __device__ __forceinline__ void dispatch_mlp_swiglu_combine_kernel(const globals &g) {
    int cluster_idx = clusterIdx().x;
    const int cta_rank = cluster_ctarank();
    const int shared_row_blocks = g.x_shared.rows() / config::MLP_Mb;
    const int minibatch_routed_row_blocks = g.minibatch_size / config::MLP_Mb;
    const int shared_quantize_tiles = (g.x_shared.rows() / config::QUANT_Mb) * (g.x_shared.cols() / config::QUANT_Nb);
    const int minibatch_routed_quantize_tiles = (g.minibatch_size / config::QUANT_Mb) * (g.x_routed.cols() / config::QUANT_Nb);
    const int shared_quantize_tasks = (shared_quantize_tiles + config::CLUSTER_SIZE * config::QUANT_TILES_PER_CTA - 1) / (config::CLUSTER_SIZE * config::QUANT_TILES_PER_CTA);
    const int minibatch_routed_quantize_tasks = (minibatch_routed_quantize_tiles + config::CLUSTER_SIZE * config::QUANT_TILES_PER_CTA - 1) / (config::CLUSTER_SIZE * config::QUANT_TILES_PER_CTA);
    const int shared_gate_up_tasks = shared_row_blocks * (g.w_shared_gate.rows() / config::MLP_Nb);
    const int minibatch_routed_gate_up_tasks = minibatch_routed_row_blocks * (g.w_routed_gate.rows() / config::MLP_Nb);
    const int shared_swiglu_tiles = (g.hidden_fp8_shared.rows() / config::SWIGLU_Mb) * (g.hidden_fp8_shared.cols() / config::SWIGLU_Nb);
    const int minibatch_routed_swiglu_tiles = (g.minibatch_size / config::SWIGLU_Mb) * (g.hidden_fp8_routed.cols() / config::SWIGLU_Nb);
    const int shared_swiglu_tasks = (shared_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_PIPE_DEPTH);
    const int minibatch_routed_swiglu_tasks = (minibatch_routed_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_PIPE_DEPTH);
    const int shared_down_tasks = shared_row_blocks * (g.w_shared_down.rows() / config::MLP_Nb);
    const int minibatch_routed_down_tasks = minibatch_routed_row_blocks * (g.w_routed_down.rows() / config::MLP_Nb);
    const int shared_tasks = shared_quantize_tasks + 2 * shared_gate_up_tasks + shared_swiglu_tasks + shared_down_tasks;
    const int minibatch_tasks = minibatch_routed_quantize_tasks + 2 * minibatch_routed_gate_up_tasks + minibatch_routed_swiglu_tasks + minibatch_routed_down_tasks;
    const int comm_clusters = g.num_comm_sms / config::CLUSTER_SIZE;
    const int macrobatch_size = g.macrobatch_size;

    const int num_tokens = g.num_tokens[{0}];
    const int true_num_global_minibatches = (num_tokens + g.minibatch_size - 1) / g.minibatch_size;
    const int true_num_clusters = comm_clusters + shared_tasks + true_num_global_minibatches * minibatch_tasks;
    if (cluster_idx >= true_num_clusters) return;

    warpgroup::increase_registers<256>();

    extern __shared__ int __shm[];
    const uint64_t smem_base_addr = (reinterpret_cast<uint64_t>(&__shm[0]) + 1023) & ~uint64_t(1023);

    uint32_t gemm_bitfield = 0xFFFF0000;
    uint32_t quantize_bitfield = 0xFFFF0000;
    uint32_t swiglu_bitfield = 0xFFFF0000;
    uint32_t dispatch_combine_bitfield = 0xFFFF0000;

    __shared__ clc::handle clc_handle[config::CLC_PIPE_DEPTH];
    __shared__ clc::handle clc_drain_handle[config::CLC_DRAIN_PIPE_DEPTH];
    __shared__ semaphore schedule_arrived[config::CLC_PIPE_DEPTH], schedule_finished[config::CLC_PIPE_DEPTH];
    __shared__ semaphore drain_schedule_arrived[config::CLC_DRAIN_PIPE_DEPTH];
    __shared__ semaphore quantize_inputs_arrived[config::QUANT_PIPE_DEPTH];
    __shared__ semaphore swiglu_inputs_arrived[config::SWIGLU_PIPE_DEPTH];
    __shared__ semaphore gemm_inputs_arrived[config::LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_scales_arrived[config::LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_inputs_finished[config::LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_outputs_arrived, gemm_outputs_finished;
    __shared__ semaphore dispatch_combine_inputs_arrived[config::DISPATCH_COMBINE_PIPE_DEPTH];

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < config::QUANT_PIPE_DEPTH; ++i) {
            init_semaphore(quantize_inputs_arrived[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < config::SWIGLU_PIPE_DEPTH; ++i) {
            init_semaphore(swiglu_inputs_arrived[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < config::LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(gemm_inputs_arrived[i], 0, 1);
            init_semaphore(gemm_scales_arrived[i], 0, 1);
            init_semaphore(gemm_inputs_finished[i], 0, 1);
        }
        init_semaphore(gemm_outputs_arrived, 0, 1);
        init_semaphore(gemm_outputs_finished, 0, config::CLUSTER_SIZE);
        #pragma unroll
        for (int i = 0; i < config::CLC_PIPE_DEPTH; ++i) {
            init_semaphore(schedule_arrived[i], 0, 1);
            init_semaphore(schedule_finished[i], 0, config::CLUSTER_SIZE * config::NUM_WARPS);
        }
        #pragma unroll
        for (int i = 0; i < config::CLC_DRAIN_PIPE_DEPTH; ++i) {
            init_semaphore(drain_schedule_arrived[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < config::DISPATCH_COMBINE_PIPE_DEPTH; ++i) {
            init_semaphore(dispatch_combine_inputs_arrived[i], 0, 1);
        }
    }

    tensor_allocator<1, config::CLUSTER_SIZE> tm_alloc{};
    tt<float, config::MLP_Mb / 2, config::MLP_Nb> d_tt = tm_alloc.template allocate<tt<float, config::MLP_Mb / 2, config::MLP_Nb>>(0);
    full_tt_fp8e8m0<16 * config::LOAD_PIPE_DEPTH> a_sc_tt = tm_alloc.template allocate<full_tt_fp8e8m0<16 * config::LOAD_PIPE_DEPTH>>(256);
    full_tt_fp8e8m0<32 * config::LOAD_PIPE_DEPTH> b_sc_tt = tm_alloc.template allocate<full_tt_fp8e8m0<32 * config::LOAD_PIPE_DEPTH>>(384);
    everyone::tma::cluster::sync();

    if (cluster_idx < comm_clusters) {
        const int comm_cta_idx = cluster_idx * config::CLUSTER_SIZE + cta_rank;
        const int num_macrobatches = (num_tokens + macrobatch_size - 1) / macrobatch_size;
        auto num_dispatch_combine_tasks = [&](int macrobatch_idx) {
            const int macrobatch_tokens = min(macrobatch_size, num_tokens - macrobatch_idx * macrobatch_size);
            const int dispatch_combine_tiles = (macrobatch_tokens / config::DISPATCH_COMBINE_Mb) * (g.x_routed.cols() / config::DISPATCH_COMBINE_Nb);
            return (dispatch_combine_tiles + config::DISPATCH_COMBINE_PIPE_DEPTH - 1) / config::DISPATCH_COMBINE_PIPE_DEPTH;
        };
        for (int task_idx = comm_cta_idx; task_idx < num_dispatch_combine_tasks(0); task_idx += g.num_comm_sms)
            dispatch_combine_kernel<true>(g, dispatch_combine_inputs_arrived, dispatch_combine_bitfield, 0, task_idx, smem_base_addr);
        for (int macrobatch_idx = 0; macrobatch_idx < num_macrobatches; ++macrobatch_idx) {
            const int combine_tasks = num_dispatch_combine_tasks(macrobatch_idx);
            const int dispatch_tasks = macrobatch_idx + 1 < num_macrobatches ? num_dispatch_combine_tasks(macrobatch_idx + 1) : 0;
            for (int task_idx = comm_cta_idx; task_idx < combine_tasks; task_idx += g.num_comm_sms) {
                dispatch_combine_kernel<false>(g, dispatch_combine_inputs_arrived, dispatch_combine_bitfield, macrobatch_idx, task_idx, smem_base_addr);
                if (task_idx < dispatch_tasks)
                    dispatch_combine_kernel<true>(g, dispatch_combine_inputs_arrived, dispatch_combine_bitfield, macrobatch_idx + 1, task_idx, smem_base_addr);
            }
        }
        return;
    }

    // Quantize and Swiglu tasks are CTA-local, GEMM is not
    auto is_cta_local_task = [&](int comp_cluster_idx) {
        const int minibatch_task_idx = (comp_cluster_idx - shared_tasks) % minibatch_tasks;
        if (comp_cluster_idx < 0) return false;
        else if (comp_cluster_idx < shared_quantize_tasks) return true; // shared quantize
        else if (comp_cluster_idx < shared_quantize_tasks + 2 * shared_gate_up_tasks) return false; // shared gate/up
        else if (comp_cluster_idx < shared_quantize_tasks + 2 * shared_gate_up_tasks + shared_swiglu_tasks) return true; // shared swiglu
        else if (comp_cluster_idx < shared_tasks) return false; // shared down
        else if (minibatch_task_idx < minibatch_routed_quantize_tasks) return true; // routed quantize
        else if (minibatch_task_idx < minibatch_routed_quantize_tasks + 2 * minibatch_routed_gate_up_tasks) return false; // routed gate/up
        else if (minibatch_task_idx < minibatch_routed_quantize_tasks + 2 * minibatch_routed_gate_up_tasks + minibatch_routed_swiglu_tasks) return true; // routed swiglu
        else return false; // routed down
    };

    for (int task_iter = 0; cluster_idx >= 0 && cluster_idx < true_num_clusters; ++task_iter) {
        const int clc_stage = task_iter % config::CLC_PIPE_DEPTH;
        if (warpgroup::groupid() == config::NUM_CONSUMERS && warpgroup::warpid() == 1 && warp::elect_leader()) { // warp not used by the gemms
            if (cta_rank == 0) {
                wait(schedule_finished[clc_stage], ((task_iter + config::CLC_PIPE_DEPTH) / config::CLC_PIPE_DEPTH) % 2);
                clc::schedule(clc_handle[clc_stage], schedule_arrived[clc_stage]);
            }
            tma::expect_bytes(schedule_arrived[clc_stage], sizeof(clc_handle[clc_stage]));
        }

        const int comp_cluster_idx = cluster_idx - comm_clusters;
        bool current_is_cta_local = is_cta_local_task(comp_cluster_idx);

        if (comp_cluster_idx < shared_quantize_tasks) {
            // Shared quantize
            const int task_idx = comp_cluster_idx;
            quantize<true>(g, quantize_inputs_arrived, quantize_bitfield, cta_rank, 0, 0, task_idx, smem_base_addr);
        } else if (comp_cluster_idx < shared_quantize_tasks + shared_gate_up_tasks) {
            // Shared gate
            const int task_idx = comp_cluster_idx - shared_quantize_tasks;
            expert_grouped_gemm<true>(g, gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished,
                                      gemm_bitfield, cta_rank, 0, 0, task_idx, smem_base_addr, expert_gemm_kind::GATE, d_tt, a_sc_tt, b_sc_tt);
        } else if (comp_cluster_idx < shared_quantize_tasks + shared_gate_up_tasks * 2) {
            // Shared up
            const int task_idx = comp_cluster_idx - shared_quantize_tasks - shared_gate_up_tasks;
            expert_grouped_gemm<true>(g, gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished,
                                      gemm_bitfield, cta_rank, 0, 0, task_idx, smem_base_addr, expert_gemm_kind::UP, d_tt, a_sc_tt, b_sc_tt);
        } else if (comp_cluster_idx < shared_quantize_tasks + shared_gate_up_tasks * 2 + shared_swiglu_tasks) {
            // Shared Swiglu
            const int task_idx = comp_cluster_idx - shared_quantize_tasks - shared_gate_up_tasks * 2;
            swiglu<true>(g, swiglu_inputs_arrived, swiglu_bitfield, cta_rank, 0, 0, task_idx, smem_base_addr);
        } else if (comp_cluster_idx < shared_tasks) {
            // Shared down
            const int task_idx = comp_cluster_idx - shared_quantize_tasks - shared_gate_up_tasks * 2 - shared_swiglu_tasks;
            expert_grouped_gemm<true>(g, gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished,
                                      gemm_bitfield, cta_rank, 0, 0, task_idx, smem_base_addr, expert_gemm_kind::DOWN, d_tt, a_sc_tt, b_sc_tt);
        } else {
            // Routed expert with macro/minibatching
            const int global_minibatch_idx = (comp_cluster_idx - shared_tasks) / minibatch_tasks;
            const int minibatch_task_idx = (comp_cluster_idx - shared_tasks) - global_minibatch_idx * minibatch_tasks;
            const int minibatches_per_macrobatch = macrobatch_size / g.minibatch_size;
            const int macrobatch_idx = global_minibatch_idx / minibatches_per_macrobatch;
            const int minibatch_idx = global_minibatch_idx - macrobatch_idx * minibatches_per_macrobatch;

            if (minibatch_task_idx < minibatch_routed_quantize_tasks) {
                // Routed quantize
                const int task_idx = minibatch_task_idx;
                quantize<false>(g, quantize_inputs_arrived, quantize_bitfield, cta_rank, macrobatch_idx, minibatch_idx, task_idx, smem_base_addr);
            } else if (minibatch_task_idx < minibatch_routed_quantize_tasks + minibatch_routed_gate_up_tasks) {
                // Routed gate
                const int task_idx = minibatch_task_idx - minibatch_routed_quantize_tasks;
                expert_grouped_gemm<false>(g, gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished,
                                           gemm_bitfield, cta_rank, macrobatch_idx, minibatch_idx, task_idx, smem_base_addr, expert_gemm_kind::GATE, d_tt, a_sc_tt, b_sc_tt);
            } else if (minibatch_task_idx < minibatch_routed_quantize_tasks + minibatch_routed_gate_up_tasks * 2) {
                // Routed up
                const int task_idx = minibatch_task_idx - minibatch_routed_quantize_tasks - minibatch_routed_gate_up_tasks;
                expert_grouped_gemm<false>(g, gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished,
                                           gemm_bitfield, cta_rank, macrobatch_idx, minibatch_idx, task_idx, smem_base_addr, expert_gemm_kind::UP, d_tt, a_sc_tt, b_sc_tt);
            } else if (minibatch_task_idx < minibatch_routed_quantize_tasks + minibatch_routed_gate_up_tasks * 2 + minibatch_routed_swiglu_tasks) {
                // Routed Swiglu
                const int task_idx = minibatch_task_idx - minibatch_routed_quantize_tasks - minibatch_routed_gate_up_tasks * 2;
                swiglu<false>(g, swiglu_inputs_arrived, swiglu_bitfield, cta_rank, macrobatch_idx, minibatch_idx, task_idx, smem_base_addr);
            } else {
                // Routed down
                const int task_idx = minibatch_task_idx - minibatch_routed_quantize_tasks - minibatch_routed_gate_up_tasks * 2 - minibatch_routed_swiglu_tasks;
                expert_grouped_gemm<false>(g, gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished,
                                           gemm_bitfield, cta_rank, macrobatch_idx, minibatch_idx, task_idx, smem_base_addr, expert_gemm_kind::DOWN, d_tt, a_sc_tt, b_sc_tt);
            }
        }

        wait(schedule_arrived[clc_stage], (task_iter / config::CLC_PIPE_DEPTH) % 2);
        const auto schedule = clc::query(clc_handle[clc_stage]);
        cluster_idx = schedule.success ? static_cast<int>(schedule.x / config::CLUSTER_SIZE) : -1;
        __syncwarp();
        warp::tma::cluster::arrive(schedule_finished[clc_stage], 0);

        // Quantize/SWIGLU -> GEMM requires a cluster-wide sync
        const int next_comp_cluster_idx = cluster_idx - comm_clusters;
        if (current_is_cta_local && cluster_idx >= 0 && !is_cta_local_task(next_comp_cluster_idx))
            everyone::tma::cluster::sync();
    }

    everyone::tma::cluster::sync();

    // CLC drain for no-op threadblocks
    if (cluster_idx >= 0 && warpgroup::groupid() == config::NUM_CONSUMERS && warpgroup::warpid() == 1 && warp::elect_leader()) {
        #pragma unroll
        for (int i = 0; i < config::CLC_DRAIN_PIPE_DEPTH; ++i) {
            if (cta_rank == 0) clc::schedule(clc_drain_handle[i], drain_schedule_arrived[i]);
            tma::expect_bytes(drain_schedule_arrived[i], sizeof(clc::handle));
        }
        for (int i = 0;; ++i) {
            const int stage = i % config::CLC_DRAIN_PIPE_DEPTH;
            wait(drain_schedule_arrived[stage], (i / config::CLC_DRAIN_PIPE_DEPTH) % 2);
            if (!clc::query(clc_drain_handle[stage]).success) break; // no worries bc we can let few leftovers launch and exit early
            if (cta_rank == 0) clc::schedule(clc_drain_handle[stage], drain_schedule_arrived[stage]);
            tma::expect_bytes(drain_schedule_arrived[stage], sizeof(clc::handle));
        }
    }
}

static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
dispatch_mlp_swiglu_combine(
    // Inputs and communication buffers
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,
    const at::Tensor &combine_buffer,
    const std::vector<int64_t> &combine_buffer_ptrs,

    // Weights
    const at::Tensor &w_shared_gate,
    const at::Tensor &w_shared_gate_sc,
    const at::Tensor &w_routed_gate,
    const at::Tensor &w_routed_gate_sc,
    const at::Tensor &w_shared_up,
    const at::Tensor &w_shared_up_sc,
    const at::Tensor &w_routed_up,
    const at::Tensor &w_routed_up_sc,
    const at::Tensor &w_shared_down,
    const at::Tensor &w_shared_down_sc,
    const at::Tensor &w_routed_down,
    const at::Tensor &w_routed_down_sc,

    // Dispatch/combine schedule
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    const at::Tensor &tokens_per_expert,

    // Metadata
    int topk,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size,
    int quantize_tile_granularity
) {
    const int num_local_tokens = x.size(0);
    const int schedule_capacity = schedule_peer_rank.size(0);
    const int model_dim = x.size(1);
    const int intermediate_dim = w_shared_gate.size(0);
    const int num_global_minibatches = (schedule_capacity + minibatch_size - 1) / minibatch_size;
    const int shared_row_blocks = num_local_tokens / config::MLP_Mb;
    const int routed_row_blocks = schedule_capacity / config::MLP_Mb;
    const int shared_gate_up_tasks = shared_row_blocks * (w_shared_gate.size(0) / config::MLP_Nb);
    const int routed_gate_up_tasks = routed_row_blocks * (w_routed_gate.size(1) / config::MLP_Nb);
    const int dispatch_row_blocks = schedule_capacity / config::DISPATCH_COMBINE_Mb;
    const int dispatch_col_blocks = model_dim / config::DISPATCH_COMBINE_Nb;

    bf16 *x_routed_send_buffer_data[NUM_DEVICES];
    bf16 *y_routed_recv_buffer_data[NUM_DEVICES];
    for (int i = 0; i < NUM_DEVICES; ++i) {
        x_routed_send_buffer_data[i] = reinterpret_cast<bf16*>(x_ptrs[i]);
        y_routed_recv_buffer_data[i] = reinterpret_cast<bf16*>(combine_buffer_ptrs[i]);
    }

    at::Tensor x_routed = at::empty({macrobatch_size, model_dim}, x.options());
    at::Tensor x_fp8_shared = at::empty({num_local_tokens, model_dim}, x.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor x_fp8_routed = at::empty({macrobatch_size, model_dim}, x.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor x_sc_shared = at::empty({num_local_tokens / 128, model_dim / 128, 32, 16}, x.options().dtype(at::kByte));
    at::Tensor x_sc_routed = at::empty({macrobatch_size / 128, model_dim / 128, 32, 16}, x.options().dtype(at::kByte));
    at::Tensor gate_shared = at::empty({num_local_tokens, intermediate_dim}, x.options());
    at::Tensor gate_routed = at::empty({macrobatch_size, intermediate_dim}, x.options());
    at::Tensor up_shared = at::empty({num_local_tokens, intermediate_dim}, x.options());
    at::Tensor up_routed = at::empty({macrobatch_size, intermediate_dim}, x.options());
    at::Tensor hidden_fp8_shared = at::empty({num_local_tokens, intermediate_dim}, x.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor hidden_fp8_routed = at::empty({macrobatch_size, intermediate_dim}, x.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor hidden_sc_shared = at::empty({num_local_tokens / 128, intermediate_dim / 128, 32, 16}, x.options().dtype(at::kByte));
    at::Tensor hidden_sc_routed = at::empty({macrobatch_size / 128, intermediate_dim / 128, 32, 16}, x.options().dtype(at::kByte));
    at::Tensor y_shared = at::empty_like(x);
    at::Tensor y_routed = at::empty_like(x_routed);
    at::Tensor mlp_swiglu_counter = at::zeros({shared_gate_up_tasks + routed_gate_up_tasks + shared_row_blocks + routed_row_blocks}, tokens_per_expert.options());
    at::Tensor quantize_counter = at::zeros({1 + num_global_minibatches}, tokens_per_expert.options());
    at::Tensor dispatch_row_counter = at::zeros({dispatch_row_blocks}, tokens_per_expert.options());
    at::Tensor dispatch_tile_counter = at::zeros({dispatch_row_blocks * dispatch_col_blocks}, tokens_per_expert.options());
    at::Tensor combine_counter = at::zeros({num_global_minibatches}, tokens_per_expert.options());

    globals g {
        .x_shared = kittens::py::tensor_to_gl<typename globals::activation_gl>(x),
        .x_routed = kittens::py::tensor_to_gl<typename globals::activation_gl>(x_routed),
        .x_fp8_shared = kittens::py::tensor_to_gl<typename globals::activation_fp8_gl>(x_fp8_shared),
        .x_fp8_routed = kittens::py::tensor_to_gl<typename globals::activation_fp8_gl>(x_fp8_routed),
        .x_sc_shared = kittens::py::tensor_to_gl<typename globals::sc_gl>(x_sc_shared),
        .x_sc_routed = kittens::py::tensor_to_gl<typename globals::sc_gl>(x_sc_routed),
        .gate_shared = kittens::py::tensor_to_gl<typename globals::activation_gl>(gate_shared),
        .gate_routed = kittens::py::tensor_to_gl<typename globals::activation_gl>(gate_routed),
        .up_shared = kittens::py::tensor_to_gl<typename globals::activation_gl>(up_shared),
        .up_routed = kittens::py::tensor_to_gl<typename globals::activation_gl>(up_routed),
        .hidden_fp8_shared = kittens::py::tensor_to_gl<typename globals::activation_fp8_gl>(hidden_fp8_shared),
        .hidden_fp8_routed = kittens::py::tensor_to_gl<typename globals::activation_fp8_gl>(hidden_fp8_routed),
        .hidden_sc_shared = kittens::py::tensor_to_gl<typename globals::sc_gl>(hidden_sc_shared),
        .hidden_sc_routed = kittens::py::tensor_to_gl<typename globals::sc_gl>(hidden_sc_routed),
        .y_shared = kittens::py::tensor_to_gl<typename globals::activation_gl>(y_shared),
        .y_routed = kittens::py::tensor_to_gl<typename globals::activation_gl>(y_routed),
        .x_routed_send_buffer = typename globals::activation_pgl{x_routed_send_buffer_data, nullptr, nullptr, static_cast<size_t>(num_local_tokens), static_cast<size_t>(model_dim)},
        .y_routed_recv_buffer = typename globals::activation_pgl{y_routed_recv_buffer_data, nullptr, nullptr, static_cast<size_t>(num_local_tokens * topk), static_cast<size_t>(model_dim)},
        .w_shared_gate = kittens::py::tensor_to_gl<typename globals::weight_fp8_gl>(w_shared_gate),
        .w_routed_gate = kittens::py::tensor_to_gl<typename globals::weight_fp8_gl>(w_routed_gate),
        .w_shared_up = kittens::py::tensor_to_gl<typename globals::weight_fp8_gl>(w_shared_up),
        .w_routed_up = kittens::py::tensor_to_gl<typename globals::weight_fp8_gl>(w_routed_up),
        .w_shared_down = kittens::py::tensor_to_gl<typename globals::weight_fp8_gl>(w_shared_down),
        .w_routed_down = kittens::py::tensor_to_gl<typename globals::weight_fp8_gl>(w_routed_down),
        .w_shared_gate_sc = kittens::py::tensor_to_gl<typename globals::sc_gl>(w_shared_gate_sc),
        .w_routed_gate_sc = kittens::py::tensor_to_gl<typename globals::sc_gl>(w_routed_gate_sc),
        .w_shared_up_sc = kittens::py::tensor_to_gl<typename globals::sc_gl>(w_shared_up_sc),
        .w_routed_up_sc = kittens::py::tensor_to_gl<typename globals::sc_gl>(w_routed_up_sc),
        .w_shared_down_sc = kittens::py::tensor_to_gl<typename globals::sc_gl>(w_shared_down_sc),
        .w_routed_down_sc = kittens::py::tensor_to_gl<typename globals::sc_gl>(w_routed_down_sc),
        .schedule_peer_rank = kittens::py::tensor_to_gl<typename globals::index_gl>(schedule_peer_rank),
        .schedule_peer_token_idx = kittens::py::tensor_to_gl<typename globals::index_gl>(schedule_peer_token_idx),
        .num_tokens = kittens::py::tensor_to_gl<typename globals::index_gl>(num_tokens),
        .tokens_per_expert = kittens::py::tensor_to_gl<typename globals::index_gl>(tokens_per_expert),
        .mlp_swiglu_counter = kittens::py::tensor_to_gl<typename globals::index_gl>(mlp_swiglu_counter),
        .quantize_counter = kittens::py::tensor_to_gl<typename globals::index_gl>(quantize_counter),
        .dispatch_row_counter = kittens::py::tensor_to_gl<typename globals::index_gl>(dispatch_row_counter),
        .dispatch_tile_counter = kittens::py::tensor_to_gl<typename globals::index_gl>(dispatch_tile_counter),
        .combine_counter = kittens::py::tensor_to_gl<typename globals::index_gl>(combine_counter),
        .topk = topk,
        .num_comm_sms = num_comm_sms,
        .macrobatch_size = macrobatch_size,
        .minibatch_size = minibatch_size,
        .quantize_tile_granularity = quantize_tile_granularity
    };

    kittens::py::launch_kernel<config, globals, dispatch_mlp_swiglu_combine_kernel>(g);

    return {x_routed, gate_shared, gate_routed, up_shared, up_routed,
            hidden_fp8_shared, hidden_sc_shared, hidden_fp8_routed, hidden_sc_routed,
            y_shared, y_routed, combine_buffer};
}

}; // struct dispatch_mlp_swiglu_combiner

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("schedule", &scheduler::schedule, "",
          pybind11::arg("topk_all"), pybind11::arg("num_local_experts"), pybind11::arg("schedule_capacity"), pybind11::arg("rank"));
    m.def("mxfp8_quantize", &mxfp8_quantize_entrypoint, "",
          pybind11::arg("x_bf16"),
          pybind11::arg("x_fp8"), pybind11::arg("x_sc"),
          pybind11::arg("x_fp8_t"), pybind11::arg("x_sc_t"),
          pybind11::arg("return_normal"), pybind11::arg("return_transposed"));
    m.def("dispatch_mlp_swiglu_combine", &dispatch_mlp_swiglu_combiner<4>::dispatch_mlp_swiglu_combine, "",
          pybind11::arg("x"), pybind11::arg("x_ptrs"),
          pybind11::arg("combine_buffer"), pybind11::arg("combine_buffer_ptrs"),
          pybind11::arg("w_shared_gate"), pybind11::arg("w_shared_gate_sc"),
          pybind11::arg("w_routed_gate"), pybind11::arg("w_routed_gate_sc"),
          pybind11::arg("w_shared_up"), pybind11::arg("w_shared_up_sc"),
          pybind11::arg("w_routed_up"), pybind11::arg("w_routed_up_sc"),
          pybind11::arg("w_shared_down"), pybind11::arg("w_shared_down_sc"),
          pybind11::arg("w_routed_down"), pybind11::arg("w_routed_down_sc"),
          pybind11::arg("schedule_peer_rank"), pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("num_tokens"), pybind11::arg("tokens_per_expert"),
          pybind11::arg("topk"), pybind11::arg("num_comm_sms"),
          pybind11::arg("macrobatch_size"), pybind11::arg("minibatch_size"),
          pybind11::arg("quantize_tile_granularity"));
}
