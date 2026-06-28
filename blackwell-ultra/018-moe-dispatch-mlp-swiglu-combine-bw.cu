#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros.h>

using namespace kittens;

namespace scheduler {

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
    index_gl schedule_peer_rank;         // (capacity,) must be initialized to -1
    index_gl schedule_peer_token_idx;    // (capacity,) original_token_idx * topk + k
    index_gl tokens_per_expert;          // (num_local_experts,) padded per-expert token counts
    index_gl tokens_per_expert_and_peer; // (num_local_experts * world_size,) per-(local_expert, peer_rank) token counts, must be zero-initialized

    int rank;                            // this (destination) rank
};

// Stage 1: Count the number of tokens routed from each peer rank to each local expert
__global__ void count_kernel(const __grid_constant__ globals G) {
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

// Stage 2: Pad each expert's total token count by EXPERT_PADDING
__global__ void pad_kernel(const __grid_constant__ globals G) {
    const int local_expert = blockIdx.x;
    const int world_size = G.topk.depth();
    int num_tokens = 0;
    for (int peer_rank = 0; peer_rank < world_size; ++peer_rank)
        num_tokens += G.tokens_per_expert_and_peer[{local_expert * world_size + peer_rank}];
    G.tokens_per_expert[{local_expert}] = (num_tokens + config::EXPERT_PADDING - 1) / config::EXPERT_PADDING * config::EXPERT_PADDING;
}

// Stage 3: Schedule each token into its expert's 256-padded segment
__global__ void schedule_kernel(const __grid_constant__ globals G) {
    const int world_size = G.topk.depth();
    const int num_local_tokens = G.topk.rows();
    const int topk = G.topk.cols();
    const int rank_stride = num_local_tokens * topk;
    const int num_local_experts = G.tokens_per_expert.cols();
    const int first_expert = G.rank * num_local_experts;

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
                // TODO: prevent capacity overflow
                G.schedule_peer_rank[{dst_token_idx}] = peer_rank;
                G.schedule_peer_token_idx[{dst_token_idx}] = peer_token_idx; // original_token_idx * topk + k
                ++j;
            }
        }
        __syncthreads(); // before the next iteration reuses cumulative_tokens_from_peer_rank
    }
}

__host__ void schedule(
    const at::Tensor &topk_all,
    at::Tensor &schedule_peer_rank, // must be initialized to -1
    at::Tensor &schedule_peer_token_idx,
    at::Tensor &tokens_per_expert,
    at::Tensor &tokens_per_expert_and_peer, // must be zero'ed
    const int rank
) {
    const int world_size = static_cast<int>(topk_all.size(0));
    const int num_local_experts = static_cast<int>(tokens_per_expert.size(0));

    globals G {
        .topk = kittens::py::tensor_to_gl<globals::topk_gl>(topk_all),
        .schedule_peer_rank = kittens::py::tensor_to_gl<globals::index_gl>(schedule_peer_rank),
        .schedule_peer_token_idx = kittens::py::tensor_to_gl<globals::index_gl>(schedule_peer_token_idx),
        .tokens_per_expert = kittens::py::tensor_to_gl<globals::index_gl>(tokens_per_expert),
        .tokens_per_expert_and_peer =kittens::py::tensor_to_gl<globals::index_gl>(tokens_per_expert_and_peer),
        .rank = rank,
    };

    auto stream = at::cuda::getCurrentCUDAStream();
    count_kernel<<<(G.topk.numel() + config::NUM_THREADS - 1) / config::NUM_THREADS, config::NUM_THREADS, num_local_experts * world_size * sizeof(int), stream>>>(G);
    pad_kernel<<<num_local_experts, 1, 0, stream>>>(G);
    schedule_kernel<<<num_local_experts * world_size, config::NUM_THREADS, world_size * sizeof(int), stream>>>(G);
}

} // namespace scheduler

namespace dispatcher {

struct config {
    static constexpr int NUM_DEVICES = 4;

    static constexpr int MINIBATCH_SIZE = 4096;

    static constexpr int Mb = 64;
    static constexpr int Nb = 256;

    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 6;
    static constexpr int NUM_THREADS = 64;

    static_assert(MINIBATCH_SIZE % Mb == 0, "MINIBATCH_SIZE must be a multiple of Mb");
};

struct globals {
    using token_tile = st_bf<config::Mb, config::Nb, false>;
    using token_vec = sv_bf<config::Nb>;

    using send_buffer_pgl = pgl<gl<bf16, 1, 1, -1, -1, token_vec>, config::NUM_DEVICES, false>;
    using recv_buffer_gl = gl<bf16, 1, 1, -1, -1, token_vec>;
    using index_gl = gl<int, 1, 1, 1, -1>;

    send_buffer_pgl send_buffer;     // (num_local_tokens, H)
    recv_buffer_gl recv_buffer;      // (capacity, H)
    index_gl schedule_src_rank;      // (capacity,)
    index_gl schedule_src_token_idx; // (capacity,) original_token_idx * topk + k
    index_gl tokens_per_expert;      // (num_local_experts,)
    index_gl dispatch_counter;       // (num_minibatches,)

    int topk;

    __host__ inline dim3 grid() const {
        return dim3(recv_buffer.cols() / config::Nb, recv_buffer.rows() / config::Mb);
    }
    __host__ inline int dynamic_shared_memory() const {
        return static_cast<int>(sizeof(token_tile) + 1024);
    }
};

__device__ inline void dispatch_kernel(const globals &G) {
    const int tid = threadIdx.x;
    const int col_block_idx = blockIdx.x;
    const int row_block_idx = blockIdx.y;

    const int dst_row_idx = row_block_idx * config::Mb + tid;
    const int src_rank = G.schedule_src_rank[{dst_row_idx}];
    const int src_row_idx = G.schedule_src_token_idx[{dst_row_idx}] / G.topk;
    const int minibatch_idx = dst_row_idx / config::MINIBATCH_SIZE;

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    globals::token_tile &token_tile = allocator.allocate<globals::token_tile>();
    globals::token_vec &token_vec = *reinterpret_cast<globals::token_vec*>(&token_tile.data[tid * config::Nb]);

    __shared__ semaphore inputs_arrived;

    int num_tokens = 0;
    for (int expert_idx = 0; expert_idx < G.tokens_per_expert.cols(); ++expert_idx)
        num_tokens += G.tokens_per_expert[{expert_idx}];
    if (row_block_idx * config::Mb >= num_tokens) return;

    const int num_valid = __syncthreads_count(src_rank >= 0);
    if (num_valid == 0) {
        if (tid == 0)
            asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}" :: "l"(&G.dispatch_counter[{minibatch_idx}]), "r"(1) : "memory");
        return; // whole recv block is padding
    }

    if (tid == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        tma::expect_bytes(inputs_arrived, num_valid * sizeof(globals::token_vec));
    }
    __syncthreads();

    if (src_rank >= 0)
        tma::load_async(token_vec, G.send_buffer[src_rank], {src_row_idx, col_block_idx}, inputs_arrived);
    wait(inputs_arrived, 0);
    if (src_rank >= 0) {
        tma::store_async(G.recv_buffer, token_vec, {dst_row_idx, col_block_idx});
        tma::store_async_wait();
    }
    __syncthreads();
    if (tid == 0)
        asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}" :: "l"(&G.dispatch_counter[{minibatch_idx}]), "r"(1) : "memory");
}

__host__ void dispatch(
    const at::Tensor &send_buffer,
    const std::vector<int64_t> &send_buffer_ptrs,
    const at::Tensor &recv_buffer,
    const at::Tensor &schedule_src_rank,
    const at::Tensor &schedule_src_token_idx,
    const at::Tensor &tokens_per_expert,
    at::Tensor dispatch_counter, // must be zero'ed
    int topk
) {
    bf16 *send_buffer_data[config::NUM_DEVICES];
    for (int i = 0; i < config::NUM_DEVICES; ++i)
        send_buffer_data[i] = reinterpret_cast<bf16*>(send_buffer_ptrs[i]);

    globals G {
        .send_buffer = globals::send_buffer_pgl{send_buffer_data, nullptr, nullptr, static_cast<size_t>(send_buffer.size(0)), static_cast<size_t>(send_buffer.size(1))},
        .recv_buffer = kittens::py::tensor_to_gl<globals::recv_buffer_gl>(recv_buffer),
        .schedule_src_rank = kittens::py::tensor_to_gl<globals::index_gl>(schedule_src_rank),
        .schedule_src_token_idx = kittens::py::tensor_to_gl<globals::index_gl>(schedule_src_token_idx),
        .tokens_per_expert = kittens::py::tensor_to_gl<globals::index_gl>(tokens_per_expert),
        .dispatch_counter = kittens::py::tensor_to_gl<globals::index_gl>(dispatch_counter),
        .topk = topk,
    };

    kittens::py::launch_kernel<config, globals, dispatch_kernel>(G);
}

} // namespace dispatcher

namespace combiner {

struct config {
    static constexpr int NUM_DEVICES = 4;

    static constexpr int MINIBATCH_SIZE = 4096;

    static constexpr int Mb = 64;
    static constexpr int Nb = 256;

    static constexpr int MLP_Mb = 256;
    static constexpr int MLP_Nb = 256;
    static constexpr int MLP_CLUSTER_SIZE = 2;

    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 6;
    static constexpr int NUM_THREADS = 64;
};

struct globals {
    using token_tile = st_bf<config::Mb, config::Nb, false>;
    using token_vec = sv_bf<config::Nb>;

    using send_buffer_gl = gl<bf16, 1, 1, -1, -1, token_vec>;
    using recv_buffer_pgl = pgl<gl<bf16, 1, 1, -1, -1, token_vec>, config::NUM_DEVICES, false>;
    using index_gl = gl<int, 1, 1, 1, -1>;

    send_buffer_gl send_buffer;      // (capacity, H)
    recv_buffer_pgl recv_buffer;     // (num_local_tokens * topk, H)
    index_gl schedule_dst_rank;      // (capacity,)
    index_gl schedule_dst_token_idx; // (capacity,)
    index_gl tokens_per_expert;      // (num_local_experts,)
    index_gl combine_counter;        // (num_minibatches,)

    __host__ inline dim3 grid() const {
        return dim3(send_buffer.cols() / config::Nb, send_buffer.rows() / config::Mb);
    }
    __host__ inline int dynamic_shared_memory() const {
        return static_cast<int>(sizeof(token_tile) + 1024);
    }
};

__device__ inline void combine_kernel(const globals &G) {
    const int tid = threadIdx.x;
    const int col_block_idx = blockIdx.x;
    const int row_block_idx = blockIdx.y;

    const int src_row_idx = row_block_idx * config::Mb + tid;
    const int dst_rank = G.schedule_dst_rank[{src_row_idx}];
    const int dst_row_idx = G.schedule_dst_token_idx[{src_row_idx}];
    const int minibatch_idx = src_row_idx / config::MINIBATCH_SIZE;

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    globals::token_tile &token_tile = allocator.allocate<globals::token_tile>();
    globals::token_vec &token_vec = *reinterpret_cast<globals::token_vec*>(&token_tile.data[tid * config::Nb]);

    __shared__ semaphore inputs_arrived;

    int num_tokens = 0;
    for (int expert_idx = 0; expert_idx < G.tokens_per_expert.cols(); ++expert_idx)
        num_tokens += G.tokens_per_expert[{expert_idx}];
    if (row_block_idx * config::Mb >= num_tokens) return;

    const int num_valid = __syncthreads_count(dst_rank >= 0);
    if (num_valid == 0) return; // whole send block is padding

    if (tid == 0) {
        const int minibatch_first_row = minibatch_idx * config::MINIBATCH_SIZE;
        const int minibatch_rows = max(0, min(config::MINIBATCH_SIZE, num_tokens - minibatch_first_row));
        const int expected = ((minibatch_rows + config::MLP_Mb - 1) / config::MLP_Mb) * (G.send_buffer.cols() / config::MLP_Nb) * config::MLP_CLUSTER_SIZE;
        int combine_counter;
        while (true) {
            asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}" : "=r"(combine_counter) : "l"(&G.combine_counter[{minibatch_idx}]) : "memory");
            if (combine_counter >= expected) break;
            __nanosleep(16);
        }
        asm volatile("{fence.acquire.gpu;}" ::: "memory");

        init_semaphore(inputs_arrived, 0, 1);
        tma::expect_bytes(inputs_arrived, num_valid * sizeof(globals::token_vec));
    }
    __syncthreads();

    if (dst_rank >= 0)
        tma::load_async(token_vec, G.send_buffer, {src_row_idx, col_block_idx}, inputs_arrived);
    wait(inputs_arrived, 0);
    if (dst_rank >= 0)
        tma::store_async(G.recv_buffer[dst_rank], token_vec, {dst_row_idx, col_block_idx});
}

__host__ void combine(
    const at::Tensor &send_buffer,
    const at::Tensor &recv_buffer,
    const std::vector<int64_t> &recv_buffer_ptrs,
    const at::Tensor &schedule_dst_rank,
    const at::Tensor &schedule_dst_token_idx,
    const at::Tensor &tokens_per_expert,
    const at::Tensor &combine_counter
) {
    bf16 *recv_buffer_data[config::NUM_DEVICES];
    for (int i = 0; i < config::NUM_DEVICES; ++i)
        recv_buffer_data[i] = reinterpret_cast<bf16*>(recv_buffer_ptrs[i]);

    globals G {
        .send_buffer = kittens::py::tensor_to_gl<globals::send_buffer_gl>(send_buffer),
        .recv_buffer = globals::recv_buffer_pgl{recv_buffer_data, nullptr, nullptr, static_cast<size_t>(recv_buffer.size(0)), static_cast<size_t>(recv_buffer.size(1))},
        .schedule_dst_rank = kittens::py::tensor_to_gl<globals::index_gl>(schedule_dst_rank),
        .schedule_dst_token_idx = kittens::py::tensor_to_gl<globals::index_gl>(schedule_dst_token_idx),
        .tokens_per_expert = kittens::py::tensor_to_gl<globals::index_gl>(tokens_per_expert),
        .combine_counter = kittens::py::tensor_to_gl<globals::index_gl>(combine_counter),
    };

    kittens::py::launch_kernel<config, globals, combine_kernel>(G);
}

} // namespace combiner

namespace mlp_swigluer {

struct config {
    static constexpr int MINIBATCH_SIZE = 4096;

    static constexpr int Mb = 256;
    static constexpr int Nb = 256;
    static constexpr int Kb = 64;
    static constexpr int SUPERGROUP_SIZE = 8;
    static constexpr int LOAD_PIPE_DEPTH = 5;
    static constexpr int EPI_PIPE_DEPTH = 4;

    static constexpr int SWIGLU_Mb = 128;
    static constexpr int SWIGLU_Nb = 128;
    static constexpr int SWIGLU_PIPE_DEPTH = 3;
    static constexpr int CLC_PIPE_DEPTH = 1;

    static constexpr int DISPATCH_Mb = 64;
    static constexpr int DISPATCH_Nb = 256;

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int NUM_CONSUMERS = 1;
    static constexpr int NUM_PRODUCERS = 1;
    static constexpr int NUM_WARPS = (NUM_CONSUMERS + NUM_PRODUCERS) * WARPGROUP_WARPS; // 8
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS; // 256

    static constexpr int NUM_D_TILES = EPI_PIPE_DEPTH > 1 ? 2 : 1;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1024;

    static_assert(MINIBATCH_SIZE % Mb == 0,        "MINIBATCH_SIZE must be a multiple of Mb");
    static_assert(MINIBATCH_SIZE % SWIGLU_Mb == 0, "MINIBATCH_SIZE must be a multiple of SWIGLU_Mb");
    static_assert(MINIBATCH_SIZE % DISPATCH_Mb == 0, "MINIBATCH_SIZE must be a multiple of DISPATCH_Mb");
};

template <typename C>
struct globals {
    using a_tile = st_bf<C::Mb / 2, C::Kb>;
    using b_tile = st_bf<C::Nb / 2, C::Kb>;
    using d_tile = st_bf<C::Mb / 2, C::Nb / C::EPI_PIPE_DEPTH>;
    using swiglu_tile = st_bf<C::SWIGLU_Mb, C::SWIGLU_Nb>;

    using activation_gl = gl<bf16, 1, 1, -1, -1, a_tile, swiglu_tile>;
    using weight_gl = gl<bf16, 1, -1, -1, -1, b_tile>;
    using index_gl = gl<int, 1, 1, 1, -1>;

    activation_gl x_shared;
    activation_gl x_routed;
    activation_gl gate_shared;
    activation_gl gate_routed;
    activation_gl up_shared;
    activation_gl up_routed;
    activation_gl hidden_shared;
    activation_gl hidden_routed;
    activation_gl y_shared;
    activation_gl y_routed;
    weight_gl w_shared_gate;
    weight_gl w_routed_gate;
    weight_gl w_shared_up;
    weight_gl w_routed_up;
    weight_gl w_shared_down;
    weight_gl w_routed_down;
    index_gl tokens_per_expert;
    index_gl mlp_swiglu_counter;
    index_gl dispatch_counter;
    index_gl combine_counter;

    __host__ inline dim3 grid() const {
        const int num_minibatches = (x_routed.rows() + C::MINIBATCH_SIZE - 1) / C::MINIBATCH_SIZE;
        const int shared_row_blocks = x_shared.rows() / C::Mb;
        const int minibatch_routed_row_blocks = C::MINIBATCH_SIZE / C::Mb;
        const int shared_gate_up_tasks = shared_row_blocks * (w_shared_gate.rows() / C::Nb);
        const int minibatch_routed_gate_up_tasks = minibatch_routed_row_blocks * (w_routed_gate.rows() / C::Nb);
        const int shared_swiglu_tiles = (hidden_shared.rows() / C::SWIGLU_Mb) * (hidden_shared.cols() / C::SWIGLU_Nb);
        const int minibatch_routed_swiglu_tiles = (C::MINIBATCH_SIZE / C::SWIGLU_Mb) * (hidden_routed.cols() / C::SWIGLU_Nb);
        const int shared_swiglu_tasks = (shared_swiglu_tiles + C::CLUSTER_SIZE * C::SWIGLU_PIPE_DEPTH - 1) / (C::CLUSTER_SIZE * C::SWIGLU_PIPE_DEPTH);
        const int minibatch_routed_swiglu_tasks = (minibatch_routed_swiglu_tiles + C::CLUSTER_SIZE * C::SWIGLU_PIPE_DEPTH - 1) / (C::CLUSTER_SIZE * C::SWIGLU_PIPE_DEPTH);
        const int shared_down_tasks = shared_row_blocks * (w_shared_down.rows() / C::Nb);
        const int minibatch_routed_down_tasks = minibatch_routed_row_blocks * (w_routed_down.rows() / C::Nb);
        const int shared_tasks = 2 * shared_gate_up_tasks + shared_swiglu_tasks + shared_down_tasks;
        const int minibatch_tasks = 2 * minibatch_routed_gate_up_tasks + minibatch_routed_swiglu_tasks + minibatch_routed_down_tasks;
        return dim3(C::CLUSTER_SIZE * (shared_tasks + num_minibatches * minibatch_tasks));
    }
};

template <typename C, bool IS_SHARED>
__device__ __forceinline__ void swiglu(
    const globals<C> &g,
    semaphore (&swiglu_inputs_arrived)[C::SWIGLU_PIPE_DEPTH],
    uint32_t &swiglu_bitfield,
    int cta_rank,
    int minibatch_idx,
    int task_idx,
    uint64_t smem_base_addr
) {
    using G = globals<C>;

    typename G::swiglu_tile (&a_smem)[C::SWIGLU_PIPE_DEPTH] = *reinterpret_cast<typename G::swiglu_tile (*)[C::SWIGLU_PIPE_DEPTH]>(smem_base_addr);
    typename G::swiglu_tile (&b_smem)[C::SWIGLU_PIPE_DEPTH] = *reinterpret_cast<typename G::swiglu_tile (*)[C::SWIGLU_PIPE_DEPTH]>(smem_base_addr + sizeof(a_smem));

    const typename G::activation_gl &x_gmem      = IS_SHARED ? g.x_shared : g.x_routed;
    const typename G::activation_gl &gate_gmem   = IS_SHARED ? g.gate_shared : g.gate_routed;
    const typename G::activation_gl &up_gmem     = IS_SHARED ? g.up_shared : g.up_routed;
    const typename G::activation_gl &hidden_gmem = IS_SHARED ? g.hidden_shared : g.hidden_routed;
    const typename G::weight_gl     &w_gate_gmem = IS_SHARED ? g.w_shared_gate : g.w_routed_gate;

    const int shared_row_blocks = g.x_shared.rows() / C::Mb;
    const int routed_row_blocks = g.x_routed.rows() / C::Mb;
    const int shared_gate_up_tasks = shared_row_blocks * (g.w_shared_gate.rows() / C::Nb);
    const int routed_gate_up_tasks = routed_row_blocks * (g.w_routed_gate.rows() / C::Nb);
    const int gate_counter_offset = IS_SHARED ? 0 : shared_gate_up_tasks;
    const int swiglu_counter_offset = shared_gate_up_tasks + routed_gate_up_tasks + (IS_SHARED ? 0 : shared_row_blocks);
    const int intermediate_col_blocks = w_gate_gmem.rows() / C::Nb;

    int num_tokens;
    if constexpr (IS_SHARED) {
        num_tokens = x_gmem.rows();
    } else {
        num_tokens = 0;
        for (int expert_idx = 0; expert_idx < w_gate_gmem.depth(); ++expert_idx)
            num_tokens += g.tokens_per_expert[{expert_idx}];
    }
    const int row_blocks = num_tokens / C::SWIGLU_Mb;
    const int col_blocks = hidden_gmem.cols() / C::SWIGLU_Nb;
    const int num_tiles = row_blocks * col_blocks;
    int first_tile_idx, tile_end;
    if constexpr (IS_SHARED) {
        first_tile_idx = task_idx * C::CLUSTER_SIZE * C::SWIGLU_PIPE_DEPTH + cta_rank * C::SWIGLU_PIPE_DEPTH;
        tile_end = num_tiles;
    } else {
        const int num_tiles_per_minibatch = (C::MINIBATCH_SIZE / C::SWIGLU_Mb) * col_blocks;
        const int minibatch_first_tile_idx = minibatch_idx * num_tiles_per_minibatch;
        first_tile_idx = minibatch_first_tile_idx + task_idx * C::CLUSTER_SIZE * C::SWIGLU_PIPE_DEPTH + cta_rank * C::SWIGLU_PIPE_DEPTH;
        tile_end = min(num_tiles, minibatch_first_tile_idx + num_tiles_per_minibatch);
    }
    if (first_tile_idx >= tile_end)
        return;

    int first_row, first_col;
    if (threadIdx.x == 0) {
        first_row = first_tile_idx / col_blocks;
        first_col = first_tile_idx % col_blocks;
        #pragma unroll
        for (int stage = 0; stage < C::SWIGLU_PIPE_DEPTH; ++stage) {
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

                const int parent_task_idx = (row / (C::Mb / C::SWIGLU_Mb)) * intermediate_col_blocks + col / (C::Nb / C::SWIGLU_Nb);
                while (true) {
                    int gate_up_counter;
                    asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}" : "=r"(gate_up_counter) : "l"(&g.mlp_swiglu_counter[{gate_counter_offset + parent_task_idx}]) : "memory");
                    if (gate_up_counter >= 2 * C::CLUSTER_SIZE) break;
                    __nanosleep(16);
                }
                asm volatile("{fence.acquire.gpu;}" ::: "memory");

                tma::load_async(a_smem[stage], gate_gmem, {row, col}, swiglu_inputs_arrived[stage]);
                tma::load_async(b_smem[stage], up_gmem,   {row, col}, swiglu_inputs_arrived[stage]);
            }
        }
    }

    using compute_group = group<C::NUM_WARPS>;
    #pragma unroll
    for (int stage = 0; stage < C::SWIGLU_PIPE_DEPTH; ++stage) {
        const int tile_idx = first_tile_idx + stage;
        if (tile_idx < tile_end) {
            rt_fl<C::SWIGLU_Mb / C::NUM_WARPS, C::SWIGLU_Nb> gate, up, denominator;
            wait(swiglu_inputs_arrived[stage], get_phasebit<0>(swiglu_bitfield, stage));
            update_phasebit<0>(swiglu_bitfield, stage);
            compute_group::load(gate, a_smem[stage]);
            compute_group::load(up, b_smem[stage]);
            compute_group::mul(denominator, gate, -1.4426950408889634f);
            compute_group::exp2(denominator, denominator);
            compute_group::add(denominator, denominator, 1.0f);
            compute_group::div(gate, gate, denominator);
            compute_group::mul(gate, gate, up);
            compute_group::store(a_smem[stage], gate);
            __syncthreads();
            if (threadIdx.x == 0) {
                // This improves throughput
                int row = first_row;
                int col = first_col + stage;
                if (col >= col_blocks) {
                    ++row;
                    col -= col_blocks;
                }
                tma::store_async(hidden_gmem, a_smem[stage], {row, col});
            }
        }
    }

    if (threadIdx.x == 0) {
        tma::store_async_wait();
        #pragma unroll
        for (int stage = 0; stage < C::SWIGLU_PIPE_DEPTH; ++stage) {
            const int tile_idx = first_tile_idx + stage;
            if (tile_idx < tile_end) {
                // This improves throughput
                int row = first_row;
                int col = first_col + stage;
                if (col >= col_blocks)
                    ++row;
                asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}" :: "l"(&g.mlp_swiglu_counter[{swiglu_counter_offset + row / (C::Mb / C::SWIGLU_Mb)}]), "r"(1) : "memory");
            }
        }
    }
}

enum class expert_gemm_kind { GATE, UP, DOWN };

template <typename C, bool IS_SHARED>
__device__ __forceinline__ void expert_grouped_gemm(
    const globals<C> &g,
    semaphore (&gemm_inputs_arrived)[C::LOAD_PIPE_DEPTH],
    semaphore (&gemm_inputs_finished)[C::LOAD_PIPE_DEPTH],
    semaphore &gemm_outputs_arrived,
    semaphore &gemm_outputs_finished,
    uint32_t &gemm_bitfield,
    int cta_rank,
    int minibatch_idx,
    int task_idx,
    uint64_t smem_base_addr,
    expert_gemm_kind kind,
    tt<float, C::Mb / 2, C::Nb> d_tt
) {
    using G = globals<C>;

    typename G::a_tile (&a_smem)[C::LOAD_PIPE_DEPTH] = *reinterpret_cast<typename G::a_tile (*)[C::LOAD_PIPE_DEPTH]>(smem_base_addr);
    typename G::b_tile (&b_smem)[C::LOAD_PIPE_DEPTH] = *reinterpret_cast<typename G::b_tile (*)[C::LOAD_PIPE_DEPTH]>(smem_base_addr + sizeof(a_smem));
    typename G::d_tile (&d_smem)[C::NUM_D_TILES] = *reinterpret_cast<typename G::d_tile (*)[C::NUM_D_TILES]>(smem_base_addr + sizeof(a_smem) + sizeof(b_smem));

    const typename G::activation_gl &x_gmem      = IS_SHARED ? g.x_shared : g.x_routed;
    const typename G::activation_gl &gate_gmem   = IS_SHARED ? g.gate_shared : g.gate_routed;
    const typename G::activation_gl &up_gmem     = IS_SHARED ? g.up_shared : g.up_routed;
    const typename G::activation_gl &hidden_gmem = IS_SHARED ? g.hidden_shared : g.hidden_routed;
    const typename G::activation_gl &y_gmem      = IS_SHARED ? g.y_shared : g.y_routed;
    const typename G::weight_gl     &w_gate_gmem = IS_SHARED ? g.w_shared_gate : g.w_routed_gate;
    const typename G::weight_gl     &w_up_gmem   = IS_SHARED ? g.w_shared_up : g.w_routed_up;
    const typename G::weight_gl     &w_down_gmem = IS_SHARED ? g.w_shared_down : g.w_routed_down;

    const typename G::activation_gl &a_gmem = kind == expert_gemm_kind::DOWN ? hidden_gmem : x_gmem;
    const typename G::weight_gl     &b_gmem = kind == expert_gemm_kind::GATE ? w_gate_gmem : (kind == expert_gemm_kind::UP ? w_up_gmem : w_down_gmem);
    const typename G::activation_gl &d_gmem = kind == expert_gemm_kind::GATE ? gate_gmem   : (kind == expert_gemm_kind::UP ? up_gmem : y_gmem);

    const int iters_per_task = a_gmem.cols() / C::Kb;
    const int col_blocks     = b_gmem.rows() / C::Nb;
    const int shared_row_blocks = g.x_shared.rows() / C::Mb;
    const int routed_row_blocks = g.x_routed.rows() / C::Mb;
    const int shared_gate_up_tasks = shared_row_blocks * (g.w_shared_gate.rows() / C::Nb);
    const int routed_gate_up_tasks = routed_row_blocks * (g.w_routed_gate.rows() / C::Nb);
    const int gate_counter_offset = IS_SHARED ? 0 : shared_gate_up_tasks;
    const int swiglu_counter_offset = shared_gate_up_tasks + routed_gate_up_tasks + (IS_SHARED ? 0 : shared_row_blocks);

    int3 tile_coord = {-1, -1, -1};
    if constexpr (IS_SHARED) {
        const int row_blocks = x_gmem.rows() / C::Mb;
        const int num_tasks = row_blocks * col_blocks;
        if (task_idx < num_tasks) {
            const int2 swizzled = get_swizzled_2d_idx<C::SUPERGROUP_SIZE>(row_blocks, col_blocks, task_idx);
            tile_coord = {swizzled.x, swizzled.y, 0};
        }
    } else {
        constexpr int minibatch_routed_row_blocks = C::MINIBATCH_SIZE / C::Mb;
        const int minibatch_routed_row_offset = minibatch_idx * minibatch_routed_row_blocks;
        int row_block_offset = 0;
        for (int expert_idx = 0; expert_idx < b_gmem.depth(); ++expert_idx) {
            const int expert_row_blocks = g.tokens_per_expert[{expert_idx}] / C::Mb;
            const int first_row_block = max(minibatch_routed_row_offset, row_block_offset);
            const int row_blocks = max(0, min(minibatch_routed_row_offset + minibatch_routed_row_blocks, row_block_offset + expert_row_blocks) - first_row_block);
            const int num_tasks = row_blocks * col_blocks;
            if (task_idx < num_tasks) {
                const int2 swizzled = get_swizzled_2d_idx<C::SUPERGROUP_SIZE>(row_blocks, col_blocks, task_idx);
                tile_coord = {first_row_block + swizzled.x, swizzled.y, expert_idx};
                break;
            }
            task_idx -= num_tasks;
            row_block_offset += expert_row_blocks;
        }
    }
    if (tile_coord.z < 0) return;

    if (warpgroup::groupid() == C::NUM_CONSUMERS) {
        if (warpgroup::warpid() == 3 && warp::elect_leader()) {
            if (kind == expert_gemm_kind::DOWN) {
                const int expected = (C::Mb / C::SWIGLU_Mb) * (hidden_gmem.cols() / C::SWIGLU_Nb);
                int swiglu_counter;
                while (true) {
                    asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}" : "=r"(swiglu_counter) : "l"(&g.mlp_swiglu_counter[{swiglu_counter_offset + tile_coord.x}]) : "memory");
                    if (swiglu_counter >= expected) break;
                    __nanosleep(16);
                }
                asm volatile("{fence.acquire.gpu;}" ::: "memory");
            } else if constexpr (!IS_SHARED) {
                int num_tokens = 0;
                for (int expert_idx = 0; expert_idx < g.w_routed_gate.depth(); ++expert_idx)
                    num_tokens += g.tokens_per_expert[{expert_idx}];
                const int minibatch_first_row = minibatch_idx * C::MINIBATCH_SIZE;
                const int minibatch_rows = max(0, min(C::MINIBATCH_SIZE, num_tokens - minibatch_first_row));
                const int expected = ((minibatch_rows + C::DISPATCH_Mb - 1) / C::DISPATCH_Mb) * (g.x_routed.cols() / C::DISPATCH_Nb);
                int dispatch_counter;
                while (true) {
                    asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}" : "=r"(dispatch_counter) : "l"(&g.dispatch_counter[{minibatch_idx}]) : "memory");
                    if (dispatch_counter >= expected) break;
                    __nanosleep(16);
                }
                asm volatile("{fence.acquire.gpu;}" ::: "memory");
            }
            int input_ring = 0;
            for (int idx = 0; idx < iters_per_task; ++idx) {
                wait(gemm_inputs_finished[input_ring], get_phasebit<1>(gemm_bitfield, input_ring));
                tma::cluster::load_async(a_smem[input_ring], a_gmem, {tile_coord.x * 2 + cta_rank, idx},               gemm_inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                tma::cluster::load_async(b_smem[input_ring], b_gmem, {tile_coord.z, tile_coord.y * 2 + cta_rank, idx}, gemm_inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                update_phasebit<1>(gemm_bitfield, input_ring);
                input_ring = ring_advance<C::LOAD_PIPE_DEPTH>(input_ring);
            }
        } else if (cta_rank == 0 && warpgroup::warpid() == 0 && warp::elect_leader()) {
            int input_ring = 0;
            wait(gemm_outputs_finished, get_phasebit<1>(gemm_bitfield, C::LOAD_PIPE_DEPTH));
            update_phasebit<1>(gemm_bitfield, C::LOAD_PIPE_DEPTH);
            for (int idx = 0; idx < iters_per_task; ++idx) {
                tma::expect_bytes(gemm_inputs_arrived[input_ring], C::CLUSTER_SIZE * sizeof(typename G::a_tile) + 2 * sizeof(typename G::b_tile));
                wait(gemm_inputs_arrived[input_ring], get_phasebit<0>(gemm_bitfield, input_ring));
                if (idx == 0) mm2_ABt (d_tt, a_smem[input_ring], b_smem[input_ring], gemm_inputs_finished[input_ring]);
                else          mma2_ABt(d_tt, a_smem[input_ring], b_smem[input_ring], gemm_inputs_finished[input_ring]);
                update_phasebit<0>(gemm_bitfield, input_ring);
                input_ring = ring_advance<C::LOAD_PIPE_DEPTH>(input_ring);
            }
            detail::tcgen05::commit<C::CLUSTER_SIZE>(gemm_outputs_arrived);
        }
    } else {
        using epilogue_group = group<WARPGROUP_WARPS>;
        wait(gemm_outputs_arrived, get_phasebit<0>(gemm_bitfield, C::LOAD_PIPE_DEPTH));
        update_phasebit<0>(gemm_bitfield, C::LOAD_PIPE_DEPTH);
        rt_bf<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> d_reg[C::EPI_PIPE_DEPTH];
        #pragma unroll
        for (int i = 0; i < C::EPI_PIPE_DEPTH; ++i)
            warpgroup::load_async(d_reg[i], d_tt.template subtile<tt<float, C::Mb / 2, C::Nb / C::EPI_PIPE_DEPTH>>(0, C::Nb / C::EPI_PIPE_DEPTH * i));
        tensor_load_wait();
        warpgroup::sync(1);
        warpgroup::tma::cluster::arrive(gemm_outputs_finished, 0);
        #pragma unroll
        for (int i = 0; i < C::EPI_PIPE_DEPTH; ++i) {
            warpgroup::tma::store_async_read_wait<C::NUM_D_TILES - 1>();
            warpgroup::sync(1);
            warpgroup::store(d_smem[i % C::NUM_D_TILES], d_reg[i]);
            warpgroup::sync(1);
            warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(d_gmem, d_smem[i % C::NUM_D_TILES], {2 * tile_coord.x + cta_rank, C::EPI_PIPE_DEPTH * tile_coord.y + i});
        }
        epilogue_group::sync(4);
        if (epilogue_group::warpid() == 0 && warp::elect_leader()) {
            if (kind != expert_gemm_kind::DOWN) {
                // Up/gate is complete; signal Swiglu
                tma::store_async_wait();
                asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}" :: "l"(&g.mlp_swiglu_counter[{gate_counter_offset + tile_coord.x * col_blocks + tile_coord.y}]), "r"(1) : "memory");
            } else if constexpr (!IS_SHARED) {
                // Routed down is complete; signal combine
                tma::store_async_wait();
                asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}" :: "l"(&g.combine_counter[{minibatch_idx}]), "r"(1) : "memory");
            }
        }
    }
}

template <typename C>
__device__ __forceinline__ void mlp_swiglu_kernel(const globals<C> &g) {
    using G = globals<C>;

    warpgroup::increase_registers<256>();

    extern __shared__ int __shm[];
    const uint64_t smem_base_addr = (reinterpret_cast<uint64_t>(&__shm[0]) + 1023) & ~uint64_t(1023);

    int cluster_idx = clusterIdx().x;
    const int cta_rank = cluster_ctarank();
    const int shared_row_blocks = g.x_shared.rows() / C::Mb;
    const int minibatch_routed_row_blocks = C::MINIBATCH_SIZE / C::Mb;
    const int shared_gate_up_tasks = shared_row_blocks * (g.w_shared_gate.rows() / C::Nb);
    const int minibatch_routed_gate_up_tasks = minibatch_routed_row_blocks * (g.w_routed_gate.rows() / C::Nb);
    const int shared_swiglu_tiles = (g.hidden_shared.rows() / C::SWIGLU_Mb) * (g.hidden_shared.cols() / C::SWIGLU_Nb);
    const int minibatch_routed_swiglu_tiles = (C::MINIBATCH_SIZE / C::SWIGLU_Mb) * (g.hidden_routed.cols() / C::SWIGLU_Nb);
    const int shared_swiglu_tasks = (shared_swiglu_tiles + C::CLUSTER_SIZE * C::SWIGLU_PIPE_DEPTH - 1) / (C::CLUSTER_SIZE * C::SWIGLU_PIPE_DEPTH);
    const int minibatch_routed_swiglu_tasks = (minibatch_routed_swiglu_tiles + C::CLUSTER_SIZE * C::SWIGLU_PIPE_DEPTH - 1) / (C::CLUSTER_SIZE * C::SWIGLU_PIPE_DEPTH);
    const int shared_down_tasks = shared_row_blocks * (g.w_shared_down.rows() / C::Nb);
    const int minibatch_routed_down_tasks = minibatch_routed_row_blocks * (g.w_routed_down.rows() / C::Nb);
    const int shared_tasks = 2 * shared_gate_up_tasks + shared_swiglu_tasks + shared_down_tasks;
    const int minibatch_tasks = 2 * minibatch_routed_gate_up_tasks + minibatch_routed_swiglu_tasks + minibatch_routed_down_tasks;
    uint32_t gemm_bitfield = 0xFFFF0000;
    uint32_t swiglu_bitfield = 0xFFFF0000;

    __shared__ clc::handle clc_handle[C::CLC_PIPE_DEPTH];
    __shared__ semaphore schedule_arrived[C::CLC_PIPE_DEPTH], schedule_finished[C::CLC_PIPE_DEPTH];
    __shared__ semaphore swiglu_inputs_arrived[C::SWIGLU_PIPE_DEPTH];
    __shared__ semaphore gemm_inputs_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_outputs_arrived, gemm_outputs_finished;

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < C::SWIGLU_PIPE_DEPTH; ++i) {
            init_semaphore(swiglu_inputs_arrived[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(gemm_inputs_arrived[i], 0, 1);
            init_semaphore(gemm_inputs_finished[i], 0, 1);
        }
        init_semaphore(gemm_outputs_arrived, 0, 1);
        init_semaphore(gemm_outputs_finished, 0, C::CLUSTER_SIZE);
        #pragma unroll
        for (int i = 0; i < C::CLC_PIPE_DEPTH; ++i) {
            init_semaphore(schedule_arrived[i], 0, 1);
            init_semaphore(schedule_finished[i], 0, C::CLUSTER_SIZE * C::NUM_WARPS);
        }
    }

    tensor_allocator<1, C::CLUSTER_SIZE> tm_alloc{};
    tt<float, C::Mb / 2, C::Nb> d_tt = tm_alloc.template allocate<tt<float, C::Mb / 2, C::Nb>>(0);
    everyone::tma::cluster::sync();

    bool current_is_swiglu;
    for (int task_iter = 0; cluster_idx >= 0; ++task_iter) {
        const int clc_stage = task_iter % C::CLC_PIPE_DEPTH;
        if (warpgroup::groupid() == C::NUM_CONSUMERS && warpgroup::warpid() == 1 && warp::elect_leader()) { // warp not used by the gemms
            if (cta_rank == 0) {
                wait(schedule_finished[clc_stage], ((task_iter + C::CLC_PIPE_DEPTH) / C::CLC_PIPE_DEPTH) % 2);
                clc::schedule(clc_handle[clc_stage], schedule_arrived[clc_stage]);
            }
            tma::expect_bytes(schedule_arrived[clc_stage], sizeof(clc_handle[clc_stage]));
        }

        if (cluster_idx < shared_gate_up_tasks) {
            // Shared gate
            current_is_swiglu = false;
            const int task_idx = cluster_idx;
            expert_grouped_gemm<C, true>(g, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished,
                                            gemm_bitfield, cta_rank, 0, task_idx, smem_base_addr, expert_gemm_kind::GATE, d_tt);
        } else if (cluster_idx < shared_gate_up_tasks * 2) {
            // Shared up
            current_is_swiglu = false;
            const int task_idx = cluster_idx - shared_gate_up_tasks;
            expert_grouped_gemm<C, true>(g, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished,
                                            gemm_bitfield, cta_rank, 0, task_idx, smem_base_addr, expert_gemm_kind::UP, d_tt);
        } else if (cluster_idx < shared_gate_up_tasks * 2 + shared_swiglu_tasks) {
            // Shared Swiglu
            current_is_swiglu = true;
            const int task_idx = cluster_idx - shared_gate_up_tasks * 2;
            swiglu<C, true>(g, swiglu_inputs_arrived, swiglu_bitfield, cta_rank, 0, task_idx, smem_base_addr);
        } else if (cluster_idx < shared_tasks) {
            // Shared down
            current_is_swiglu = false;
            const int task_idx = cluster_idx - shared_gate_up_tasks * 2 - shared_swiglu_tasks;
            expert_grouped_gemm<C, true>(g, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished,
                                            gemm_bitfield, cta_rank, 0, task_idx, smem_base_addr, expert_gemm_kind::DOWN, d_tt);
        } else {
            // Routed expert with minibatching
            const int minibatch_idx = (cluster_idx - shared_tasks) / minibatch_tasks;
            const int minibatch_task_idx = (cluster_idx - shared_tasks) - minibatch_idx * minibatch_tasks;

            if (minibatch_task_idx < minibatch_routed_gate_up_tasks) {
                // Routed gate
                current_is_swiglu = false;
                const int task_idx = minibatch_task_idx;
                expert_grouped_gemm<C, false>(g, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished,
                                                gemm_bitfield, cta_rank, minibatch_idx, task_idx, smem_base_addr, expert_gemm_kind::GATE, d_tt);
            } else if (minibatch_task_idx < minibatch_routed_gate_up_tasks * 2) {
                // Routed up
                current_is_swiglu = false;
                const int task_idx = minibatch_task_idx - minibatch_routed_gate_up_tasks;
                expert_grouped_gemm<C, false>(g, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished,
                                                gemm_bitfield, cta_rank, minibatch_idx, task_idx, smem_base_addr, expert_gemm_kind::UP, d_tt);
            } else if (minibatch_task_idx < minibatch_routed_gate_up_tasks * 2 + minibatch_routed_swiglu_tasks) {
                // Routed Swiglu
                current_is_swiglu = true;
                const int task_idx = minibatch_task_idx - minibatch_routed_gate_up_tasks * 2;
                swiglu<C, false>(g, swiglu_inputs_arrived, swiglu_bitfield, cta_rank, minibatch_idx, task_idx, smem_base_addr);
            } else {
                // Routed down
                current_is_swiglu = false;
                const int task_idx = minibatch_task_idx - minibatch_routed_gate_up_tasks * 2 - minibatch_routed_swiglu_tasks;
                expert_grouped_gemm<C, false>(g, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished,
                                                gemm_bitfield, cta_rank, minibatch_idx, task_idx, smem_base_addr, expert_gemm_kind::DOWN, d_tt);
            }
        }

        wait(schedule_arrived[clc_stage], (task_iter / C::CLC_PIPE_DEPTH) % 2);
        const auto schedule = clc::query(clc_handle[clc_stage]);
        cluster_idx = schedule.success ? static_cast<int>(schedule.x / C::CLUSTER_SIZE) : -1;
        __syncwarp();
        warp::tma::cluster::arrive(schedule_finished[clc_stage], 0);

        // SWIGLU -> GEMM requires a cluster-wide sync
        const int next_minibatch_task_idx = (cluster_idx - shared_tasks) - ((cluster_idx - shared_tasks) / minibatch_tasks) * minibatch_tasks;
        const bool next_is_shared_swiglu = cluster_idx >= shared_gate_up_tasks * 2 && cluster_idx < shared_gate_up_tasks * 2 + shared_swiglu_tasks;
        const bool next_is_routed_swiglu = next_minibatch_task_idx >= minibatch_routed_gate_up_tasks * 2 && next_minibatch_task_idx < minibatch_routed_gate_up_tasks * 2 + minibatch_routed_swiglu_tasks;;
        if (current_is_swiglu && cluster_idx >= 0 && !next_is_shared_swiglu && !next_is_routed_swiglu)
            everyone::tma::cluster::sync();
    }
}

__host__ void mlp_swiglu(
    const at::Tensor &x_shared,
    const at::Tensor &x_routed,
    at::Tensor &gate_shared,
    at::Tensor &gate_routed,
    at::Tensor &up_shared,
    at::Tensor &up_routed,
    at::Tensor &hidden_shared,
    at::Tensor &hidden_routed,
    at::Tensor &y_shared,
    at::Tensor &y_routed,
    const at::Tensor &w_shared_gate,
    const at::Tensor &w_routed_gate,
    const at::Tensor &w_shared_up,
    const at::Tensor &w_routed_up,
    const at::Tensor &w_shared_down,
    const at::Tensor &w_routed_down,
    const at::Tensor &tokens_per_expert,
    const at::Tensor &dispatch_counter,
    at::Tensor mlp_swiglu_counter, // must be zero'ed
    at::Tensor combine_counter // must be zero'ed
) {
    using C = config;
    using G = globals<C>;

    G g {
        .x_shared = kittens::py::tensor_to_gl<G::activation_gl>(x_shared),
        .x_routed = kittens::py::tensor_to_gl<G::activation_gl>(x_routed),
        .gate_shared = kittens::py::tensor_to_gl<G::activation_gl>(gate_shared),
        .gate_routed = kittens::py::tensor_to_gl<G::activation_gl>(gate_routed),
        .up_shared = kittens::py::tensor_to_gl<G::activation_gl>(up_shared),
        .up_routed = kittens::py::tensor_to_gl<G::activation_gl>(up_routed),
        .hidden_shared = kittens::py::tensor_to_gl<G::activation_gl>(hidden_shared),
        .hidden_routed = kittens::py::tensor_to_gl<G::activation_gl>(hidden_routed),
        .y_shared = kittens::py::tensor_to_gl<G::activation_gl>(y_shared),
        .y_routed = kittens::py::tensor_to_gl<G::activation_gl>(y_routed),
        .w_shared_gate = kittens::py::tensor_to_gl<G::weight_gl>(w_shared_gate),
        .w_routed_gate = kittens::py::tensor_to_gl<G::weight_gl>(w_routed_gate),
        .w_shared_up = kittens::py::tensor_to_gl<G::weight_gl>(w_shared_up),
        .w_routed_up = kittens::py::tensor_to_gl<G::weight_gl>(w_routed_up),
        .w_shared_down = kittens::py::tensor_to_gl<G::weight_gl>(w_shared_down),
        .w_routed_down = kittens::py::tensor_to_gl<G::weight_gl>(w_routed_down),
        .tokens_per_expert = kittens::py::tensor_to_gl<G::index_gl>(tokens_per_expert),
        .mlp_swiglu_counter = kittens::py::tensor_to_gl<G::index_gl>(mlp_swiglu_counter),
        .dispatch_counter = kittens::py::tensor_to_gl<G::index_gl>(dispatch_counter),
        .combine_counter = kittens::py::tensor_to_gl<G::index_gl>(combine_counter)
    };

    kittens::py::launch_kernel<C, G, mlp_swiglu_kernel<C>>(g);
}

} // namespace mlp_swigluer

__host__ std::tuple<at::Tensor, at::Tensor, at::Tensor> schedule(
    const at::Tensor &topk_all,
    const int num_local_experts,
    const int capacity,
    const int rank
) {
    const int world_size = static_cast<int>(topk_all.size(0));

    at::Tensor schedule_peer_rank = at::empty({capacity}, topk_all.options().dtype(at::kInt));
    at::Tensor schedule_peer_token_idx = at::empty({capacity}, topk_all.options().dtype(at::kInt));
    at::Tensor tokens_per_expert = at::empty({num_local_experts}, topk_all.options().dtype(at::kInt));
    at::Tensor tokens_per_expert_and_peer = at::zeros({num_local_experts * world_size}, topk_all.options().dtype(at::kInt));
    schedule_peer_rank.fill_(-1);

    scheduler::schedule(topk_all, schedule_peer_rank, schedule_peer_token_idx, tokens_per_expert, tokens_per_expert_and_peer, rank);
    
    return {schedule_peer_rank, schedule_peer_token_idx, tokens_per_expert};
}

__host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> dispatch_mlp_swiglu_combine(
    // Inputs and communication buffers
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,
    const at::Tensor &dispatch_buffer,
    const at::Tensor &combine_buffer,
    const std::vector<int64_t> &combine_buffer_ptrs,

    // Weights
    const at::Tensor &w_shared_gate,
    const at::Tensor &w_routed_gate,
    const at::Tensor &w_shared_up,
    const at::Tensor &w_routed_up,
    const at::Tensor &w_shared_down,
    const at::Tensor &w_routed_down,

    // Dispatch/combine schedule
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &tokens_per_expert,

    // Metadata
    int topk
) {

    const int num_local_tokens = x.size(0);
    const int buffer_capacity = dispatch_buffer.size(0);
    const int num_minibatches = (buffer_capacity + mlp_swigluer::config::MINIBATCH_SIZE - 1) / mlp_swigluer::config::MINIBATCH_SIZE;
    const int shared_row_blocks = num_local_tokens / mlp_swigluer::config::Mb;
    const int routed_row_blocks = buffer_capacity / mlp_swigluer::config::Mb;
    const int shared_gate_up_tasks = shared_row_blocks * (w_shared_gate.size(0) / mlp_swigluer::config::Nb);
    const int routed_gate_up_tasks = routed_row_blocks * (w_routed_gate.size(1) / mlp_swigluer::config::Nb);
    
    at::Tensor gate_shared = at::empty({x.size(0), w_shared_gate.size(0)}, x.options());
    at::Tensor gate_routed = at::empty({dispatch_buffer.size(0), w_routed_gate.size(1)}, dispatch_buffer.options());
    at::Tensor up_shared = at::empty({x.size(0), w_shared_up.size(0)}, x.options());
    at::Tensor up_routed = at::empty({dispatch_buffer.size(0), w_routed_up.size(1)}, dispatch_buffer.options());
    at::Tensor hidden_shared = at::empty({x.size(0), w_shared_gate.size(0)}, x.options());
    at::Tensor hidden_routed = at::empty({dispatch_buffer.size(0), w_routed_gate.size(1)}, dispatch_buffer.options());
    at::Tensor y_shared = at::empty_like(x);
    at::Tensor y_routed = at::empty_like(dispatch_buffer);
    at::Tensor dispatch_counter = at::zeros({num_minibatches}, tokens_per_expert.options());
    at::Tensor mlp_swiglu_counter = at::zeros({shared_gate_up_tasks + routed_gate_up_tasks + shared_row_blocks + routed_row_blocks}, tokens_per_expert.options());
    at::Tensor combine_counter = at::zeros({num_minibatches}, tokens_per_expert.options());

    dispatcher::dispatch(x, x_ptrs, dispatch_buffer, schedule_peer_rank, schedule_peer_token_idx, tokens_per_expert, dispatch_counter, topk);
    mlp_swigluer::mlp_swiglu(x, dispatch_buffer, gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, y_shared, y_routed, w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down, tokens_per_expert, dispatch_counter, mlp_swiglu_counter, combine_counter);
    combiner::combine(y_routed, combine_buffer, combine_buffer_ptrs, schedule_peer_rank, schedule_peer_token_idx, tokens_per_expert, combine_counter);

    return {gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, y_shared, y_routed, combine_buffer};
}

PYBIND11_MODULE(_C, m) {
    m.def("schedule", &schedule, "",
          pybind11::arg("topk_all"), pybind11::arg("num_local_experts"), pybind11::arg("capacity"), pybind11::arg("rank"));
    m.def("dispatch", &dispatcher::dispatch, "",
          pybind11::arg("send_buffer"), pybind11::arg("send_buffer_ptrs"), pybind11::arg("recv_buffer"),
          pybind11::arg("schedule_src_rank"), pybind11::arg("schedule_src_token_idx"),
          pybind11::arg("tokens_per_expert"), pybind11::arg("dispatch_counter"), pybind11::arg("topk"));
    m.def("combine", &combiner::combine, "",
          pybind11::arg("send_buffer"), pybind11::arg("recv_buffer"), pybind11::arg("recv_buffer_ptrs"),
          pybind11::arg("schedule_dst_rank"), pybind11::arg("schedule_dst_token_idx"),
          pybind11::arg("tokens_per_expert"), pybind11::arg("combine_counter"));
    m.def("mlp_swiglu", &mlp_swigluer::mlp_swiglu, "",
          pybind11::arg("x_shared"), pybind11::arg("x_routed"),
          pybind11::arg("gate_shared"), pybind11::arg("gate_routed"),
          pybind11::arg("up_shared"), pybind11::arg("up_routed"),
          pybind11::arg("hidden_shared"), pybind11::arg("hidden_routed"),
          pybind11::arg("y_shared"), pybind11::arg("y_routed"),
          pybind11::arg("w_shared_gate"), pybind11::arg("w_routed_gate"),
          pybind11::arg("w_shared_up"), pybind11::arg("w_routed_up"),
          pybind11::arg("w_shared_down"), pybind11::arg("w_routed_down"),
          pybind11::arg("tokens_per_expert"),
          pybind11::arg("dispatch_counter"), pybind11::arg("mlp_swiglu_counter"), pybind11::arg("combine_counter"));
    m.def("dispatch_mlp_swiglu_combine", &dispatch_mlp_swiglu_combine, "",
          pybind11::arg("x"), pybind11::arg("x_ptrs"),
          pybind11::arg("dispatch_buffer"), pybind11::arg("combine_buffer"), pybind11::arg("combine_buffer_ptrs"),
          pybind11::arg("w_shared_gate"), pybind11::arg("w_routed_gate"),
          pybind11::arg("w_shared_up"), pybind11::arg("w_routed_up"),
          pybind11::arg("w_shared_down"), pybind11::arg("w_routed_down"),
          pybind11::arg("schedule_peer_rank"), pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("tokens_per_expert"), pybind11::arg("topk"));
}
