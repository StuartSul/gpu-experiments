#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

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

void schedule(
    const at::Tensor &topk_all,
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &tokens_per_expert,
    const int rank
) {
    const int world_size = static_cast<int>(topk_all.size(0));
    const int num_local_experts = static_cast<int>(tokens_per_expert.size(0));
    at::Tensor tokens_per_expert_and_peer = at::zeros({num_local_experts * world_size}, topk_all.options().dtype(at::kInt));
    schedule_peer_rank.fill_(-1);

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
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 6;
    static constexpr int NUM_THREADS = 128;
};

struct globals {
    static constexpr int NUM_DEVICES = 4;
    static constexpr int ROW_BLOCK_SIZE = 128;
    static constexpr int COL_BLOCK_SIZE = 128;

    using token_tile = st_bf<ROW_BLOCK_SIZE, COL_BLOCK_SIZE, false>;
    using token_vec = sv_bf<COL_BLOCK_SIZE>;

    using send_buffer_pgl = pgl<gl<bf16, 1, 1, -1, -1, token_vec>, NUM_DEVICES, false>;
    using recv_buffer_gl = gl<bf16, 1, 1, -1, -1, token_vec>;
    using index_gl = gl<int, 1, 1, 1, -1>;

    send_buffer_pgl send_buffer;     // (num_local_tokens, H)
    recv_buffer_gl recv_buffer;      // (capacity, H)
    index_gl schedule_src_rank;      // (capacity,)
    index_gl schedule_src_token_idx; // (capacity,) original_token_idx * topk + k

    int topk;

    __host__ inline dim3 grid() const {
        return dim3(recv_buffer.cols() / COL_BLOCK_SIZE, recv_buffer.rows() / ROW_BLOCK_SIZE);
    }
    __host__ inline int dynamic_shared_memory() const {
        return static_cast<int>(sizeof(token_tile) + 1024);
    }
};

__device__ inline void dispatch_kernel(const globals &G) {
    const int tid = threadIdx.x;
    const int col_block_idx = blockIdx.x;
    const int row_block_idx = blockIdx.y;

    const int dst_row_idx = row_block_idx * globals::ROW_BLOCK_SIZE + tid;
    const int src_rank = G.schedule_src_rank[{dst_row_idx}];
    const int src_row_idx = G.schedule_src_token_idx[{dst_row_idx}] / G.topk;

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    globals::token_tile &token_tile = allocator.allocate<globals::token_tile>();
    globals::token_vec &token_vec = *reinterpret_cast<globals::token_vec*>(&token_tile.data[tid * globals::COL_BLOCK_SIZE]);

    __shared__ semaphore inputs_arrived;
    const int num_valid = __syncthreads_count(src_rank >= 0);
    if (num_valid == 0) return; // whole recv block is padding
    if (tid == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        tma::expect_bytes(inputs_arrived, num_valid * sizeof(globals::token_vec));
    }
    __syncthreads();

    if (src_rank >= 0)
        tma::load_async(token_vec, G.send_buffer[src_rank], {src_row_idx, col_block_idx}, inputs_arrived);
    wait(inputs_arrived, 0);
    if (src_rank >= 0)
        tma::store_async(G.recv_buffer, token_vec, {dst_row_idx, col_block_idx});
}

void dispatch(
    const at::Tensor &send_buffer,
    const std::vector<int64_t> &send_buffer_ptrs,
    const at::Tensor &recv_buffer,
    const at::Tensor &schedule_src_rank,
    const at::Tensor &schedule_src_token_idx,
    int topk
) {
    bf16 *send_buffer_data[globals::NUM_DEVICES];
    for (int i = 0; i < globals::NUM_DEVICES; ++i)
        send_buffer_data[i] = reinterpret_cast<bf16*>(send_buffer_ptrs[i]);

    globals G {
        .send_buffer = globals::send_buffer_pgl{send_buffer_data, nullptr, nullptr, static_cast<size_t>(send_buffer.size(0)), static_cast<size_t>(send_buffer.size(1))},
        .recv_buffer = kittens::py::tensor_to_gl<globals::recv_buffer_gl>(recv_buffer),
        .schedule_src_rank = kittens::py::tensor_to_gl<globals::index_gl>(schedule_src_rank),
        .schedule_src_token_idx = kittens::py::tensor_to_gl<globals::index_gl>(schedule_src_token_idx),
        .topk = topk,
    };

    kittens::py::launch_kernel<config, globals, dispatch_kernel>(G);
}

} // namespace dispatcher

namespace combiner {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 6;
    static constexpr int NUM_THREADS = 128;
};

struct globals {
    static constexpr int NUM_DEVICES = 4;
    static constexpr int ROW_BLOCK_SIZE = 128;
    static constexpr int COL_BLOCK_SIZE = 128;

    using token_tile = st_bf<ROW_BLOCK_SIZE, COL_BLOCK_SIZE, false>;
    using token_vec = sv_bf<COL_BLOCK_SIZE>;

    using send_buffer_gl = gl<bf16, 1, 1, -1, -1, token_vec>;
    using recv_buffer_pgl = pgl<gl<bf16, 1, 1, -1, -1, token_vec>, NUM_DEVICES, false>;
    using index_gl = gl<int, 1, 1, 1, -1>;

    send_buffer_gl send_buffer;      // (capacity, H)
    recv_buffer_pgl recv_buffer;     // (num_local_tokens * topk, H)
    index_gl schedule_dst_rank;      // (capacity,)
    index_gl schedule_dst_token_idx; // (capacity,)

    __host__ inline dim3 grid() const {
        return dim3(send_buffer.cols() / COL_BLOCK_SIZE, send_buffer.rows() / ROW_BLOCK_SIZE);
    }
    __host__ inline int dynamic_shared_memory() const {
        return static_cast<int>(sizeof(token_tile) + 1024);
    }
};

__device__ inline void combine_kernel(const globals &G) {
    const int tid = threadIdx.x;
    const int col_block_idx = blockIdx.x;
    const int row_block_idx = blockIdx.y;

    const int src_row_idx = row_block_idx * globals::ROW_BLOCK_SIZE + tid;
    const int dst_rank = G.schedule_dst_rank[{src_row_idx}];
    const int dst_row_idx = G.schedule_dst_token_idx[{src_row_idx}];

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    globals::token_tile &token_tile = allocator.allocate<globals::token_tile>();
    globals::token_vec &token_vec = *reinterpret_cast<globals::token_vec*>(&token_tile.data[tid * globals::COL_BLOCK_SIZE]);

    __shared__ semaphore inputs_arrived;
    const int num_valid = __syncthreads_count(dst_rank >= 0);
    if (num_valid == 0) return; // whole send block is padding
    if (tid == 0) {
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

void combine(
    const at::Tensor &send_buffer,
    const at::Tensor &recv_buffer,
    const std::vector<int64_t> &recv_buffer_ptrs,
    const at::Tensor &schedule_dst_rank,
    const at::Tensor &schedule_dst_token_idx
) {
    bf16 *recv_buffer_data[globals::NUM_DEVICES];
    for (int i = 0; i < globals::NUM_DEVICES; ++i)
        recv_buffer_data[i] = reinterpret_cast<bf16*>(recv_buffer_ptrs[i]);

    globals G {
        .send_buffer = kittens::py::tensor_to_gl<globals::send_buffer_gl>(send_buffer),
        .recv_buffer = globals::recv_buffer_pgl{recv_buffer_data, nullptr, nullptr, static_cast<size_t>(recv_buffer.size(0)), static_cast<size_t>(recv_buffer.size(1))},
        .schedule_dst_rank = kittens::py::tensor_to_gl<globals::index_gl>(schedule_dst_rank),
        .schedule_dst_token_idx = kittens::py::tensor_to_gl<globals::index_gl>(schedule_dst_token_idx),
    };

    kittens::py::launch_kernel<config, globals, combine_kernel>(G);
}

} // namespace combiner

PYBIND11_MODULE(_C, m) {
    m.def("schedule", &scheduler::schedule, "Build this rank's dispatch schedule from all-gathered topk routing",
          pybind11::arg("topk_all"), pybind11::arg("schedule_peer_rank"), pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("tokens_per_expert"), pybind11::arg("rank"));
    m.def("dispatch", &dispatcher::dispatch, "MoE pull-based dispatch",
          pybind11::arg("send_buffer"), pybind11::arg("send_buffer_ptrs"), pybind11::arg("recv_buffer"),
          pybind11::arg("schedule_src_rank"), pybind11::arg("schedule_src_token_idx"), pybind11::arg("topk"));
    m.def("combine", &combiner::combine, "MoE push-based combine",
          pybind11::arg("send_buffer"), pybind11::arg("recv_buffer"), pybind11::arg("recv_buffer_ptrs"),
          pybind11::arg("schedule_dst_rank"), pybind11::arg("schedule_dst_token_idx"));
}
