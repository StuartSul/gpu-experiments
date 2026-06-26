#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include <ATen/ops/zeros.h>

using namespace kittens;

namespace scheduler {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 256;
};

struct globals {
    using topk_gl = gl<int, 1, -1, -1, -1>; // (world_size, num_local_tokens, topk)
    using index_gl = gl<int, 1, 1, 1, -1>;  // (capacity,)

    topk_gl topk;                    // (world_size, num_local_tokens, topk)
    index_gl schedule_src_rank;      // (capacity,)
    index_gl schedule_src_token_idx; // (capacity,)
    index_gl tokens_per_src_rank;    // (world_size,) must be zero-initialized
    index_gl cursor;                 // (world_size,) must be zero-initialized

    int first_expert;                // inclusive
    int last_expert;                 // exclusive

    __host__ inline dim3 grid() const {
        const int blocks = std::clamp(static_cast<int>((topk.numel() + config::NUM_THREADS - 1) / config::NUM_THREADS), 1, 2048);
        return dim3(static_cast<unsigned int>(blocks));
    }
    __host__ inline int dynamic_shared_memory() const {
        return static_cast<int>(topk.depth() * sizeof(int));
    }
};

// Stage 1: Build the shared histogram
__device__ inline void count_kernel(const globals &G) {
    const int world_size = G.topk.depth();
    const int num_local_tokens = G.topk.rows();
    const int topk = G.topk.cols();
    const int rank_stride = num_local_tokens * topk;
    const int num_global_tokens = world_size * rank_stride;

    extern __shared__ int tokens_per_src_rank[]; // (world_size,)
    for (int rank = threadIdx.x; rank < world_size; rank += blockDim.x)
        tokens_per_src_rank[rank] = 0;
    __syncthreads();

    const int grid_stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_global_tokens; idx += grid_stride) {
        const int src_rank = idx / rank_stride;
        const int local_topk_idx = idx - src_rank * rank_stride;
        const int expert_idx = G.topk[{src_rank, local_topk_idx / topk, local_topk_idx % topk}];
        if (expert_idx >= G.first_expert && expert_idx < G.last_expert)
            atomicAdd(&tokens_per_src_rank[src_rank], 1);
    }
    __syncthreads();

    for (int rank = threadIdx.x; rank < world_size; rank += blockDim.x)
        if (tokens_per_src_rank[rank] != 0)
            atomicAdd(&G.tokens_per_src_rank[{rank}], tokens_per_src_rank[rank]);
}

// Stage 2: scatter each qualifying route to its round-robin row (outputs pre-filled with -1 by host).
__device__ inline void place_kernel(const globals &G) {
    const int world_size = G.topk.depth();
    const int num_local_tokens = G.topk.rows();
    const int topk = G.topk.cols();
    const int rank_stride = num_local_tokens * topk;
    const int num_global_tokens = world_size * rank_stride;

    extern __shared__ int tokens_per_src_rank[]; // (world_size,)
    for (int rank = threadIdx.x; rank < world_size; rank += blockDim.x)
        tokens_per_src_rank[rank] = G.tokens_per_src_rank[{rank}];
    __syncthreads();

    const int grid_stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_global_tokens; idx += grid_stride) {
        const int src_rank = idx / rank_stride;
        const int local_topk_idx = idx - src_rank * rank_stride;
        const int src_token_idx = local_topk_idx / topk;
        const int expert_idx = G.topk[{src_rank, src_token_idx, local_topk_idx % topk}];
        if (expert_idx >= G.first_expert && expert_idx < G.last_expert) {
            const int j = atomicAdd(&G.cursor[{src_rank}], 1); // this route is source rank src_rank'src_rank (j+1)-th
            int row = 0;
            for (int rank = 0; rank < world_size; ++rank) {
                const int num_tokens = tokens_per_src_rank[rank];
                row += (num_tokens < j ? num_tokens : j);                 // emits from rank during passes 0..j-1
                row += (rank < src_rank && num_tokens > j) ? 1 : 0;   // emits from rank before us during pass j
            }
            // TODO: prevent capacity overflow
            G.schedule_src_rank[{row}] = src_rank;
            G.schedule_src_token_idx[{row}] = src_token_idx;
        }
    }
}

void schedule(
    const at::Tensor &topk_all,
    const at::Tensor &schedule_src_rank,
    const at::Tensor &schedule_src_token_idx,
    int rank,
    int num_local_experts
) {
    const int world_size = static_cast<int>(topk_all.size(0));
    const auto i32 = topk_all.options().dtype(at::kInt);
    at::Tensor tokens_per_src_rank = at::zeros({world_size}, i32);
    at::Tensor cursor = at::zeros({world_size}, i32);
    schedule_src_rank.fill_(-1);
    schedule_src_token_idx.fill_(-1);

    globals G {
        .topk = kittens::py::tensor_to_gl<globals::topk_gl>(topk_all),
        .schedule_src_rank = kittens::py::tensor_to_gl<globals::index_gl>(schedule_src_rank),
        .schedule_src_token_idx = kittens::py::tensor_to_gl<globals::index_gl>(schedule_src_token_idx),
        .tokens_per_src_rank = kittens::py::tensor_to_gl<globals::index_gl>(tokens_per_src_rank),
        .cursor = kittens::py::tensor_to_gl<globals::index_gl>(cursor),
        .first_expert = rank * num_local_experts,
        .last_expert = rank * num_local_experts + num_local_experts,
    };

    kittens::py::launch_kernel<config, globals, count_kernel>(G);
    kittens::py::launch_kernel<config, globals, place_kernel>(G);
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

    using tokens_pgl = pgl<gl<bf16, 1, 1, -1, -1, token_vec>, NUM_DEVICES, false>;
    using recv_buffer_gl = gl<bf16, 1, 1, -1, -1, token_vec>;
    using index_gl = gl<int, 1, 1, 1, -1>;

    tokens_pgl tokens;               // (num_local_tokens, H)
    recv_buffer_gl recv_buffer;      // (capacity, H)
    index_gl schedule_src_rank;      // (capacity,)
    index_gl schedule_src_token_idx; // (capacity,)

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
    const int src_row_idx = G.schedule_src_token_idx[{dst_row_idx}];

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    globals::token_tile &token_tile = allocator.allocate<globals::token_tile>();
    globals::token_vec &dst_vec = *reinterpret_cast<globals::token_vec*>(&token_tile.data[tid * globals::COL_BLOCK_SIZE]);

    __shared__ semaphore inputs_arrived;
    const int num_valid = __syncthreads_count(src_row_idx >= 0);
    if (num_valid == 0) return; // whole recv block is padding
    if (tid == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        tma::expect_bytes(inputs_arrived, num_valid * sizeof(globals::token_vec));
    }
    __syncthreads();

    if (src_row_idx >= 0)
        tma::load_async(dst_vec, G.tokens[src_rank], {src_row_idx, col_block_idx}, inputs_arrived);
    wait(inputs_arrived, 0);
    if (src_row_idx >= 0)
        tma::store_async(G.recv_buffer, dst_vec, {dst_row_idx, col_block_idx});
}

void dispatch(
    const at::Tensor &tokens,
    const std::vector<int64_t> &tokens_ptrs,
    const at::Tensor &recv_buffer,
    const at::Tensor &schedule_src_rank,
    const at::Tensor &schedule_src_token_idx
) {
    bf16 *tokens_data[globals::NUM_DEVICES];
    for (int i = 0; i < globals::NUM_DEVICES; ++i)
        tokens_data[i] = reinterpret_cast<bf16*>(tokens_ptrs[i]);

    globals G {
        .tokens = globals::tokens_pgl{tokens_data, nullptr, nullptr, static_cast<size_t>(tokens.size(0)), static_cast<size_t>(tokens.size(1))},
        .recv_buffer = kittens::py::tensor_to_gl<globals::recv_buffer_gl>(recv_buffer),
        .schedule_src_rank = kittens::py::tensor_to_gl<globals::index_gl>(schedule_src_rank),
        .schedule_src_token_idx = kittens::py::tensor_to_gl<globals::index_gl>(schedule_src_token_idx),
    };

    kittens::py::launch_kernel<config, globals, dispatch_kernel>(G);
}

} // namespace dispatcher

PYBIND11_MODULE(_C, m) {
    m.def("schedule", &scheduler::schedule, "Build this rank's dispatch schedule from all-gathered topk routing",
          pybind11::arg("topk_all"), pybind11::arg("schedule_src_rank"), pybind11::arg("schedule_src_token_idx"),
          pybind11::arg("rank"), pybind11::arg("num_local_experts"));
    m.def("dispatch", &dispatcher::dispatch, "MoE pull-based dispatch (destination pulls remote tokens over NVLink)",
          pybind11::arg("tokens"), pybind11::arg("tokens_ptrs"), pybind11::arg("recv_buffer"),
          pybind11::arg("schedule_src_rank"), pybind11::arg("schedule_src_token_idx"));
}
