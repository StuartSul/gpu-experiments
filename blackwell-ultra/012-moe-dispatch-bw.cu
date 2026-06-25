#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

namespace scheduler {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int NUM_THREADS = 256;
};

struct globals {
    using topk_gl = gl<int, 1, -1, -1, -1>; // (world_size, num_local_tokens, topk)
    using index_gl = gl<int, 1, 1, 1, -1>;

    topk_gl topk;
    index_gl out_src_token_idx;
    index_gl out_dst_rank;
    index_gl out_dst_token_idx;
    int rank;
    int num_local_experts;
    int recv_capacity;

    __host__ inline int dynamic_shared_memory() const {
        const int world_size = topk.depth();
        const int num_experts = num_local_experts * world_size;
        return static_cast<int>((world_size * num_experts + 2 * num_experts) * sizeof(int)) + 1024;
    }
};

__device__ inline void schedule_kernel(const globals &G) {
    const int world_size = G.topk.depth();
    const int num_local_tokens = G.topk.rows();
    const int topk = G.topk.cols();
    const int num_local_experts = G.num_local_experts;
    const int num_experts = num_local_experts * world_size;
    const int rank = G.rank;
    const int recv_capacity = G.recv_capacity;
    const int num_slots = G.out_src_token_idx.cols();

    extern __shared__ int __shm[];
    int *expert_count_per_rank = __shm;                                   // [world_size * num_experts]
    int *send_offset = expert_count_per_rank + world_size * num_experts;  // [num_experts] this rank's dense send-slot start per expert
    int *rank_prefix = send_offset + num_experts;          // [num_experts] tokens contributed by earlier ranks per expert (dense)

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    for (int s = tid; s < num_slots; s += num_threads) {
        G.out_src_token_idx[{s}] = -1;
        G.out_dst_rank[{s}] = -1;
        G.out_dst_token_idx[{s}] = -1;
    }
    for (int x = tid; x < world_size * num_experts; x += num_threads)
        expert_count_per_rank[x] = 0;
    __syncthreads();

    // Phase 1: per-(rank, expert) token expert_count_per_rank over the gathered routing.
    const int total = world_size * num_local_tokens * topk;
    for (int idx = tid; idx < total; idx += num_threads) {
        const int source_rank = idx / (num_local_tokens * topk);
        const int local_idx = idx - source_rank * num_local_tokens * topk;
        const int local_token_idx = local_idx / topk;
        const int local_k = local_idx - local_token_idx * topk;
        atomicAdd(&expert_count_per_rank[source_rank * num_experts + G.topk[{source_rank, local_token_idx, local_k}]], 1);
    }
    __syncthreads();

    // Phase 2: dense per-expert offsets (no 128 rounding anywhere). send_offset[e] = this rank's
    // start slot for expert e in the token-dense send schedule; rank_prefix[e] = tokens earlier ranks
    // contribute to expert e, so this rank's tokens pack right after them in e's recv region. Padding
    // exists only as slack at the tail of each expert's recv region -- never per rank.
    if (tid == 0) {
        int next_offset = 0;
        for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
            send_offset[expert_idx] = next_offset;
            next_offset += expert_count_per_rank[rank * num_experts + expert_idx];
            int pre = 0;
            for (int r = 0; r < rank; ++r)
                pre += expert_count_per_rank[r * num_experts + expert_idx];
            rank_prefix[expert_idx] = pre;
        }
    }
    __syncthreads();

    // Phase 3: one thread per expert scans this rank's tokens in order and emits dense slots.
    for (int expert_idx = tid; expert_idx < num_experts; expert_idx += num_threads) {
        const int local_expert_idx = expert_idx % num_local_experts;
        const int dst_rank = expert_idx / num_local_experts;
        const int send_row0 = send_offset[expert_idx];
        const int recv_row0 = local_expert_idx * recv_capacity * 128 + rank_prefix[expert_idx];
        int w = 0;
        for (int i = 0; i < num_local_tokens; ++i)
            for (int k = 0; k < topk; ++k)
                if (G.topk[{rank, i, k}] == expert_idx) {
                    G.out_src_token_idx[{send_row0 + w}] = i;
                    G.out_dst_rank[{send_row0 + w}] = dst_rank;
                    G.out_dst_token_idx[{send_row0 + w}] = recv_row0 + w;
                    ++w;
                }
    }
}

void schedule(
    const at::Tensor &topk_ids,                 // (world_size, num_local_tokens, topk) int32, all-gathered
    const at::Tensor &schedule_src_token_idx,   // (send_capacity * 128,) int32  [out]
    const at::Tensor &schedule_dst_rank,        // (send_capacity * 128,) int32  [out]
    const at::Tensor &schedule_dst_token_idx,   // (send_capacity * 128,) int32  [out]
    int64_t rank,
    int64_t num_local_experts,
    int64_t recv_capacity
) {
    globals G {
        .topk = kittens::py::tensor_to_gl<globals::topk_gl>(topk_ids),
        .out_src_token_idx = kittens::py::tensor_to_gl<globals::index_gl>(schedule_src_token_idx),
        .out_dst_rank = kittens::py::tensor_to_gl<globals::index_gl>(schedule_dst_rank),
        .out_dst_token_idx = kittens::py::tensor_to_gl<globals::index_gl>(schedule_dst_token_idx),
        .rank = static_cast<int>(rank),
        .num_local_experts = static_cast<int>(num_local_experts),
        .recv_capacity = static_cast<int>(recv_capacity),
    };
    kittens::py::launch_kernel<config, globals, schedule_kernel>(G);
}

}

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

    using tokens_gl = gl<bf16, 1, 1, -1, -1, token_vec>;
    using recv_buffer_gl = pgl<gl<bf16, 1, -1, -1, -1, token_vec>, NUM_DEVICES, false>;
    using index_gl = gl<int, 1, 1, 1, -1>;

    tokens_gl tokens;
    recv_buffer_gl recv_buffer;
    index_gl schedule_src_token_idx;
    index_gl schedule_dst_rank;
    index_gl schedule_dst_token_idx;

    __host__ inline dim3 grid() const {
        return dim3(tokens.cols() / COL_BLOCK_SIZE, schedule_src_token_idx.cols() / ROW_BLOCK_SIZE);
    }
    __host__ inline int dynamic_shared_memory() const {
        return static_cast<int>(sizeof(token_tile) + 1024);
    }
};

__device__ inline void dispatch_kernel(const globals &G) {
    const int tid = threadIdx.x;
    const int col_block_idx = blockIdx.x;
    const int row_block_idx = blockIdx.y;

    const int schedule_idx = row_block_idx * globals::ROW_BLOCK_SIZE + tid;
    const int src_token_idx = G.schedule_src_token_idx[{schedule_idx}];
    const int dst_rank = G.schedule_dst_rank[{schedule_idx}];
    const int dst_token_idx = G.schedule_dst_token_idx[{schedule_idx}];

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    globals::token_tile &token_tile = allocator.allocate<globals::token_tile>();
    globals::token_vec &dst_vec = *reinterpret_cast<globals::token_vec*>(&token_tile.data[tid * globals::COL_BLOCK_SIZE]);

    __shared__ semaphore inputs_arrived;
    const int num_valid = __syncthreads_count(src_token_idx >= 0);
    if (num_valid == 0) return; // whole source block is empty (past the dispatch schedule)
    if (tid == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        tma::expect_bytes(inputs_arrived, num_valid * sizeof(globals::token_vec));
    }
    __syncthreads();

    if (src_token_idx >= 0)
        tma::load_async(dst_vec, G.tokens, {src_token_idx, col_block_idx}, inputs_arrived);
    wait(inputs_arrived, 0);
    if (src_token_idx >= 0)
        tma::store_async(G.recv_buffer[dst_rank], dst_vec, {dst_token_idx, col_block_idx});
}

void dispatch(
    const at::Tensor &tokens,
    const at::Tensor &recv_buffer,
    const std::vector<int64_t> &recv_buffer_ptrs,
    const at::Tensor &schedule_src_token_idx,
    const at::Tensor &schedule_dst_rank,
    const at::Tensor &schedule_dst_token_idx
) {
    bf16 *recv_buffer_data[globals::NUM_DEVICES];
    for (int i = 0; i < globals::NUM_DEVICES; ++i)
        recv_buffer_data[i] = reinterpret_cast<bf16*>(recv_buffer_ptrs[i]);

    globals G {
        .tokens = kittens::py::tensor_to_gl<globals::tokens_gl>(tokens),
        .recv_buffer = globals::recv_buffer_gl{recv_buffer_data, nullptr, static_cast<size_t>(1), static_cast<size_t>(recv_buffer.size(0)), static_cast<size_t>(recv_buffer.size(1))},
        .schedule_src_token_idx = kittens::py::tensor_to_gl<globals::index_gl>(schedule_src_token_idx),
        .schedule_dst_rank = kittens::py::tensor_to_gl<globals::index_gl>(schedule_dst_rank),
        .schedule_dst_token_idx = kittens::py::tensor_to_gl<globals::index_gl>(schedule_dst_token_idx),
    };

    kittens::py::launch_kernel<config, globals, dispatch_kernel>(G);
}

} // namespace dispatcher

PYBIND11_MODULE(_C, m) {
    m.def("schedule", &scheduler::schedule, "Build this rank's dispatch schedule from all-gathered topk routing",
          pybind11::arg("topk_ids"),
          pybind11::arg("schedule_src_token_idx"), pybind11::arg("schedule_dst_rank"), pybind11::arg("schedule_dst_token_idx"),
          pybind11::arg("rank"), pybind11::arg("num_local_experts"), pybind11::arg("recv_capacity"));
    m.def("dispatch", &dispatcher::dispatch, "MoE gather-and-dispatch (local gather -> push over NVLink)",
          pybind11::arg("tokens"), pybind11::arg("recv_buffer"), pybind11::arg("recv_buffer_ptrs"), 
          pybind11::arg("schedule_src_token_idx"), pybind11::arg("schedule_dst_rank"), pybind11::arg("schedule_dst_token_idx"));
}
