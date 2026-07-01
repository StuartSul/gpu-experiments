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

namespace dispatch_combiner {

struct config {
    static constexpr int NUM_DEVICES = 4;

    static constexpr int MINIBATCH_SIZE = 4096;

    static constexpr int Mb = 64;
    static constexpr int Nb = 256;
    static constexpr int PIPE_DEPTH = 7;

    static constexpr int MLP_Mb = 256;
    static constexpr int MLP_Nb = 256;
    static constexpr int MLP_CLUSTER_SIZE = 2;

    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = Mb;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1024;

    static_assert(MINIBATCH_SIZE % Mb == 0, "MINIBATCH_SIZE must be a multiple of Mb");
    static_assert(PIPE_DEPTH * Mb * Nb * sizeof(bf16) <= DYNAMIC_SHARED_MEMORY, "Dispatch pipeline does not fit in shared memory");
};

struct globals {
    using token_tile = st_bf<config::Mb, config::Nb, false>;
    using token_vec = sv_bf<config::Nb>;

    using buffer_gl = gl<bf16, 1, 1, -1, -1, token_vec>;
    using buffer_pgl = pgl<buffer_gl, config::NUM_DEVICES, false>;
    using index_gl = gl<int, 1, 1, 1, -1>;

    buffer_pgl dispatch_send_buffer;   // (num_local_tokens, H)
    buffer_gl  dispatch_recv_buffer;   // (capacity, H)
    buffer_gl  combine_send_buffer;    // (capacity, H)
    buffer_pgl combine_recv_buffer;    // (num_local_tokens * topk, H)
    index_gl schedule_peer_rank;       // (capacity,)
    index_gl schedule_peer_token_idx;  // (capacity,) original_token_idx * topk + k
    index_gl tokens_per_expert;        // (num_local_experts,)
    index_gl dispatch_counter;         // (num_minibatches,)
    index_gl combine_counter;          // (num_minibatches,)

    int topk;

    __host__ inline dim3 grid() const {
        const int num_tiles = (dispatch_recv_buffer.rows() / config::Mb) * (dispatch_recv_buffer.cols() / config::Nb);
        return dim3(2 * ((num_tiles + config::PIPE_DEPTH - 1) / config::PIPE_DEPTH));
    }
};

__device__ __forceinline__ void dispatch_combine_kernel(const globals &G) {
    extern __shared__ int __shm[];
    const uint64_t smem_base_addr = (reinterpret_cast<uint64_t>(&__shm[0]) + 1023) & ~uint64_t(1023);
    auto &token_vecs = *reinterpret_cast<globals::token_vec (*)[config::PIPE_DEPTH][config::Mb]>(smem_base_addr);

    __shared__ semaphore inputs_arrived[config::PIPE_DEPTH];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int stage = 0; stage < config::PIPE_DEPTH; ++stage)
            init_semaphore(inputs_arrived[stage], 0, 1);
    } // no sync needed

    const int tid = threadIdx.x;
    const int col_blocks = G.dispatch_recv_buffer.cols() / config::Nb;
    const int num_tasks = gridDim.x / 2;
    const bool is_dispatch = blockIdx.x < num_tasks;
    const int first_tile_idx = (is_dispatch ? blockIdx.x : blockIdx.x - num_tasks) * config::PIPE_DEPTH;

    int num_tokens = 0;
    for (int expert_idx = 0; expert_idx < G.tokens_per_expert.cols(); ++expert_idx)
        num_tokens += G.tokens_per_expert[{expert_idx}];
    const int num_active_stages = min(config::PIPE_DEPTH, num_tokens / config::Mb * col_blocks - first_tile_idx); // because we pad to 256
    if (num_active_stages <= 0) return;

    const int first_row_idx = first_tile_idx / col_blocks * config::Mb + tid;
    const int first_col_block_idx = first_tile_idx % col_blocks;

    int row_idx[config::PIPE_DEPTH], col_block_idx[config::PIPE_DEPTH], peer_rank[config::PIPE_DEPTH], peer_token_idx[config::PIPE_DEPTH], num_valid[config::PIPE_DEPTH];
    #pragma unroll
    for (int stage = 0, row = first_row_idx, col = first_col_block_idx; stage < config::PIPE_DEPTH; ++stage) {
        const bool active = stage < num_active_stages;
        row_idx[stage] = row;
        col_block_idx[stage] = col;
        peer_rank[stage] = active ? G.schedule_peer_rank[{row}] : -1;
        peer_token_idx[stage] = active ? G.schedule_peer_token_idx[{row}] : -1;
        num_valid[stage] = !active ? 0
                         : (stage == 0 || col == 0) ? __syncthreads_count(peer_rank[stage] >= 0) // whole block reaches this, so it also orders init_semaphore too!
                         : num_valid[stage - 1];
        if (++col == col_blocks) { col = 0; row += config::Mb; }
    }

    if (tid == 0) {
        if (!is_dispatch) {
            const int first_minibatch_idx = first_row_idx / config::MINIBATCH_SIZE;
            const int last_minibatch_idx = (first_tile_idx + num_active_stages - 1) / col_blocks * config::Mb / config::MINIBATCH_SIZE;
            for (int minibatch_idx = first_minibatch_idx; minibatch_idx <= last_minibatch_idx; ++minibatch_idx) {
                const int minibatch_rows = min(config::MINIBATCH_SIZE, num_tokens - minibatch_idx * config::MINIBATCH_SIZE);
                const int expected = ((minibatch_rows + config::MLP_Mb - 1) / config::MLP_Mb) * (G.combine_send_buffer.cols() / config::MLP_Nb) * config::MLP_CLUSTER_SIZE;
                int combine_counter;
                while (true) {
                    asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}" : "=r"(combine_counter) : "l"(&G.combine_counter[{minibatch_idx}]) : "memory");
                    if (combine_counter >= expected) break;
                    __nanosleep(16);
                }
            }
            asm volatile("{fence.acquire.gpu;}" ::: "memory");
        }
        #pragma unroll
        for (int stage = 0; stage < config::PIPE_DEPTH; ++stage)
            if (stage < num_active_stages)
                tma::expect_bytes(inputs_arrived[stage], num_valid[stage] * sizeof(globals::token_vec)); // 0 bytes completes the phase immediately
    }
    __syncthreads();

    #pragma unroll
    for (int stage = 0; stage < config::PIPE_DEPTH; ++stage) {
        if (peer_rank[stage] >= 0) {
            if (is_dispatch)
                tma::load_async(token_vecs[stage][tid], G.dispatch_send_buffer[peer_rank[stage]], {peer_token_idx[stage] / G.topk, col_block_idx[stage]}, inputs_arrived[stage]);
            else
                tma::load_async(token_vecs[stage][tid], G.combine_send_buffer, {row_idx[stage], col_block_idx[stage]}, inputs_arrived[stage]);
        }
    }

    #pragma unroll
    for (int stage = 0; stage < config::PIPE_DEPTH; ++stage) {
        if (stage < num_active_stages) {
            wait(inputs_arrived[stage], 0);
            if (peer_rank[stage] >= 0) {
                if (is_dispatch)
                    tma::store_async(G.dispatch_recv_buffer, token_vecs[stage][tid], {row_idx[stage], col_block_idx[stage]});
                else
                    tma::store_async(G.combine_recv_buffer[peer_rank[stage]], token_vecs[stage][tid], {peer_token_idx[stage], col_block_idx[stage]});
            }
        }
    }

    if (is_dispatch) {
        tma::store_async_wait();
        __syncthreads();
        if (tid == 0) {
            const int tiles_per_minibatch = config::MINIBATCH_SIZE / config::Mb * col_blocks;
            const int minibatch_idx = first_tile_idx / tiles_per_minibatch;
            const int first_count = min(num_active_stages, (minibatch_idx + 1) * tiles_per_minibatch - first_tile_idx);
            asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}" :: "l"(&G.dispatch_counter[{minibatch_idx}]), "r"(first_count) : "memory");
            if (first_count < num_active_stages)
                asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}" :: "l"(&G.dispatch_counter[{minibatch_idx + 1}]), "r"(num_active_stages - first_count) : "memory");
        }
    }
}

void dispatch_combine(
    const at::Tensor &dispatch_send_buffer,
    const std::vector<int64_t> &dispatch_send_buffer_ptrs,
    const at::Tensor &dispatch_recv_buffer,
    const at::Tensor &combine_send_buffer,
    const at::Tensor &combine_recv_buffer,
    const std::vector<int64_t> &combine_recv_buffer_ptrs,
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &tokens_per_expert,
    const at::Tensor dispatch_counter,
    const at::Tensor &combine_counter,
    const int topk
) {
    dispatch_counter.zero_();

    bf16 *dispatch_send_buffer_data[config::NUM_DEVICES];
    bf16 *combine_recv_buffer_data[config::NUM_DEVICES];
    for (int i = 0; i < config::NUM_DEVICES; ++i) {
        dispatch_send_buffer_data[i] = reinterpret_cast<bf16*>(dispatch_send_buffer_ptrs[i]);
        combine_recv_buffer_data[i] = reinterpret_cast<bf16*>(combine_recv_buffer_ptrs[i]);
    }

    globals G {
        .dispatch_send_buffer = globals::buffer_pgl{dispatch_send_buffer_data, nullptr, nullptr, static_cast<size_t>(dispatch_send_buffer.size(0)), static_cast<size_t>(dispatch_send_buffer.size(1))},
        .dispatch_recv_buffer = kittens::py::tensor_to_gl<globals::buffer_gl>(dispatch_recv_buffer),
        .combine_send_buffer = kittens::py::tensor_to_gl<globals::buffer_gl>(combine_send_buffer),
        .combine_recv_buffer = globals::buffer_pgl{combine_recv_buffer_data, nullptr, nullptr, static_cast<size_t>(combine_recv_buffer.size(0)), static_cast<size_t>(combine_recv_buffer.size(1))},
        .schedule_peer_rank = kittens::py::tensor_to_gl<globals::index_gl>(schedule_peer_rank),
        .schedule_peer_token_idx = kittens::py::tensor_to_gl<globals::index_gl>(schedule_peer_token_idx),
        .tokens_per_expert = kittens::py::tensor_to_gl<globals::index_gl>(tokens_per_expert),
        .dispatch_counter = kittens::py::tensor_to_gl<globals::index_gl>(dispatch_counter),
        .combine_counter = kittens::py::tensor_to_gl<globals::index_gl>(combine_counter),
        .topk = topk,
    };

    kittens::py::launch_kernel<config, globals, dispatch_combine_kernel>(G);
}

} // namespace dispatch_combiner

PYBIND11_MODULE(_C, m) {
    m.def("schedule", &scheduler::schedule, "Build this rank's dispatch schedule from all-gathered topk routing",
          pybind11::arg("topk_all"), pybind11::arg("schedule_peer_rank"), pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("tokens_per_expert"), pybind11::arg("rank"));
    m.def("dispatch_combine", &dispatch_combiner::dispatch_combine, "Combined MoE pull-based dispatch and push-based combine",
          pybind11::arg("dispatch_send_buffer"), pybind11::arg("dispatch_send_buffer_ptrs"), pybind11::arg("dispatch_recv_buffer"),
          pybind11::arg("combine_send_buffer"), pybind11::arg("combine_recv_buffer"), pybind11::arg("combine_recv_buffer_ptrs"), 
          pybind11::arg("schedule_peer_rank"), pybind11::arg("schedule_peer_token_idx"), pybind11::arg("tokens_per_expert"),
          pybind11::arg("dispatch_counter"), pybind11::arg("combine_counter"), pybind11::arg("topk"));
}
