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
    index_gl schedule_peer_rank;         // (capacity,) must be initialized to -1
    index_gl schedule_peer_token_idx;    // (capacity,) original_token_idx * topk + k
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

// Stage 2: Pad each expert's total token count by EXPERT_PADDING
static __device__ __forceinline__ void pad_kernel(const globals &G) {
    const int local_expert = blockIdx.x;
    const int world_size = G.topk.depth();
    int num_tokens = 0;
    for (int peer_rank = 0; peer_rank < world_size; ++peer_rank)
        num_tokens += G.tokens_per_expert_and_peer[{local_expert * world_size + peer_rank}];
    G.tokens_per_expert[{local_expert}] = (num_tokens + config::EXPERT_PADDING - 1) / config::EXPERT_PADDING * config::EXPERT_PADDING;
}

// Stage 3: Schedule each token into its expert's 256-padded segment
static __device__ __forceinline__ void schedule_kernel(const globals &G) {
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

static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor> schedule(
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

    globals G {
        .topk = kittens::py::tensor_to_gl<globals::topk_gl>(topk_all),
        .schedule_peer_rank = kittens::py::tensor_to_gl<globals::index_gl>(schedule_peer_rank),
        .schedule_peer_token_idx = kittens::py::tensor_to_gl<globals::index_gl>(schedule_peer_token_idx),
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

    return {schedule_peer_rank, schedule_peer_token_idx, tokens_per_expert};
}

}; // struct scheduler

template <int NUM_DEVICES>
struct dispatch_mlp_swiglu_combiner {

struct config {
    // Grouped GEMM
    static constexpr int MLP_Mb = 256;
    static constexpr int MLP_Nb = 256;
    static constexpr int MLP_Kb = 64;
    static constexpr int SUPERGROUP_SIZE = 8;
    static constexpr int LOAD_PIPE_DEPTH = 5;
    static constexpr int EPI_PIPE_DEPTH = 4;
    static constexpr int NUM_D_TILES = EPI_PIPE_DEPTH > 1 ? 2 : 1;

    // Fused SwiGLU
    static constexpr int SWIGLU_Mb = 128;
    static constexpr int SWIGLU_Nb = 128;
    static constexpr int SWIGLU_PIPE_DEPTH = 3;

    // Dispatch/Combine
    static constexpr int DISPATCH_COMBINE_Mb = 64;
    static constexpr int DISPATCH_COMBINE_Nb = 256;
    static constexpr int DISPATCH_COMBINE_PIPE_DEPTH = 7;
    
    // Kernel launch
    static constexpr int CLC_PIPE_DEPTH = 1;
    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int NUM_CONSUMERS = 1;
    static constexpr int NUM_PRODUCERS = 1;
    static constexpr int NUM_WARPS = (NUM_CONSUMERS + NUM_PRODUCERS) * WARPGROUP_WARPS; // 8
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS; // 256
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1024;
};

struct globals {
    // Grouped GEMM
    using a_tile = st_bf<config::MLP_Mb / 2, config::MLP_Kb>;
    using b_tile = st_bf<config::MLP_Nb / 2, config::MLP_Kb>;
    using d_tile = st_bf<config::MLP_Mb / 2, config::MLP_Nb / config::EPI_PIPE_DEPTH>;

    // Fused SwiGLU
    using swiglu_tile = st_bf<config::SWIGLU_Mb, config::SWIGLU_Nb>;

    // Dispatch/Combine
    using token_vec = sv_bf<config::DISPATCH_COMBINE_Nb>;

    // Global layouts
    using activation_gl = gl<bf16, 1, 1, -1, -1, a_tile, swiglu_tile, token_vec>;
    using activation_pgl = pgl<activation_gl, NUM_DEVICES, false>;
    using weight_gl = gl<bf16, 1, -1, -1, -1, b_tile>;
    using index_gl = gl<int, 1, 1, 1, -1>;

    activation_gl x_shared;              // (num_local_tokens, H)
    activation_gl x_routed;              // (capacity, H)
    activation_gl gate_shared;           // (num_local_tokens, I)
    activation_gl gate_routed;           // (capacity, I)
    activation_gl up_shared;             // (num_local_tokens, I)
    activation_gl up_routed;             // (capacity, I)
    activation_gl hidden_shared;         // (num_local_tokens, I)
    activation_gl hidden_routed;         // (capacity, I)
    activation_gl y_shared;              // (num_local_tokens, H)
    activation_gl y_routed;              // (capacity, H)

    activation_pgl x_routed_send_buffer; // (num_local_tokens, H)
    activation_pgl y_routed_recv_buffer; // (num_local_tokens * topk, H)

    weight_gl w_shared_gate;             // (I, H)
    weight_gl w_routed_gate;             // (num_local_experts, I, H)
    weight_gl w_shared_up;               // (I, H)
    weight_gl w_routed_up;               // (num_local_experts, I, H)
    weight_gl w_shared_down;             // (H, I)
    weight_gl w_routed_down;             // (num_local_experts, H, I)

    index_gl schedule_peer_rank;         // (capacity,)
    index_gl schedule_peer_token_idx;    // (capacity,)
    index_gl tokens_per_expert;          // (num_local_experts,)

    index_gl mlp_swiglu_counter;         // (shared_gate_up_tasks + routed_gate_up_tasks + shared_row_blocks + routed_row_blocks,)
    index_gl dispatch_counter;           // (num_minibatches,)
    index_gl combine_counter;            // (num_minibatches,)

    const int topk;
    const int num_comm_sms;
    const int minibatch_size;

    __host__ inline dim3 grid() const {
        const int num_minibatches = (x_routed.rows() + minibatch_size - 1) / minibatch_size;
        const int shared_row_blocks = x_shared.rows() / config::MLP_Mb;
        const int minibatch_routed_row_blocks = minibatch_size / config::MLP_Mb;
        const int shared_gate_up_tasks = shared_row_blocks * (w_shared_gate.rows() / config::MLP_Nb);
        const int minibatch_routed_gate_up_tasks = minibatch_routed_row_blocks * (w_routed_gate.rows() / config::MLP_Nb);
        const int shared_swiglu_tiles = (hidden_shared.rows() / config::SWIGLU_Mb) * (hidden_shared.cols() / config::SWIGLU_Nb);
        const int minibatch_routed_swiglu_tiles = (minibatch_size / config::SWIGLU_Mb) * (hidden_routed.cols() / config::SWIGLU_Nb);
        const int shared_swiglu_tasks = (shared_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_PIPE_DEPTH);
        const int minibatch_routed_swiglu_tasks = (minibatch_routed_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_PIPE_DEPTH);
        const int shared_down_tasks = shared_row_blocks * (w_shared_down.rows() / config::MLP_Nb);
        const int minibatch_routed_down_tasks = minibatch_routed_row_blocks * (w_routed_down.rows() / config::MLP_Nb);
        const int shared_tasks = 2 * shared_gate_up_tasks + shared_swiglu_tasks + shared_down_tasks;
        const int minibatch_tasks = 2 * minibatch_routed_gate_up_tasks + minibatch_routed_swiglu_tasks + minibatch_routed_down_tasks;
        return dim3(config::CLUSTER_SIZE * (shared_tasks + num_minibatches * minibatch_tasks) + num_comm_sms);
    }
};

template <bool IS_DISPATCH>
static __device__ __forceinline__ void dispatch_combine_kernel(
    const globals &G,
    semaphore (&inputs_arrived)[config::DISPATCH_COMBINE_PIPE_DEPTH],
    uint32_t &bitfield,
    int task_idx,
    uint64_t smem_base_addr
) {
    auto &token_vecs = *reinterpret_cast<typename globals::token_vec (*)[config::DISPATCH_COMBINE_PIPE_DEPTH][config::DISPATCH_COMBINE_Mb]>(smem_base_addr);

    const int tid = threadIdx.x;
    const bool is_worker = tid < config::DISPATCH_COMBINE_Mb; // only these threads move tokens, but all threads join the barriers and waits

    const int col_blocks = G.x_routed.cols() / config::DISPATCH_COMBINE_Nb;
    const int first_tile_idx = task_idx * config::DISPATCH_COMBINE_PIPE_DEPTH;

    int num_tokens = 0;
    for (int expert_idx = 0; expert_idx < G.tokens_per_expert.cols(); ++expert_idx)
        num_tokens += G.tokens_per_expert[{expert_idx}];
    const int num_valid_tiles = min(config::DISPATCH_COMBINE_PIPE_DEPTH, num_tokens / config::DISPATCH_COMBINE_Mb * col_blocks - first_tile_idx); // because we pad to 256
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
        peer_rank[stage] = is_valid_tile && is_worker ? G.schedule_peer_rank[{row}] : -1;
        peer_token_idx[stage] = is_valid_tile && is_worker ? G.schedule_peer_token_idx[{row}] : -1;
        num_valid[stage] = !is_valid_tile ? 0
                         : (stage == 0 || col == 0) ? __syncthreads_count(peer_rank[stage] >= 0)
                         : num_valid[stage - 1];
        if (++col == col_blocks) { col = 0; row += config::DISPATCH_COMBINE_Mb; }
    }

    if (tid == 0) {
        if (!IS_DISPATCH) {
            // Wait until the routed down GEMMs have fully written every minibatch this task reads
            const int first_minibatch_idx = first_row_idx / G.minibatch_size;
            const int last_minibatch_idx = (first_tile_idx + num_valid_tiles - 1) / col_blocks * config::DISPATCH_COMBINE_Mb / G.minibatch_size;
            for (int minibatch_idx = first_minibatch_idx; minibatch_idx <= last_minibatch_idx; ++minibatch_idx) {
                const int minibatch_rows = min(G.minibatch_size, num_tokens - minibatch_idx * G.minibatch_size);
                const int expected = ((minibatch_rows + config::MLP_Mb - 1) / config::MLP_Mb) * (G.y_routed.cols() / config::MLP_Nb) * config::CLUSTER_SIZE;
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
            const int tiles_per_minibatch = G.minibatch_size / config::DISPATCH_COMBINE_Mb * col_blocks; // a task straddles at most one minibatch boundary
            const int minibatch_idx = first_tile_idx / tiles_per_minibatch;
            const int first_count = min(num_valid_tiles, (minibatch_idx + 1) * tiles_per_minibatch - first_tile_idx);
            asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}" :: "l"(&G.dispatch_counter[{minibatch_idx}]), "r"(first_count) : "memory");
            if (first_count < num_valid_tiles)
                asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}" :: "l"(&G.dispatch_counter[{minibatch_idx + 1}]), "r"(num_valid_tiles - first_count) : "memory");
        }
    } else {
        // The next task on this CTA reuses token_vecs; make sure outgoing stores are done reading shared memory
        tma::store_async_read_wait();
        __syncthreads();
    }
}

template <bool IS_SHARED>
static __device__ __forceinline__ void swiglu(
    const globals &g,
    semaphore (&swiglu_inputs_arrived)[config::SWIGLU_PIPE_DEPTH],
    uint32_t &swiglu_bitfield,
    int cta_rank,
    int minibatch_idx,
    int task_idx,
    uint64_t smem_base_addr
) {
    typename globals::swiglu_tile (&a_smem)[config::SWIGLU_PIPE_DEPTH] = *reinterpret_cast<typename globals::swiglu_tile (*)[config::SWIGLU_PIPE_DEPTH]>(smem_base_addr);
    typename globals::swiglu_tile (&b_smem)[config::SWIGLU_PIPE_DEPTH] = *reinterpret_cast<typename globals::swiglu_tile (*)[config::SWIGLU_PIPE_DEPTH]>(smem_base_addr + sizeof(a_smem));

    const typename globals::activation_gl &x_gmem      = IS_SHARED ? g.x_shared : g.x_routed;
    const typename globals::activation_gl &gate_gmem   = IS_SHARED ? g.gate_shared : g.gate_routed;
    const typename globals::activation_gl &up_gmem     = IS_SHARED ? g.up_shared : g.up_routed;
    const typename globals::activation_gl &hidden_gmem = IS_SHARED ? g.hidden_shared : g.hidden_routed;
    const typename globals::weight_gl     &w_gate_gmem = IS_SHARED ? g.w_shared_gate : g.w_routed_gate;

    const int shared_row_blocks = g.x_shared.rows() / config::MLP_Mb;
    const int routed_row_blocks = g.x_routed.rows() / config::MLP_Mb;
    const int shared_gate_up_tasks = shared_row_blocks * (g.w_shared_gate.rows() / config::MLP_Nb);
    const int routed_gate_up_tasks = routed_row_blocks * (g.w_routed_gate.rows() / config::MLP_Nb);
    const int gate_counter_offset = IS_SHARED ? 0 : shared_gate_up_tasks;
    const int swiglu_counter_offset = shared_gate_up_tasks + routed_gate_up_tasks + (IS_SHARED ? 0 : shared_row_blocks);
    const int intermediate_col_blocks = w_gate_gmem.rows() / config::MLP_Nb;

    int num_tokens;
    if constexpr (IS_SHARED) {
        num_tokens = x_gmem.rows();
    } else {
        num_tokens = 0;
        for (int expert_idx = 0; expert_idx < w_gate_gmem.depth(); ++expert_idx)
            num_tokens += g.tokens_per_expert[{expert_idx}];
    }
    const int row_blocks = num_tokens / config::SWIGLU_Mb;
    const int col_blocks = hidden_gmem.cols() / config::SWIGLU_Nb;
    const int num_tiles = row_blocks * col_blocks;
    int first_tile_idx, tile_end;
    if constexpr (IS_SHARED) {
        first_tile_idx = task_idx * config::CLUSTER_SIZE * config::SWIGLU_PIPE_DEPTH + cta_rank * config::SWIGLU_PIPE_DEPTH;
        tile_end = num_tiles;
    } else {
        const int num_tiles_per_minibatch = (g.minibatch_size / config::SWIGLU_Mb) * col_blocks;
        const int minibatch_first_tile_idx = minibatch_idx * num_tiles_per_minibatch;
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

                tma::load_async(a_smem[stage], gate_gmem, {row, col}, swiglu_inputs_arrived[stage]);
                tma::load_async(b_smem[stage], up_gmem,   {row, col}, swiglu_inputs_arrived[stage]);
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
    semaphore (&gemm_inputs_finished)[config::LOAD_PIPE_DEPTH],
    semaphore &gemm_outputs_arrived,
    semaphore &gemm_outputs_finished,
    uint32_t &gemm_bitfield,
    int cta_rank,
    int minibatch_idx,
    int task_idx,
    uint64_t smem_base_addr,
    expert_gemm_kind kind,
    tt<float, config::MLP_Mb / 2, config::MLP_Nb> d_tt
) {
    typename globals::a_tile (&a_smem)[config::LOAD_PIPE_DEPTH] = *reinterpret_cast<typename globals::a_tile (*)[config::LOAD_PIPE_DEPTH]>(smem_base_addr);
    typename globals::b_tile (&b_smem)[config::LOAD_PIPE_DEPTH] = *reinterpret_cast<typename globals::b_tile (*)[config::LOAD_PIPE_DEPTH]>(smem_base_addr + sizeof(a_smem));
    typename globals::d_tile (&d_smem)[config::NUM_D_TILES] = *reinterpret_cast<typename globals::d_tile (*)[config::NUM_D_TILES]>(smem_base_addr + sizeof(a_smem) + sizeof(b_smem));

    const typename globals::activation_gl &x_gmem      = IS_SHARED ? g.x_shared : g.x_routed;
    const typename globals::activation_gl &gate_gmem   = IS_SHARED ? g.gate_shared : g.gate_routed;
    const typename globals::activation_gl &up_gmem     = IS_SHARED ? g.up_shared : g.up_routed;
    const typename globals::activation_gl &hidden_gmem = IS_SHARED ? g.hidden_shared : g.hidden_routed;
    const typename globals::activation_gl &y_gmem      = IS_SHARED ? g.y_shared : g.y_routed;
    const typename globals::weight_gl     &w_gate_gmem = IS_SHARED ? g.w_shared_gate : g.w_routed_gate;
    const typename globals::weight_gl     &w_up_gmem   = IS_SHARED ? g.w_shared_up : g.w_routed_up;
    const typename globals::weight_gl     &w_down_gmem = IS_SHARED ? g.w_shared_down : g.w_routed_down;

    const typename globals::activation_gl &a_gmem = kind == expert_gemm_kind::DOWN ? hidden_gmem : x_gmem;
    const typename globals::weight_gl     &b_gmem = kind == expert_gemm_kind::GATE ? w_gate_gmem : (kind == expert_gemm_kind::UP ? w_up_gmem : w_down_gmem);
    const typename globals::activation_gl &d_gmem = kind == expert_gemm_kind::GATE ? gate_gmem   : (kind == expert_gemm_kind::UP ? up_gmem : y_gmem);

    const int iters_per_task = a_gmem.cols() / config::MLP_Kb;
    const int col_blocks     = b_gmem.rows() / config::MLP_Nb;
    const int shared_row_blocks = g.x_shared.rows() / config::MLP_Mb;
    const int routed_row_blocks = g.x_routed.rows() / config::MLP_Mb;
    const int shared_gate_up_tasks = shared_row_blocks * (g.w_shared_gate.rows() / config::MLP_Nb);
    const int routed_gate_up_tasks = routed_row_blocks * (g.w_routed_gate.rows() / config::MLP_Nb);
    const int gate_counter_offset = IS_SHARED ? 0 : shared_gate_up_tasks;
    const int swiglu_counter_offset = shared_gate_up_tasks + routed_gate_up_tasks + (IS_SHARED ? 0 : shared_row_blocks);

    int3 tile_coord = {-1, -1, -1};
    if constexpr (IS_SHARED) {
        const int row_blocks = x_gmem.rows() / config::MLP_Mb;
        const int num_tasks = row_blocks * col_blocks;
        if (task_idx < num_tasks) {
            const int2 swizzled = get_swizzled_2d_idx<config::SUPERGROUP_SIZE>(row_blocks, col_blocks, task_idx);
            tile_coord = {swizzled.x, swizzled.y, 0};
        }
    } else {
        const int minibatch_routed_row_blocks = g.minibatch_size / config::MLP_Mb;
        const int minibatch_routed_row_offset = minibatch_idx * minibatch_routed_row_blocks;
        int row_block_offset = 0;
        for (int expert_idx = 0; expert_idx < b_gmem.depth(); ++expert_idx) {
            const int expert_row_blocks = g.tokens_per_expert[{expert_idx}] / config::MLP_Mb;
            const int first_row_block = max(minibatch_routed_row_offset, row_block_offset);
            const int row_blocks = max(0, min(minibatch_routed_row_offset + minibatch_routed_row_blocks, row_block_offset + expert_row_blocks) - first_row_block);
            const int num_tasks = row_blocks * col_blocks;
            if (task_idx < num_tasks) {
                const int2 swizzled = get_swizzled_2d_idx<config::SUPERGROUP_SIZE>(row_blocks, col_blocks, task_idx);
                tile_coord = {first_row_block + swizzled.x, swizzled.y, expert_idx};
                break;
            }
            task_idx -= num_tasks;
            row_block_offset += expert_row_blocks;
        }
    }
    if (tile_coord.z < 0) return;

    if (warpgroup::groupid() == config::NUM_CONSUMERS) {
        if (warpgroup::warpid() == 3 && warp::elect_leader()) {
            if (kind == expert_gemm_kind::DOWN) {
                const int expected = (config::MLP_Mb / config::SWIGLU_Mb) * (hidden_gmem.cols() / config::SWIGLU_Nb);
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
                const int minibatch_first_row = minibatch_idx * g.minibatch_size;
                const int minibatch_rows = max(0, min(g.minibatch_size, num_tokens - minibatch_first_row));
                const int expected = ((minibatch_rows + config::DISPATCH_COMBINE_Mb - 1) / config::DISPATCH_COMBINE_Mb) * (g.x_routed.cols() / config::DISPATCH_COMBINE_Nb);
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
                input_ring = ring_advance<config::LOAD_PIPE_DEPTH>(input_ring);
            }
        } else if (cta_rank == 0 && warpgroup::warpid() == 0 && warp::elect_leader()) {
            int input_ring = 0;
            wait(gemm_outputs_finished, get_phasebit<1>(gemm_bitfield, config::LOAD_PIPE_DEPTH));
            update_phasebit<1>(gemm_bitfield, config::LOAD_PIPE_DEPTH);
            for (int idx = 0; idx < iters_per_task; ++idx) {
                tma::expect_bytes(gemm_inputs_arrived[input_ring], config::CLUSTER_SIZE * sizeof(typename globals::a_tile) + 2 * sizeof(typename globals::b_tile));
                wait(gemm_inputs_arrived[input_ring], get_phasebit<0>(gemm_bitfield, input_ring));
                if (idx == 0) mm2_ABt (d_tt, a_smem[input_ring], b_smem[input_ring], gemm_inputs_finished[input_ring]);
                else          mma2_ABt(d_tt, a_smem[input_ring], b_smem[input_ring], gemm_inputs_finished[input_ring]);
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
                asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}" :: "l"(&g.mlp_swiglu_counter[{gate_counter_offset + tile_coord.x * col_blocks + tile_coord.y}]), "r"(1) : "memory");
            } else if constexpr (!IS_SHARED) {
                // Routed down is complete; signal combine
                tma::store_async_wait();
                asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}" :: "l"(&g.combine_counter[{minibatch_idx}]), "r"(1) : "memory");
            }
        }
    }
}

static __device__ __forceinline__ void dispatch_mlp_swiglu_combine_kernel(const globals &g) {
    warpgroup::increase_registers<256>();

    extern __shared__ int __shm[];
    const uint64_t smem_base_addr = (reinterpret_cast<uint64_t>(&__shm[0]) + 1023) & ~uint64_t(1023);

    int cluster_idx = clusterIdx().x;
    const int cta_rank = cluster_ctarank();
    const int shared_row_blocks = g.x_shared.rows() / config::MLP_Mb;
    const int minibatch_routed_row_blocks = g.minibatch_size / config::MLP_Mb;
    const int shared_gate_up_tasks = shared_row_blocks * (g.w_shared_gate.rows() / config::MLP_Nb);
    const int minibatch_routed_gate_up_tasks = minibatch_routed_row_blocks * (g.w_routed_gate.rows() / config::MLP_Nb);
    const int shared_swiglu_tiles = (g.hidden_shared.rows() / config::SWIGLU_Mb) * (g.hidden_shared.cols() / config::SWIGLU_Nb);
    const int minibatch_routed_swiglu_tiles = (g.minibatch_size / config::SWIGLU_Mb) * (g.hidden_routed.cols() / config::SWIGLU_Nb);
    const int shared_swiglu_tasks = (shared_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_PIPE_DEPTH);
    const int minibatch_routed_swiglu_tasks = (minibatch_routed_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_PIPE_DEPTH);
    const int shared_down_tasks = shared_row_blocks * (g.w_shared_down.rows() / config::MLP_Nb);
    const int minibatch_routed_down_tasks = minibatch_routed_row_blocks * (g.w_routed_down.rows() / config::MLP_Nb);
    const int shared_tasks = 2 * shared_gate_up_tasks + shared_swiglu_tasks + shared_down_tasks;
    const int minibatch_tasks = 2 * minibatch_routed_gate_up_tasks + minibatch_routed_swiglu_tasks + minibatch_routed_down_tasks;
    const int comm_clusters = g.num_comm_sms / config::CLUSTER_SIZE;
    uint32_t gemm_bitfield = 0xFFFF0000;
    uint32_t swiglu_bitfield = 0xFFFF0000;
    uint32_t dispatch_combine_bitfield = 0xFFFF0000;

    __shared__ clc::handle clc_handle[config::CLC_PIPE_DEPTH];
    __shared__ semaphore schedule_arrived[config::CLC_PIPE_DEPTH], schedule_finished[config::CLC_PIPE_DEPTH];
    __shared__ semaphore swiglu_inputs_arrived[config::SWIGLU_PIPE_DEPTH];
    __shared__ semaphore gemm_inputs_arrived[config::LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_inputs_finished[config::LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_outputs_arrived, gemm_outputs_finished;
    __shared__ semaphore dispatch_combine_inputs_arrived[config::DISPATCH_COMBINE_PIPE_DEPTH];

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < config::SWIGLU_PIPE_DEPTH; ++i) {
            init_semaphore(swiglu_inputs_arrived[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < config::LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(gemm_inputs_arrived[i], 0, 1);
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
        for (int i = 0; i < config::DISPATCH_COMBINE_PIPE_DEPTH; ++i) {
            init_semaphore(dispatch_combine_inputs_arrived[i], 0, 1);
        }
    }

    tensor_allocator<1, config::CLUSTER_SIZE> tm_alloc{};
    tt<float, config::MLP_Mb / 2, config::MLP_Nb> d_tt = tm_alloc.template allocate<tt<float, config::MLP_Mb / 2, config::MLP_Nb>>(0);
    everyone::tma::cluster::sync();

    if (cluster_idx < comm_clusters) {
        const int comm_cta_idx = cluster_idx * config::CLUSTER_SIZE + cta_rank;
        const int dispatch_combine_tiles = (g.x_routed.rows() / config::DISPATCH_COMBINE_Mb) * (g.x_routed.cols() / config::DISPATCH_COMBINE_Nb);
        const int dispatch_combine_tasks = (dispatch_combine_tiles + config::DISPATCH_COMBINE_PIPE_DEPTH - 1) / config::DISPATCH_COMBINE_PIPE_DEPTH;
        for (int task_idx = comm_cta_idx; task_idx < dispatch_combine_tasks; task_idx += g.num_comm_sms)
            dispatch_combine_kernel<true>(g, dispatch_combine_inputs_arrived, dispatch_combine_bitfield, task_idx, smem_base_addr);
        for (int task_idx = comm_cta_idx; task_idx < dispatch_combine_tasks; task_idx += g.num_comm_sms)
            dispatch_combine_kernel<false>(g, dispatch_combine_inputs_arrived, dispatch_combine_bitfield, task_idx, smem_base_addr);
        return;
    }

    bool current_is_swiglu;
    for (int task_iter = 0; cluster_idx >= 0; ++task_iter) {
        const int clc_stage = task_iter % config::CLC_PIPE_DEPTH;
        if (warpgroup::groupid() == config::NUM_CONSUMERS && warpgroup::warpid() == 1 && warp::elect_leader()) { // warp not used by the gemms
            if (cta_rank == 0) {
                wait(schedule_finished[clc_stage], ((task_iter + config::CLC_PIPE_DEPTH) / config::CLC_PIPE_DEPTH) % 2);
                clc::schedule(clc_handle[clc_stage], schedule_arrived[clc_stage]);
            }
            tma::expect_bytes(schedule_arrived[clc_stage], sizeof(clc_handle[clc_stage]));
        }

        const int comp_cluster_idx = cluster_idx - comm_clusters;
        if (comp_cluster_idx < shared_gate_up_tasks) {
            // Shared gate
            current_is_swiglu = false;
            const int task_idx = comp_cluster_idx;
            expert_grouped_gemm<true>(g, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished,
                                      gemm_bitfield, cta_rank, 0, task_idx, smem_base_addr, expert_gemm_kind::GATE, d_tt);
        } else if (comp_cluster_idx < shared_gate_up_tasks * 2) {
            // Shared up
            current_is_swiglu = false;
            const int task_idx = comp_cluster_idx - shared_gate_up_tasks;
            expert_grouped_gemm<true>(g, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished,
                                      gemm_bitfield, cta_rank, 0, task_idx, smem_base_addr, expert_gemm_kind::UP, d_tt);
        } else if (comp_cluster_idx < shared_gate_up_tasks * 2 + shared_swiglu_tasks) {
            // Shared Swiglu
            current_is_swiglu = true;
            const int task_idx = comp_cluster_idx - shared_gate_up_tasks * 2;
            swiglu<true>(g, swiglu_inputs_arrived, swiglu_bitfield, cta_rank, 0, task_idx, smem_base_addr);
        } else if (comp_cluster_idx < shared_tasks) {
            // Shared down
            current_is_swiglu = false;
            const int task_idx = comp_cluster_idx - shared_gate_up_tasks * 2 - shared_swiglu_tasks;
            expert_grouped_gemm<true>(g, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished,
                                      gemm_bitfield, cta_rank, 0, task_idx, smem_base_addr, expert_gemm_kind::DOWN, d_tt);
        } else {
            // Routed expert with minibatching
            const int minibatch_idx = (comp_cluster_idx - shared_tasks) / minibatch_tasks;
            const int minibatch_task_idx = (comp_cluster_idx - shared_tasks) - minibatch_idx * minibatch_tasks;

            if (minibatch_task_idx < minibatch_routed_gate_up_tasks) {
                // Routed gate
                current_is_swiglu = false;
                const int task_idx = minibatch_task_idx;
                expert_grouped_gemm<false>(g, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished,
                                           gemm_bitfield, cta_rank, minibatch_idx, task_idx, smem_base_addr, expert_gemm_kind::GATE, d_tt);
            } else if (minibatch_task_idx < minibatch_routed_gate_up_tasks * 2) {
                // Routed up
                current_is_swiglu = false;
                const int task_idx = minibatch_task_idx - minibatch_routed_gate_up_tasks;
                expert_grouped_gemm<false>(g, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished,
                                           gemm_bitfield, cta_rank, minibatch_idx, task_idx, smem_base_addr, expert_gemm_kind::UP, d_tt);
            } else if (minibatch_task_idx < minibatch_routed_gate_up_tasks * 2 + minibatch_routed_swiglu_tasks) {
                // Routed Swiglu
                current_is_swiglu = true;
                const int task_idx = minibatch_task_idx - minibatch_routed_gate_up_tasks * 2;
                swiglu<false>(g, swiglu_inputs_arrived, swiglu_bitfield, cta_rank, minibatch_idx, task_idx, smem_base_addr);
            } else {
                // Routed down
                current_is_swiglu = false;
                const int task_idx = minibatch_task_idx - minibatch_routed_gate_up_tasks * 2 - minibatch_routed_swiglu_tasks;
                expert_grouped_gemm<false>(g, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished,
                                           gemm_bitfield, cta_rank, minibatch_idx, task_idx, smem_base_addr, expert_gemm_kind::DOWN, d_tt);
            }
        }

        wait(schedule_arrived[clc_stage], (task_iter / config::CLC_PIPE_DEPTH) % 2);
        const auto schedule = clc::query(clc_handle[clc_stage]);
        cluster_idx = schedule.success ? static_cast<int>(schedule.x / config::CLUSTER_SIZE) : -1;
        __syncwarp();
        warp::tma::cluster::arrive(schedule_finished[clc_stage], 0);

        // SWIGLU -> GEMM requires a cluster-wide sync
        const int next_comp_cluster_idx = cluster_idx - comm_clusters;
        const int next_minibatch_task_idx = (next_comp_cluster_idx - shared_tasks) - ((next_comp_cluster_idx - shared_tasks) / minibatch_tasks) * minibatch_tasks;
        const bool next_is_shared_swiglu = next_comp_cluster_idx >= shared_gate_up_tasks * 2 && next_comp_cluster_idx < shared_gate_up_tasks * 2 + shared_swiglu_tasks;
        const bool next_is_routed_swiglu = next_comp_cluster_idx >= shared_tasks && next_minibatch_task_idx >= minibatch_routed_gate_up_tasks * 2 && next_minibatch_task_idx < minibatch_routed_gate_up_tasks * 2 + minibatch_routed_swiglu_tasks;
        if (current_is_swiglu && cluster_idx >= 0 && !next_is_shared_swiglu && !next_is_routed_swiglu)
            everyone::tma::cluster::sync();
    }
}

static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, 
                           at::Tensor, at::Tensor, at::Tensor, 
                           at::Tensor, at::Tensor, at::Tensor> 
dispatch_mlp_swiglu_combine(
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
    int topk,
    int num_comm_sms,
    int minibatch_size
) {
    const int num_local_tokens = x.size(0);
    const int buffer_capacity = dispatch_buffer.size(0);
    const int model_dim = x.size(1);
    const int num_minibatches = (buffer_capacity + minibatch_size - 1) / minibatch_size;
    const int shared_row_blocks = num_local_tokens / config::MLP_Mb;
    const int routed_row_blocks = buffer_capacity / config::MLP_Mb;
    const int shared_gate_up_tasks = shared_row_blocks * (w_shared_gate.size(0) / config::MLP_Nb);
    const int routed_gate_up_tasks = routed_row_blocks * (w_routed_gate.size(1) / config::MLP_Nb);

    bf16 *x_routed_send_buffer_data[NUM_DEVICES];
    bf16 *y_routed_recv_buffer_data[NUM_DEVICES];
    for (int i = 0; i < NUM_DEVICES; ++i) {
        x_routed_send_buffer_data[i] = reinterpret_cast<bf16*>(x_ptrs[i]);
        y_routed_recv_buffer_data[i] = reinterpret_cast<bf16*>(combine_buffer_ptrs[i]);
    }
    
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

    globals g {
        .x_shared = kittens::py::tensor_to_gl<typename globals::activation_gl>(x),
        .x_routed = kittens::py::tensor_to_gl<typename globals::activation_gl>(dispatch_buffer),
        .gate_shared = kittens::py::tensor_to_gl<typename globals::activation_gl>(gate_shared),
        .gate_routed = kittens::py::tensor_to_gl<typename globals::activation_gl>(gate_routed),
        .up_shared = kittens::py::tensor_to_gl<typename globals::activation_gl>(up_shared),
        .up_routed = kittens::py::tensor_to_gl<typename globals::activation_gl>(up_routed),
        .hidden_shared = kittens::py::tensor_to_gl<typename globals::activation_gl>(hidden_shared),
        .hidden_routed = kittens::py::tensor_to_gl<typename globals::activation_gl>(hidden_routed),
        .y_shared = kittens::py::tensor_to_gl<typename globals::activation_gl>(y_shared),
        .y_routed = kittens::py::tensor_to_gl<typename globals::activation_gl>(y_routed),
        .x_routed_send_buffer = typename globals::activation_pgl{x_routed_send_buffer_data, nullptr, nullptr, static_cast<size_t>(num_local_tokens), static_cast<size_t>(model_dim)},
        .y_routed_recv_buffer = typename globals::activation_pgl{y_routed_recv_buffer_data, nullptr, nullptr, static_cast<size_t>(num_local_tokens * topk), static_cast<size_t>(model_dim)},
        .w_shared_gate = kittens::py::tensor_to_gl<typename globals::weight_gl>(w_shared_gate),
        .w_routed_gate = kittens::py::tensor_to_gl<typename globals::weight_gl>(w_routed_gate),
        .w_shared_up = kittens::py::tensor_to_gl<typename globals::weight_gl>(w_shared_up),
        .w_routed_up = kittens::py::tensor_to_gl<typename globals::weight_gl>(w_routed_up),
        .w_shared_down = kittens::py::tensor_to_gl<typename globals::weight_gl>(w_shared_down),
        .w_routed_down = kittens::py::tensor_to_gl<typename globals::weight_gl>(w_routed_down),
        .schedule_peer_rank = kittens::py::tensor_to_gl<typename globals::index_gl>(schedule_peer_rank),
        .schedule_peer_token_idx = kittens::py::tensor_to_gl<typename globals::index_gl>(schedule_peer_token_idx),
        .tokens_per_expert = kittens::py::tensor_to_gl<typename globals::index_gl>(tokens_per_expert),
        .mlp_swiglu_counter = kittens::py::tensor_to_gl<typename globals::index_gl>(mlp_swiglu_counter),
        .dispatch_counter = kittens::py::tensor_to_gl<typename globals::index_gl>(dispatch_counter),
        .combine_counter = kittens::py::tensor_to_gl<typename globals::index_gl>(combine_counter),
        .topk = topk,
        .num_comm_sms = num_comm_sms,
        .minibatch_size = minibatch_size
    };

    kittens::py::launch_kernel<config, globals, dispatch_mlp_swiglu_combine_kernel>(g);

    return {gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, y_shared, y_routed, combine_buffer};
}

}; // struct dispatch_mlp_swiglu_combiner

PYBIND11_MODULE(_C, m) {
    m.def("schedule", &scheduler::schedule, "",
          pybind11::arg("topk_all"), pybind11::arg("num_local_experts"), pybind11::arg("capacity"), pybind11::arg("rank"));
    m.def("dispatch_mlp_swiglu_combine", &dispatch_mlp_swiglu_combiner<4>::dispatch_mlp_swiglu_combine, "",
          pybind11::arg("x"), pybind11::arg("x_ptrs"),
          pybind11::arg("dispatch_buffer"), pybind11::arg("combine_buffer"), pybind11::arg("combine_buffer_ptrs"),
          pybind11::arg("w_shared_gate"), pybind11::arg("w_routed_gate"),
          pybind11::arg("w_shared_up"), pybind11::arg("w_routed_up"),
          pybind11::arg("w_shared_down"), pybind11::arg("w_routed_down"),
          pybind11::arg("schedule_peer_rank"), pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("tokens_per_expert"),
          pybind11::arg("topk"), pybind11::arg("num_comm_sms"), pybind11::arg("minibatch_size"));
}
