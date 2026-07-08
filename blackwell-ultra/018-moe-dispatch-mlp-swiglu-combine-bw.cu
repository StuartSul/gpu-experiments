#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>

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

template <int NUM_DEVICES>
struct dispatch_mlp_swiglu_combiner {

struct config {
    // Grouped GEMM
    static constexpr int MLP_Mb = 256;
    static constexpr int MLP_Nb = 256;
    static constexpr int MLP_Kb = 64;
    static constexpr int MLP_SUPERGROUP_SIZE = 8;
    static constexpr int MLP_LOAD_PIPE_DEPTH = 5;
    static constexpr int MLP_EPI_PIPE_DEPTH = 4;
    static constexpr int MLP_NUM_D_TILES = MLP_EPI_PIPE_DEPTH > 1 ? 2 : 1;

    // Fused SwiGLU
    static constexpr int SWIGLU_Mb = 128;
    static constexpr int SWIGLU_Nb = 128;
    static constexpr int SWIGLU_FWD_PIPE_DEPTH = 3; // gate / up
    static constexpr int SWIGLU_BWD_PIPE_DEPTH = 2; // gate / up / d_hidden

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

// Grouped GEMM tiles
using mlp_a_tile = st_bf<config::MLP_Mb / 2, config::MLP_Kb>;
using mlp_b_tile = st_bf<config::MLP_Nb / 2, config::MLP_Kb>;
using mlp_a_t_tile = st_bf<config::MLP_Kb, config::MLP_Mb / 2>;
using mlp_b_t_tile = st_bf<config::MLP_Kb, config::MLP_Nb / 2>;
using mlp_d_tile = st_bf<config::MLP_Mb / 2, config::MLP_Nb / config::MLP_EPI_PIPE_DEPTH>;

// Fused SwiGLU tiles
using swiglu_tile = st_bf<config::SWIGLU_Mb, config::SWIGLU_Nb>;

// Dispatch/Combine tiles
using dispatch_combine_vec = sv_bf<config::DISPATCH_COMBINE_Nb>;

// Global layouts
using activation_gl = gl<bf16, 1, 1, -1, -1, mlp_a_tile, mlp_a_t_tile, swiglu_tile, dispatch_combine_vec>;
using activation_pgl = pgl<activation_gl, NUM_DEVICES, false>;
using weight_gl = gl<bf16, 1, -1, -1, -1, mlp_b_tile>;
using index_gl = gl<int, 1, 1, 1, -1>;

struct globals_fwd {
    activation_gl x_shared;              // (num_local_tokens, H)
    activation_gl x_routed;              // (macrobatch_size, H)
    activation_gl gate_shared;           // (num_local_tokens, I)
    activation_gl gate_routed;           // (macrobatch_size, I)
    activation_gl up_shared;             // (num_local_tokens, I)
    activation_gl up_routed;             // (macrobatch_size, I)
    activation_gl hidden_shared;         // (num_local_tokens, I)
    activation_gl hidden_routed;         // (macrobatch_size, I)
    activation_gl y_shared;              // (num_local_tokens, H)
    activation_gl y_routed;              // (macrobatch_size, H)

    activation_pgl x_routed_send_buffer; // (num_local_tokens, H)
    activation_pgl y_routed_recv_buffer; // (num_local_tokens * topk, H)

    weight_gl w_shared_gate;             // (I, H)
    weight_gl w_routed_gate;             // (num_local_experts, I, H)
    weight_gl w_shared_up;               // (I, H)
    weight_gl w_routed_up;               // (num_local_experts, I, H)
    weight_gl w_shared_down;             // (H, I)
    weight_gl w_routed_down;             // (num_local_experts, H, I)

    index_gl schedule_peer_rank;         // (schedule_capacity,)
    index_gl schedule_peer_token_idx;    // (schedule_capacity,)
    index_gl num_tokens;                 // (1,)
    index_gl tokens_per_expert;          // (num_local_experts,)

    index_gl gate_up_tile_ready;         // (shared_gate_up_tasks + routed_gate_up_tasks,)
    index_gl hidden_row_block_ready;     // (shared_row_blocks + routed_row_blocks,)
    index_gl x_routed_ready;             // (num_minibatches,)
    index_gl y_routed_ready;             // (num_minibatches,)

    const int topk;
    const int num_comm_sms;
    const int macrobatch_size;
    const int minibatch_size;

    __host__ inline dim3 grid() const {
        const int num_minibatches = (schedule_peer_rank.cols() + minibatch_size - 1) / minibatch_size; // across all macrobatches
        const int shared_row_blocks = x_shared.rows() / config::MLP_Mb;
        const int minibatch_routed_row_blocks = minibatch_size / config::MLP_Mb;
        const int shared_gate_up_tasks = shared_row_blocks * (w_shared_gate.rows() / config::MLP_Nb);
        const int minibatch_routed_gate_up_tasks = minibatch_routed_row_blocks * (w_routed_gate.rows() / config::MLP_Nb);
        const int shared_swiglu_tiles = (hidden_shared.rows() / config::SWIGLU_Mb) * (hidden_shared.cols() / config::SWIGLU_Nb);
        const int minibatch_routed_swiglu_tiles = (minibatch_size / config::SWIGLU_Mb) * (hidden_routed.cols() / config::SWIGLU_Nb);
        const int shared_swiglu_tasks = (shared_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH);
        const int minibatch_routed_swiglu_tasks = (minibatch_routed_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH);
        const int shared_down_tasks = shared_row_blocks * (w_shared_down.rows() / config::MLP_Nb);
        const int minibatch_routed_down_tasks = minibatch_routed_row_blocks * (w_routed_down.rows() / config::MLP_Nb);
        const int shared_tasks = 2 * shared_gate_up_tasks + shared_swiglu_tasks + shared_down_tasks;
        const int minibatch_tasks = 2 * minibatch_routed_gate_up_tasks + minibatch_routed_swiglu_tasks + minibatch_routed_down_tasks;
        return dim3(config::CLUSTER_SIZE * (shared_tasks + num_minibatches * minibatch_tasks) + num_comm_sms);
    }
};

struct globals_bwd {
    // Saved/replayed forward inputs
    activation_gl x_shared;                // (num_local_tokens, H)
    activation_gl x_routed;                // (macrobatch_size, H)
    activation_gl gate_shared;             // (num_local_tokens, I)
    activation_gl gate_routed;             // (macrobatch_size, I)
    activation_gl up_shared;               // (num_local_tokens, I)
    activation_gl up_routed;               // (macrobatch_size, I)
    activation_gl hidden_shared;           // (num_local_tokens, I)
    activation_gl hidden_routed;           // (macrobatch_size, I)

    // Gradients
    activation_gl d_y_shared;              // (num_local_tokens, H)
    activation_gl d_y_routed;              // (macrobatch_size, H)
    activation_gl d_hidden_shared;         // (num_local_tokens, I)
    activation_gl d_hidden_routed;         // (macrobatch_size, I)
    activation_gl d_gate_shared;           // (num_local_tokens, I)
    activation_gl d_gate_routed;           // (macrobatch_size, I)
    activation_gl d_up_shared;             // (num_local_tokens, I)
    activation_gl d_up_routed;             // (macrobatch_size, I)
    activation_gl d_x_shared;              // (num_local_tokens, H)
    activation_gl d_x_routed;              // (macrobatch_size, H)

    // Symmetric buffers
    activation_pgl x_routed_send_buffer;   // (num_local_tokens, H)
    activation_pgl d_combine_buffer;       // (num_local_tokens * topk, H)
    activation_pgl d_x_routed_buffer;      // (num_local_tokens * topk, H)

    // Weights
    weight_gl w_routed_gate;               // (num_local_experts, I, H)
    weight_gl w_routed_up;                 // (num_local_experts, I, H)
    weight_gl w_shared_gate_T;             // (H, I)
    weight_gl w_routed_gate_T;             // (num_local_experts, H, I)
    weight_gl w_shared_up_T;               // (H, I)
    weight_gl w_routed_up_T;               // (num_local_experts, H, I)
    weight_gl w_shared_down_T;             // (I, H)
    weight_gl w_routed_down_T;             // (num_local_experts, I, H)

    // Weight gradients (reduce-added, must be zero-initialized)
    weight_gl d_w_shared_gate;             // (I, H)
    weight_gl d_w_routed_gate;             // (num_local_experts, I, H)
    weight_gl d_w_shared_up;               // (I, H)
    weight_gl d_w_routed_up;               // (num_local_experts, I, H)
    weight_gl d_w_shared_down;             // (H, I)
    weight_gl d_w_routed_down;             // (num_local_experts, H, I)

    index_gl schedule_peer_rank;           // (schedule_capacity,)
    index_gl schedule_peer_token_idx;      // (schedule_capacity,)
    index_gl num_tokens;                   // (1,)
    index_gl tokens_per_expert;            // (num_local_experts,)

    index_gl d_y_routed_ready;             // (num_minibatches,) reverse-combine -> dgrad/wgrad down
    index_gl d_hidden_ready;               // (shared + routed dgrad-down tasks,) dgrad down -> swiglu bwd
    index_gl d_gate_up_ready;              // (shared + routed row blocks,) swiglu bwd -> dgrad gate/up and wgrad gate/up
    index_gl d_x_routed_ready;             // (num_minibatches,) dgrad gate/up -> reverse-dispatch
    index_gl replayed_x_routed_ready;      // (num_minibatches,) replayed dispatch -> replayed gate/up and wgrad gate/up
    index_gl replayed_gate_up_ready;       // (routed gate/up tasks,) replayed gate/up -> replayed swiglu and bwd swiglu
    index_gl replayed_hidden_ready;        // (routed row blocks,) replayed swiglu -> wgrad down
    index_gl routed_buffers_done;          // (num_macrobatches,) current macrobatch -> next macrobatch

    const int topk;
    const int num_comm_sms;
    const int macrobatch_size;
    const int minibatch_size;

    __host__ inline dim3 grid() const {
        const int capacity = schedule_peer_rank.cols();
        const int num_minibatches = (capacity + minibatch_size - 1) / minibatch_size; // across all macrobatches
        const int num_macrobatches = (capacity + macrobatch_size - 1) / macrobatch_size;
        const int shared_row_blocks = d_y_shared.rows() / config::MLP_Mb;
        const int minibatch_row_blocks = minibatch_size / config::MLP_Mb;
        const int intermediate_dim_col_blocks = hidden_shared.cols() / config::MLP_Nb;
        const int hidden_dim_col_blocks = d_y_shared.cols() / config::MLP_Nb;
        const int shared_swiglu_bwd_tiles = (hidden_shared.rows() / config::SWIGLU_Mb) * (hidden_shared.cols() / config::SWIGLU_Nb);
        const int minibatch_swiglu_tiles = (minibatch_size / config::SWIGLU_Mb) * (hidden_routed.cols() / config::SWIGLU_Nb);
        const int shared_swiglu_bwd_tasks = (shared_swiglu_bwd_tiles + config::CLUSTER_SIZE * config::SWIGLU_BWD_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_BWD_PIPE_DEPTH);
        const int minibatch_swiglu_bwd_tasks = (minibatch_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_BWD_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_BWD_PIPE_DEPTH);
        const int minibatch_swiglu_fwd_tasks = (minibatch_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH);
        const int shared_tasks = shared_row_blocks * intermediate_dim_col_blocks + shared_swiglu_bwd_tasks + shared_row_blocks * hidden_dim_col_blocks + 3 * intermediate_dim_col_blocks * hidden_dim_col_blocks;
        const int minibatch_bwd_tasks = minibatch_row_blocks * intermediate_dim_col_blocks + minibatch_swiglu_bwd_tasks + minibatch_row_blocks * hidden_dim_col_blocks;
        const int minibatch_replay_tasks = 2 * minibatch_row_blocks * intermediate_dim_col_blocks + minibatch_swiglu_fwd_tasks;
        const int num_replay_minibatches = (num_macrobatches - 1) * (macrobatch_size / minibatch_size);
        const int wgrad_tasks = 3 * w_routed_gate.depth() * intermediate_dim_col_blocks * hidden_dim_col_blocks;
        return dim3(config::CLUSTER_SIZE * (shared_tasks + num_minibatches * minibatch_bwd_tasks + num_replay_minibatches * minibatch_replay_tasks + num_macrobatches * wgrad_tasks) + num_comm_sms);
    }
};

static __device__ __forceinline__ void barrier_wait(const index_gl &counter, int index, int required_count) {
    int value;
    while (true) {
        asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}" : "=r"(value) : "l"(&counter[{index}]) : "memory");
        if (value >= required_count) break;
        __nanosleep(16);
    }
    asm volatile("{fence.acquire.gpu;}" ::: "memory");
}

static __device__ __forceinline__ void barrier_arrive(const index_gl &counter, int index, int increment = 1) {
    asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}" :: "l"(&counter[{index}]), "r"(increment) : "memory");
}

template <bool IS_PULL>
static __device__ __forceinline__ void dispatch_combine_kernel(
    const activation_pgl &peer_buf,
    const activation_gl &local_buf,
    const index_gl &schedule_peer_rank,
    const index_gl &schedule_peer_token_idx,
    const index_gl *transfer_ready,
    const index_gl *transfer_done,
    semaphore (&inputs_arrived)[config::DISPATCH_COMBINE_PIPE_DEPTH],
    uint32_t &bitfield,
    const int num_tokens,
    const int macrobatch_size,
    const int minibatch_size,
    const int macrobatch_idx,
    const int task_idx,
    const int row_divisor,
    const int transfer_ready_index,
    const int transfer_ready_required_count,
    const uint64_t smem_base_addr
) {
    auto &token_vecs = *reinterpret_cast<dispatch_combine_vec (*)[config::DISPATCH_COMBINE_PIPE_DEPTH][config::DISPATCH_COMBINE_Mb]>(smem_base_addr);

    const int tid = threadIdx.x;
    const bool is_worker = tid < config::DISPATCH_COMBINE_Mb; // only these threads move tokens, but all threads join the barriers and waits

    const int col_blocks = local_buf.cols() / config::DISPATCH_COMBINE_Nb;
    const int first_tile_idx = task_idx * config::DISPATCH_COMBINE_PIPE_DEPTH;

    const int macrobatch_offset = macrobatch_idx * macrobatch_size;
    const int num_macrobatch_tokens = min(macrobatch_size, num_tokens - macrobatch_offset);
    const int num_valid_tiles = min(config::DISPATCH_COMBINE_PIPE_DEPTH, num_macrobatch_tokens / config::DISPATCH_COMBINE_Mb * col_blocks - first_tile_idx); // because we pad to 256
    if (num_valid_tiles <= 0) return;

    const int first_row_idx = first_tile_idx / col_blocks * config::DISPATCH_COMBINE_Mb + tid;
    const int first_col_block_idx = first_tile_idx % col_blocks;

    int row_idx[config::DISPATCH_COMBINE_PIPE_DEPTH], col_block_idx[config::DISPATCH_COMBINE_PIPE_DEPTH], peer_rank[config::DISPATCH_COMBINE_PIPE_DEPTH], 
        peer_token_idx[config::DISPATCH_COMBINE_PIPE_DEPTH], num_valid_rows[config::DISPATCH_COMBINE_PIPE_DEPTH];
    #pragma unroll
    for (int stage = 0, row = first_row_idx, col = first_col_block_idx; stage < config::DISPATCH_COMBINE_PIPE_DEPTH; ++stage) {
        const bool is_valid_tile = stage < num_valid_tiles;
        row_idx[stage] = row;
        col_block_idx[stage] = col;
        peer_rank[stage] = is_valid_tile && is_worker ? schedule_peer_rank[{macrobatch_offset + row}] : -1;
        peer_token_idx[stage] = is_valid_tile && is_worker ? schedule_peer_token_idx[{macrobatch_offset + row}] : -1;
        num_valid_rows[stage] = !is_valid_tile ? 0
                              : (stage == 0 || col == 0) ? __syncthreads_count(peer_rank[stage] >= 0)
                              : num_valid_rows[stage - 1];
        if (++col == col_blocks) { col = 0; row += config::DISPATCH_COMBINE_Mb; }
    }

    if (tid == 0) {
        if constexpr (IS_PULL) {
            if (transfer_ready != nullptr) barrier_wait(*transfer_ready, transfer_ready_index, transfer_ready_required_count);
        } else {
            // Wait until the GEMMs have fully written every minibatch this task reads
            const int first_global_minibatch_idx = (macrobatch_offset + first_row_idx) / minibatch_size;
            const int last_global_minibatch_idx = (macrobatch_offset + (first_tile_idx + num_valid_tiles - 1) / col_blocks * config::DISPATCH_COMBINE_Mb) / minibatch_size;
            for (int global_minibatch_idx = first_global_minibatch_idx; global_minibatch_idx <= last_global_minibatch_idx; ++global_minibatch_idx) {
                const int minibatch_rows = min(minibatch_size, num_tokens - global_minibatch_idx * minibatch_size);
                const int required_count = ((minibatch_rows + config::MLP_Mb - 1) / config::MLP_Mb) * (local_buf.cols() / config::MLP_Nb) * config::CLUSTER_SIZE;
                barrier_wait(*transfer_ready, global_minibatch_idx, required_count);
            }
        }
        #pragma unroll
        for (int stage = 0; stage < config::DISPATCH_COMBINE_PIPE_DEPTH; ++stage)
            if (stage < num_valid_tiles)
                tma::expect_bytes(inputs_arrived[stage], num_valid_rows[stage] * sizeof(dispatch_combine_vec)); // 0 bytes completes the phase immediately
    }
    __syncthreads();

    #pragma unroll
    for (int stage = 0; stage < config::DISPATCH_COMBINE_PIPE_DEPTH; ++stage) {
        if (peer_rank[stage] >= 0) {
            if constexpr (IS_PULL)
                tma::load_async(token_vecs[stage][tid], peer_buf[peer_rank[stage]], {peer_token_idx[stage] / row_divisor, col_block_idx[stage]}, inputs_arrived[stage]);
            else
                tma::load_async(token_vecs[stage][tid], local_buf, {row_idx[stage], col_block_idx[stage]}, inputs_arrived[stage]);
        } else if (IS_PULL && is_worker && stage < num_valid_tiles) { // zero-fill padding rows
            auto *vec = reinterpret_cast<float4 *>(&token_vecs[stage][tid]);
            #pragma unroll
            for (int i = 0; i < sizeof(dispatch_combine_vec) / sizeof(float4); ++i)
                vec[i] = float4{0.0f, 0.0f, 0.0f, 0.0f};
        }
    }

    // Store each tile out as its loads arrive
    #pragma unroll
    for (int stage = 0; stage < config::DISPATCH_COMBINE_PIPE_DEPTH; ++stage) {
        if (stage < num_valid_tiles) {
            wait(inputs_arrived[stage], get_phasebit<0>(bitfield, stage)); // semaphores are reused across tasks
            update_phasebit<0>(bitfield, stage);
            if constexpr (IS_PULL) { // store padding rows too
                if (is_worker) tma::store_async(local_buf, token_vecs[stage][tid], {row_idx[stage], col_block_idx[stage]});
            } else if (peer_rank[stage] >= 0) {
                tma::store_async(peer_buf[peer_rank[stage]], token_vecs[stage][tid], {peer_token_idx[stage] / row_divisor, col_block_idx[stage]});
            }
        }
    }

    if constexpr (IS_PULL) {
        tma::store_async_wait();
        __syncthreads();
        if (tid == 0) {
            const int tiles_per_minibatch = minibatch_size / config::DISPATCH_COMBINE_Mb * col_blocks; // a task straddles at most one minibatch boundary
            const int global_first_tile_idx = macrobatch_offset / config::DISPATCH_COMBINE_Mb * col_blocks + first_tile_idx;
            const int global_minibatch_idx = global_first_tile_idx / tiles_per_minibatch;
            const int first_count = min(num_valid_tiles, (global_minibatch_idx + 1) * tiles_per_minibatch - global_first_tile_idx);
            barrier_arrive(*transfer_done, global_minibatch_idx, first_count);
            if (first_count < num_valid_tiles)
                barrier_arrive(*transfer_done, global_minibatch_idx + 1, num_valid_tiles - first_count);
        }
    } else {
        // The next task on this CTA reuses token_vecs; make sure outgoing stores are done reading shared memory
        tma::store_async_read_wait();
        __syncthreads();
        if (tid == 0 && transfer_done != nullptr)
            barrier_arrive(*transfer_done, macrobatch_idx);
    }
}

template <bool IS_SHARED>
static __device__ __forceinline__ void swiglu_fwd(
    const activation_gl &gate_gmem,
    const activation_gl &up_gmem,
    const activation_gl &hidden_gmem,
    const index_gl &gate_up_tile_ready,
    const index_gl &hidden_row_block_ready,
    semaphore (&swiglu_inputs_arrived)[config::SWIGLU_FWD_PIPE_DEPTH],
    uint32_t &swiglu_bitfield,
    const int num_tokens,
    const int macrobatch_size,
    const int minibatch_size,
    const int macrobatch_idx,
    const int minibatch_idx,
    const int task_idx,
    const int cta_rank,
    const int gate_up_tile_ready_base_index,
    const int hidden_row_block_ready_base_index,
    const uint64_t smem_base_addr
) {
    swiglu_tile (&gate_smem)[config::SWIGLU_FWD_PIPE_DEPTH] = *reinterpret_cast<swiglu_tile (*)[config::SWIGLU_FWD_PIPE_DEPTH]>(smem_base_addr);
    swiglu_tile (&up_smem)[config::SWIGLU_FWD_PIPE_DEPTH] = *reinterpret_cast<swiglu_tile (*)[config::SWIGLU_FWD_PIPE_DEPTH]>(smem_base_addr + sizeof(gate_smem));

    const int intermediate_dim_col_blocks = hidden_gmem.cols() / config::MLP_Nb;
    const int global_minibatch_idx = macrobatch_idx * (macrobatch_size / minibatch_size) + minibatch_idx;
    const int macrobatch_row_block_offset = macrobatch_idx * (macrobatch_size / config::SWIGLU_Mb);

    const int row_blocks = num_tokens / config::SWIGLU_Mb;
    const int col_blocks = hidden_gmem.cols() / config::SWIGLU_Nb;
    const int num_tiles = row_blocks * col_blocks;
    int first_tile_idx, tile_end;
    if constexpr (IS_SHARED) {
        first_tile_idx = task_idx * config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH + cta_rank * config::SWIGLU_FWD_PIPE_DEPTH;
        tile_end = num_tiles;
    } else {
        const int num_tiles_per_minibatch = (minibatch_size / config::SWIGLU_Mb) * col_blocks;
        const int minibatch_first_tile_idx = global_minibatch_idx * num_tiles_per_minibatch;
        first_tile_idx = minibatch_first_tile_idx + task_idx * config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH + cta_rank * config::SWIGLU_FWD_PIPE_DEPTH;
        tile_end = min(num_tiles, minibatch_first_tile_idx + num_tiles_per_minibatch);
    }
    if (first_tile_idx >= tile_end)
        return;

    const int first_row = first_tile_idx / col_blocks;
    const int first_col = first_tile_idx % col_blocks;

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int stage = 0; stage < config::SWIGLU_FWD_PIPE_DEPTH; ++stage) {
            const int tile_idx = first_tile_idx + stage;
            if (tile_idx < tile_end) {
                // This improves throughput
                int row = first_row;
                int col = first_col + stage;
                if (col >= col_blocks) {
                    ++row;
                    col -= col_blocks;
                }
                tma::expect_bytes(swiglu_inputs_arrived[stage], sizeof(gate_smem[stage]) + sizeof(up_smem[stage]));

                const int parent_task_idx = (row / (config::MLP_Mb / config::SWIGLU_Mb)) * intermediate_dim_col_blocks + col / (config::MLP_Nb / config::SWIGLU_Nb);
                barrier_wait(gate_up_tile_ready, gate_up_tile_ready_base_index + parent_task_idx, 2 * config::CLUSTER_SIZE);

                tma::load_async(gate_smem[stage], gate_gmem, {row - macrobatch_row_block_offset, col}, swiglu_inputs_arrived[stage]);
                tma::load_async(up_smem[stage],   up_gmem,   {row - macrobatch_row_block_offset, col}, swiglu_inputs_arrived[stage]);
            }
        }
    }

    using compute_group = group<config::NUM_WARPS>;
    #pragma unroll
    for (int stage = 0; stage < config::SWIGLU_FWD_PIPE_DEPTH; ++stage) {
        const int tile_idx = first_tile_idx + stage;
        if (tile_idx < tile_end) {
            rt_fl<config::SWIGLU_Mb / config::NUM_WARPS, config::SWIGLU_Nb> gate, up, denominator;
            wait(swiglu_inputs_arrived[stage], get_phasebit<0>(swiglu_bitfield, stage));
            update_phasebit<0>(swiglu_bitfield, stage);
            compute_group::load(gate, gate_smem[stage]);
            compute_group::load(up, up_smem[stage]);
            compute_group::mul(denominator, gate, -1.0f);
            compute_group::exp(denominator, denominator);
            compute_group::add(denominator, denominator, 1.0f);
            compute_group::div(gate, gate, denominator);
            compute_group::mul(gate, gate, up);
            compute_group::store(gate_smem[stage], gate);
            __syncthreads();
            if (threadIdx.x == 0) {
                // This improves throughput
                int row = first_row;
                int col = first_col + stage;
                if (col >= col_blocks) {
                    ++row;
                    col -= col_blocks;
                }
                tma::store_async(hidden_gmem, gate_smem[stage], {row - macrobatch_row_block_offset, col});
            }
        }
    }

    if (threadIdx.x == 0) {
        tma::store_async_wait();
        #pragma unroll
        for (int stage = 0; stage < config::SWIGLU_FWD_PIPE_DEPTH; ++stage) {
            const int tile_idx = first_tile_idx + stage;
            if (tile_idx < tile_end) {
                // This improves throughput
                int row = first_row;
                int col = first_col + stage;
                if (col >= col_blocks)
                    ++row;
                barrier_arrive(hidden_row_block_ready, hidden_row_block_ready_base_index + row / (config::MLP_Mb / config::SWIGLU_Mb));
            }
        }
    }
}

template <bool IS_SHARED>
static __device__ __forceinline__ void swiglu_bwd(
    const activation_gl &d_hidden_gmem,
    const activation_gl &gate_gmem,
    const activation_gl &up_gmem,
    const activation_gl &d_gate_gmem,
    const activation_gl &d_up_gmem,
    const index_gl &d_hidden_tile_ready,
    const index_gl *replayed_gate_up_tile_ready,
    const index_gl &d_gate_up_row_block_ready,
    const index_gl *buffer_done,
    semaphore (&swiglu_inputs_arrived)[config::SWIGLU_BWD_PIPE_DEPTH],
    uint32_t &swiglu_bitfield,
    const int num_tokens,
    const int macrobatch_size,
    const int minibatch_size,
    const int macrobatch_idx,
    const int minibatch_idx,
    const int task_idx,
    const int cta_rank,
    const int d_hidden_tile_ready_base_index,
    const int replayed_gate_up_tile_ready_base_index,
    const int d_gate_up_row_block_ready_base_index,
    const int buffer_done_index,
    const uint64_t smem_base_addr
) {
    swiglu_tile (&d_hidden_smem)[config::SWIGLU_BWD_PIPE_DEPTH] = *reinterpret_cast<swiglu_tile (*)[config::SWIGLU_BWD_PIPE_DEPTH]>(smem_base_addr);
    swiglu_tile (&gate_smem)[config::SWIGLU_BWD_PIPE_DEPTH] = *reinterpret_cast<swiglu_tile (*)[config::SWIGLU_BWD_PIPE_DEPTH]>(smem_base_addr + sizeof(d_hidden_smem));
    swiglu_tile (&up_smem)[config::SWIGLU_BWD_PIPE_DEPTH] = *reinterpret_cast<swiglu_tile (*)[config::SWIGLU_BWD_PIPE_DEPTH]>(smem_base_addr + sizeof(d_hidden_smem) + sizeof(gate_smem));

    const int intermediate_dim_col_blocks = gate_gmem.cols() / config::MLP_Nb;
    const int global_minibatch_idx = macrobatch_idx * (macrobatch_size / minibatch_size) + minibatch_idx;
    const int macrobatch_row_block_offset = macrobatch_idx * (macrobatch_size / config::SWIGLU_Mb);

    const int row_blocks = num_tokens / config::SWIGLU_Mb;
    const int col_blocks = gate_gmem.cols() / config::SWIGLU_Nb;
    const int num_tiles = row_blocks * col_blocks;
    int first_tile_idx, tile_end;
    if constexpr (IS_SHARED) {
        first_tile_idx = task_idx * config::CLUSTER_SIZE * config::SWIGLU_BWD_PIPE_DEPTH + cta_rank * config::SWIGLU_BWD_PIPE_DEPTH;
        tile_end = num_tiles;
    } else {
        const int num_tiles_per_minibatch = (minibatch_size / config::SWIGLU_Mb) * col_blocks;
        const int minibatch_first_tile_idx = global_minibatch_idx * num_tiles_per_minibatch;
        first_tile_idx = minibatch_first_tile_idx + task_idx * config::CLUSTER_SIZE * config::SWIGLU_BWD_PIPE_DEPTH + cta_rank * config::SWIGLU_BWD_PIPE_DEPTH;
        tile_end = min(num_tiles, minibatch_first_tile_idx + num_tiles_per_minibatch);
    }
    if (first_tile_idx >= tile_end) {
        if (buffer_done != nullptr && threadIdx.x == 0)
            barrier_arrive(*buffer_done, buffer_done_index);
        return;
    }

    const int first_row = first_tile_idx / col_blocks;
    const int first_col = first_tile_idx % col_blocks;

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int stage = 0; stage < config::SWIGLU_BWD_PIPE_DEPTH; ++stage) {
            const int tile_idx = first_tile_idx + stage;
            if (tile_idx < tile_end) {
                int row = first_row;
                int col = first_col + stage;
                if (col >= col_blocks) {
                    ++row;
                    col -= col_blocks;
                }
                tma::expect_bytes(swiglu_inputs_arrived[stage], sizeof(d_hidden_smem[stage]) + sizeof(gate_smem[stage]) + sizeof(up_smem[stage]));

                const int parent_task_idx = (row / (config::MLP_Mb / config::SWIGLU_Mb)) * intermediate_dim_col_blocks + col / (config::MLP_Nb / config::SWIGLU_Nb);
                barrier_wait(d_hidden_tile_ready, d_hidden_tile_ready_base_index + parent_task_idx, config::CLUSTER_SIZE);
                if (replayed_gate_up_tile_ready != nullptr) // replayed macrobatch: wait for the replayed gate/up GEMMs
                    barrier_wait(*replayed_gate_up_tile_ready, replayed_gate_up_tile_ready_base_index + parent_task_idx, 2 * config::CLUSTER_SIZE);

                tma::load_async(d_hidden_smem[stage], d_hidden_gmem, {row - macrobatch_row_block_offset, col}, swiglu_inputs_arrived[stage]);
                tma::load_async(gate_smem[stage],     gate_gmem,     {row - macrobatch_row_block_offset, col}, swiglu_inputs_arrived[stage]);
                tma::load_async(up_smem[stage],       up_gmem,       {row - macrobatch_row_block_offset, col}, swiglu_inputs_arrived[stage]);
            }
        }
    }

    using compute_group = group<config::NUM_WARPS>;
    #pragma unroll
    for (int stage = 0; stage < config::SWIGLU_BWD_PIPE_DEPTH; ++stage) {
        const int tile_idx = first_tile_idx + stage;
        if (tile_idx < tile_end) {
            wait(swiglu_inputs_arrived[stage], get_phasebit<0>(swiglu_bitfield, stage));
            update_phasebit<0>(swiglu_bitfield, stage);
            rt_fl<config::SWIGLU_Mb / config::NUM_WARPS, config::SWIGLU_Nb> gate, up, d_hidden;
            compute_group::load(gate, gate_smem[stage]);
            compute_group::mul(d_hidden, gate, -1.0f);
            compute_group::exp(d_hidden, d_hidden);
            compute_group::add(d_hidden, d_hidden, 1.0f);         // d_hidden := 1 / sigmoid(gate)
            compute_group::div(gate, gate, d_hidden);             // gate := silu(gate)
            compute_group::mul(up, gate, -1.0f);
            compute_group::add(up, up, 1.0f);
            compute_group::div(up, up, d_hidden);
            compute_group::add(up, up, gate);                     // up := dsilu(gate)
            compute_group::load(d_hidden, d_hidden_smem[stage]);
            compute_group::mul(gate, gate, d_hidden);             // gate := d_up
            compute_group::mul(d_hidden, d_hidden, up);
            compute_group::load(up, up_smem[stage]);
            compute_group::mul(d_hidden, d_hidden, up);           // d_hidden := d_gate
            compute_group::store(gate_smem[stage], d_hidden);     // d_gate overwrites the gate tile in place
            compute_group::store(up_smem[stage], gate);           // d_up overwrites the up tile in place
            __syncthreads();
            if (threadIdx.x == 0) {
                int row = first_row;
                int col = first_col + stage;
                if (col >= col_blocks) {
                    ++row;
                    col -= col_blocks;
                }
                tma::store_async(d_gate_gmem, gate_smem[stage], {row - macrobatch_row_block_offset, col});
                tma::store_async(d_up_gmem, up_smem[stage], {row - macrobatch_row_block_offset, col});
            }
        }
    }

    if (threadIdx.x == 0) {
        tma::store_async_wait();
        #pragma unroll
        for (int stage = 0; stage < config::SWIGLU_BWD_PIPE_DEPTH; ++stage) {
            const int tile_idx = first_tile_idx + stage;
            if (tile_idx < tile_end) {
                int row = first_row;
                int col = first_col + stage;
                if (col >= col_blocks)
                    ++row;
                barrier_arrive(d_gate_up_row_block_ready, d_gate_up_row_block_ready_base_index + row / (config::MLP_Mb / config::SWIGLU_Mb));
            }
        }
        if (buffer_done != nullptr)
            barrier_arrive(*buffer_done, buffer_done_index);
    }
}

template <bool IS_SHARED, bool IS_WGRAD = false>
static __device__ __forceinline__ void expert_grouped_gemm(
    const activation_gl &a_gmem,
    const std::conditional_t<IS_WGRAD, activation_gl, weight_gl> &b_gmem,
    const activation_gl *a2_gmem,
    const weight_gl *b2_gmem,
    const std::conditional_t<IS_WGRAD, weight_gl, activation_gl> &d_gmem,
    const index_gl &tokens_per_expert,
    const index_gl *input_minibatch_ready,  // comms -> GEMM
    const index_gl *input_row_block_ready,  // SwiGLU -> GEMM
    const index_gl *output_tile_ready,      // GEMM -> SwiGLU
    const index_gl *output_minibatch_ready, // GEMM -> comms
    const index_gl *buffer_done,
    tt<float, config::MLP_Mb / 2, config::MLP_Nb> &d_tt,
    semaphore (&gemm_inputs_arrived)[config::MLP_LOAD_PIPE_DEPTH],
    semaphore (&gemm_inputs_finished)[config::MLP_LOAD_PIPE_DEPTH],
    semaphore &gemm_outputs_arrived,
    semaphore &gemm_outputs_finished,
    uint32_t &gemm_bitfield,
    const int num_tokens,
    const int macrobatch_size,
    const int minibatch_size,
    const int macrobatch_idx,
    const int minibatch_idx,
    int task_idx,
    const int cta_rank,
    const int input_minibatch_ready_num_cols,
    const int input_row_block_ready_base_index,
    const int input_row_block_ready_required_count,
    const int output_tile_ready_base_index,
    const int buffer_done_index,
    const uint64_t smem_base_addr
) {
    using gemm_a_tile = std::conditional_t<IS_WGRAD, mlp_a_t_tile, mlp_a_tile>;
    using gemm_b_tile = std::conditional_t<IS_WGRAD, mlp_b_t_tile, mlp_b_tile>;
    gemm_a_tile (&a_smem)[config::MLP_LOAD_PIPE_DEPTH] = *reinterpret_cast<gemm_a_tile (*)[config::MLP_LOAD_PIPE_DEPTH]>(smem_base_addr);
    gemm_b_tile (&b_smem)[config::MLP_LOAD_PIPE_DEPTH] = *reinterpret_cast<gemm_b_tile (*)[config::MLP_LOAD_PIPE_DEPTH]>(smem_base_addr + sizeof(a_smem));
    mlp_d_tile (&d_smem)[config::MLP_NUM_D_TILES] = *reinterpret_cast<mlp_d_tile (*)[config::MLP_NUM_D_TILES]>(smem_base_addr + sizeof(a_smem) + sizeof(b_smem));

    const int col_blocks = (IS_WGRAD ? b_gmem.cols() : b_gmem.rows()) / config::MLP_Nb;
    const int global_minibatch_idx = macrobatch_idx * (macrobatch_size / minibatch_size) + minibatch_idx;
    const int macrobatch_row_block_offset = macrobatch_idx * (macrobatch_size / config::MLP_Mb);

    // Output tile (and for wgrad, K range in token rows) of this task
    int3 tile_coord = {-1, -1, -1};
    int k_start = 0, k_end = 0;
    if constexpr (IS_WGRAD) {
        const int row_blocks = a_gmem.cols() / config::MLP_Mb;
        const int expert_idx = IS_SHARED ? 0 : task_idx / (row_blocks * col_blocks);
        if constexpr (IS_SHARED) {
            k_end = a_gmem.rows();
        } else {
            int expert_row_offset = 0;
            for (int i = 0; i < expert_idx; ++i)
                expert_row_offset += tokens_per_expert[{i}];
            k_start = max(expert_row_offset, macrobatch_idx * macrobatch_size);
            k_end = min(expert_row_offset + tokens_per_expert[{expert_idx}], min((macrobatch_idx + 1) * macrobatch_size, num_tokens));
        }
        if (k_start < k_end) {
            const int2 swizzled = get_swizzled_2d_idx<config::MLP_SUPERGROUP_SIZE>(row_blocks, col_blocks, IS_SHARED ? task_idx : task_idx % (row_blocks * col_blocks));
            tile_coord = {swizzled.x, swizzled.y, expert_idx};
        }
    } else if constexpr (IS_SHARED) {
        const int row_blocks = a_gmem.rows() / config::MLP_Mb;
        const int num_tasks = row_blocks * col_blocks;
        if (task_idx < num_tasks) {
            const int2 swizzled = get_swizzled_2d_idx<config::MLP_SUPERGROUP_SIZE>(row_blocks, col_blocks, task_idx);
            tile_coord = {swizzled.x, swizzled.y, 0};
        }
    } else {
        const int minibatch_routed_row_blocks = minibatch_size / config::MLP_Mb;
        const int global_minibatch_routed_first_row_block = global_minibatch_idx * minibatch_routed_row_blocks;
        int global_row_block_offset = 0;
        for (int expert_idx = 0; expert_idx < b_gmem.depth(); ++expert_idx) {
            const int expert_row_blocks = tokens_per_expert[{expert_idx}] / config::MLP_Mb;
            const int global_first_row_block = max(global_minibatch_routed_first_row_block, global_row_block_offset);
            const int row_blocks = max(0, min(global_minibatch_routed_first_row_block + minibatch_routed_row_blocks, global_row_block_offset + expert_row_blocks) - global_first_row_block);
            const int num_tasks = row_blocks * col_blocks;
            if (task_idx < num_tasks) {
                const int2 swizzled = get_swizzled_2d_idx<config::MLP_SUPERGROUP_SIZE>(row_blocks, col_blocks, task_idx);
                tile_coord = {global_first_row_block + swizzled.x - macrobatch_row_block_offset, swizzled.y, expert_idx};
                break;
            }
            task_idx -= num_tasks;
            global_row_block_offset += expert_row_blocks;
        }
    }
    if (tile_coord.z < 0) {
        if (buffer_done != nullptr && threadIdx.x == 0)
            barrier_arrive(*buffer_done, buffer_done_index);
        return;
    }

    const int first_gemm_iters = a_gmem.cols() / config::MLP_Kb;
    const int iters_per_task = IS_WGRAD ? (k_end - k_start) / config::MLP_Kb
                                        : first_gemm_iters + (a2_gmem != nullptr ? a2_gmem->cols() / config::MLP_Kb : 0);

    if (warpgroup::groupid() == config::NUM_CONSUMERS) {
        if (warpgroup::warpid() == 3 && warp::elect_leader()) {
            int input_ring = 0;
            if constexpr (IS_WGRAD) {
                const int macrobatch_row_offset = macrobatch_idx * (macrobatch_size / config::MLP_Kb);
                for (int idx = 0, k_block = k_start / config::MLP_Kb; idx < iters_per_task; ++idx, ++k_block) {
                    const int row = k_block * config::MLP_Kb;
                    if (idx == 0 || row % config::MLP_Mb == 0) {
                        if (input_row_block_ready != nullptr)
                            barrier_wait(*input_row_block_ready, input_row_block_ready_base_index + row / config::MLP_Mb, input_row_block_ready_required_count);
                    }
                    if (idx == 0 || row % minibatch_size == 0) {
                        if (input_minibatch_ready != nullptr) {
                            const int row_minibatch_idx = row / minibatch_size;
                            const int minibatch_rows = min(minibatch_size, num_tokens - row_minibatch_idx * minibatch_size);
                            const int required_count = ((minibatch_rows + config::DISPATCH_COMBINE_Mb - 1) / config::DISPATCH_COMBINE_Mb) * (input_minibatch_ready_num_cols / config::DISPATCH_COMBINE_Nb);
                            barrier_wait(*input_minibatch_ready, row_minibatch_idx, required_count);
                        }
                    }
                    wait(gemm_inputs_finished[input_ring], get_phasebit<1>(gemm_bitfield, input_ring));
                    tma::cluster::load_async(a_smem[input_ring], a_gmem, {k_block - macrobatch_row_offset, tile_coord.x * 2 + cta_rank}, gemm_inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                    tma::cluster::load_async(b_smem[input_ring], b_gmem, {k_block - macrobatch_row_offset, tile_coord.y * 2 + cta_rank}, gemm_inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                    update_phasebit<1>(gemm_bitfield, input_ring);
                    input_ring = ring_advance<config::MLP_LOAD_PIPE_DEPTH>(input_ring);
                }
            } else {
                if (input_row_block_ready != nullptr) {
                    barrier_wait(*input_row_block_ready, input_row_block_ready_base_index + macrobatch_row_block_offset + tile_coord.x, input_row_block_ready_required_count);
                }
                if (input_minibatch_ready != nullptr) {
                    const int minibatch_first_row = global_minibatch_idx * minibatch_size;
                    const int minibatch_rows = max(0, min(minibatch_size, num_tokens - minibatch_first_row));
                    const int required_count = ((minibatch_rows + config::DISPATCH_COMBINE_Mb - 1) / config::DISPATCH_COMBINE_Mb) * (a_gmem.cols() / config::DISPATCH_COMBINE_Nb);
                    barrier_wait(*input_minibatch_ready, global_minibatch_idx, required_count);
                }
                for (int idx = 0; idx < iters_per_task; ++idx) {
                    const activation_gl &a_gmem_curr = idx < first_gemm_iters ? a_gmem : *a2_gmem;
                    const auto          &b_gmem_curr = idx < first_gemm_iters ? b_gmem : *b2_gmem;
                    const int k_block = idx < first_gemm_iters ? idx : idx - first_gemm_iters;
                    wait(gemm_inputs_finished[input_ring], get_phasebit<1>(gemm_bitfield, input_ring));
                    tma::cluster::load_async(a_smem[input_ring], a_gmem_curr, {tile_coord.x * 2 + cta_rank, k_block},               gemm_inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                    tma::cluster::load_async(b_smem[input_ring], b_gmem_curr, {tile_coord.z, tile_coord.y * 2 + cta_rank, k_block}, gemm_inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                    update_phasebit<1>(gemm_bitfield, input_ring);
                    input_ring = ring_advance<config::MLP_LOAD_PIPE_DEPTH>(input_ring);
                }
            }
        } else if (cta_rank == 0 && warpgroup::warpid() == 0 && warp::elect_leader()) {
            int input_ring = 0;
            wait(gemm_outputs_finished, get_phasebit<1>(gemm_bitfield, config::MLP_LOAD_PIPE_DEPTH));
            update_phasebit<1>(gemm_bitfield, config::MLP_LOAD_PIPE_DEPTH);
            for (int idx = 0; idx < iters_per_task; ++idx) {
                tma::expect_bytes(gemm_inputs_arrived[input_ring], config::CLUSTER_SIZE * sizeof(gemm_a_tile) + 2 * sizeof(gemm_b_tile));
                wait(gemm_inputs_arrived[input_ring], get_phasebit<0>(gemm_bitfield, input_ring));
                if constexpr (IS_WGRAD) {
                    if (idx == 0) mm2_AtB (d_tt, a_smem[input_ring], b_smem[input_ring], gemm_inputs_finished[input_ring]);
                    else          mma2_AtB(d_tt, a_smem[input_ring], b_smem[input_ring], gemm_inputs_finished[input_ring]);
                } else {
                    if (idx == 0) mm2_ABt (d_tt, a_smem[input_ring], b_smem[input_ring], gemm_inputs_finished[input_ring]);
                    else          mma2_ABt(d_tt, a_smem[input_ring], b_smem[input_ring], gemm_inputs_finished[input_ring]);
                }
                update_phasebit<0>(gemm_bitfield, input_ring);
                input_ring = ring_advance<config::MLP_LOAD_PIPE_DEPTH>(input_ring);
            }
            detail::tcgen05::commit<config::CLUSTER_SIZE>(gemm_outputs_arrived);
        }
    } else {
        using epilogue_group = group<WARPGROUP_WARPS>;
        wait(gemm_outputs_arrived, get_phasebit<0>(gemm_bitfield, config::MLP_LOAD_PIPE_DEPTH));
        update_phasebit<0>(gemm_bitfield, config::MLP_LOAD_PIPE_DEPTH);
        rt_bf<config::MLP_Mb / 8, config::MLP_Nb / config::MLP_EPI_PIPE_DEPTH> d_reg[config::MLP_EPI_PIPE_DEPTH];
        #pragma unroll
        for (int i = 0; i < config::MLP_EPI_PIPE_DEPTH; ++i)
            warpgroup::load_async(d_reg[i], d_tt.template subtile<tt<float, config::MLP_Mb / 2, config::MLP_Nb / config::MLP_EPI_PIPE_DEPTH>>(0, config::MLP_Nb / config::MLP_EPI_PIPE_DEPTH * i));
        tensor_load_wait();
        warpgroup::sync(1);
        warpgroup::tma::cluster::arrive(gemm_outputs_finished, 0);
        #pragma unroll
        for (int i = 0; i < config::MLP_EPI_PIPE_DEPTH; ++i) {
            warpgroup::tma::store_async_read_wait<config::MLP_NUM_D_TILES - 1>();
            warpgroup::sync(1);
            warpgroup::store(d_smem[i % config::MLP_NUM_D_TILES], d_reg[i]);
            warpgroup::sync(1);
            if constexpr (IS_WGRAD)
                warpgroup::tma::store_add_async<dim::ROW, cache_policy::EVICT_FIRST>(d_gmem, d_smem[i % config::MLP_NUM_D_TILES], {tile_coord.z, 2 * tile_coord.x + cta_rank, config::MLP_EPI_PIPE_DEPTH * tile_coord.y + i});
            else
                warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(d_gmem, d_smem[i % config::MLP_NUM_D_TILES], {2 * tile_coord.x + cta_rank, config::MLP_EPI_PIPE_DEPTH * tile_coord.y + i});
        }
        epilogue_group::sync(4);
        if (epilogue_group::warpid() == 0 && warp::elect_leader()) {
            if (output_tile_ready != nullptr) {
                tma::store_async_wait();
                barrier_arrive(*output_tile_ready, output_tile_ready_base_index + (macrobatch_row_block_offset + tile_coord.x) * col_blocks + tile_coord.y);
            } else if (output_minibatch_ready != nullptr) {
                tma::store_async_wait();
                barrier_arrive(*output_minibatch_ready, global_minibatch_idx);
            }
            if (buffer_done != nullptr) {
                tma::store_async_wait();
                barrier_arrive(*buffer_done, buffer_done_index);
            }
        }
    }
}

static __device__ __forceinline__ void dispatch_mlp_swiglu_combine_fwd_kernel(const globals_fwd &g) {
    int cluster_idx = clusterIdx().x;
    const int cta_rank = cluster_ctarank();
    const int shared_row_blocks = g.x_shared.rows() / config::MLP_Mb;
    const int minibatch_routed_row_blocks = g.minibatch_size / config::MLP_Mb;
    const int shared_gate_up_tasks = shared_row_blocks * (g.w_shared_gate.rows() / config::MLP_Nb);
    const int minibatch_routed_gate_up_tasks = minibatch_routed_row_blocks * (g.w_routed_gate.rows() / config::MLP_Nb);
    const int shared_swiglu_tiles = (g.hidden_shared.rows() / config::SWIGLU_Mb) * (g.hidden_shared.cols() / config::SWIGLU_Nb);
    const int minibatch_routed_swiglu_tiles = (g.minibatch_size / config::SWIGLU_Mb) * (g.hidden_routed.cols() / config::SWIGLU_Nb);
    const int shared_swiglu_tasks = (shared_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH);
    const int minibatch_routed_swiglu_tasks = (minibatch_routed_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH);
    const int shared_down_tasks = shared_row_blocks * (g.w_shared_down.rows() / config::MLP_Nb);
    const int minibatch_routed_down_tasks = minibatch_routed_row_blocks * (g.w_routed_down.rows() / config::MLP_Nb);
    const int shared_tasks = 2 * shared_gate_up_tasks + shared_swiglu_tasks + shared_down_tasks;
    const int minibatch_tasks = 2 * minibatch_routed_gate_up_tasks + minibatch_routed_swiglu_tasks + minibatch_routed_down_tasks;
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
    uint32_t swiglu_bitfield = 0xFFFF0000;
    uint32_t dispatch_combine_bitfield = 0xFFFF0000;

    __shared__ clc::handle clc_handle[config::CLC_PIPE_DEPTH];
    __shared__ clc::handle clc_drain_handle[config::CLC_DRAIN_PIPE_DEPTH];
    __shared__ semaphore schedule_arrived[config::CLC_PIPE_DEPTH], schedule_finished[config::CLC_PIPE_DEPTH];
    __shared__ semaphore drain_schedule_arrived[config::CLC_DRAIN_PIPE_DEPTH];
    __shared__ semaphore swiglu_inputs_arrived[config::SWIGLU_FWD_PIPE_DEPTH];
    __shared__ semaphore gemm_inputs_arrived[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_inputs_finished[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_outputs_arrived, gemm_outputs_finished;
    __shared__ semaphore dispatch_combine_inputs_arrived[config::DISPATCH_COMBINE_PIPE_DEPTH];

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < config::SWIGLU_FWD_PIPE_DEPTH; ++i) {
            init_semaphore(swiglu_inputs_arrived[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < config::MLP_LOAD_PIPE_DEPTH; ++i) {
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
    everyone::tma::cluster::sync();

    if (cluster_idx < comm_clusters) {
        const int comm_cta_idx = cluster_idx * config::CLUSTER_SIZE + cta_rank;
        const int num_macrobatches = (num_tokens + macrobatch_size - 1) / macrobatch_size;
        auto num_dispatch_combine_tasks = [&](int macrobatch_idx) {
            const int macrobatch_tokens = min(macrobatch_size, num_tokens - macrobatch_idx * macrobatch_size);
            const int dispatch_combine_tiles = (macrobatch_tokens / config::DISPATCH_COMBINE_Mb) * (g.x_routed.cols() / config::DISPATCH_COMBINE_Nb);
            return (dispatch_combine_tiles + config::DISPATCH_COMBINE_PIPE_DEPTH - 1) / config::DISPATCH_COMBINE_PIPE_DEPTH;
        };
        auto dispatch = [&](int macrobatch_idx, int task_idx) {
            dispatch_combine_kernel<true>(g.x_routed_send_buffer, g.x_routed, g.schedule_peer_rank, g.schedule_peer_token_idx,
                                          nullptr, &g.x_routed_ready, dispatch_combine_inputs_arrived, dispatch_combine_bitfield,
                                          num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, task_idx, g.topk, 0, 0, smem_base_addr);
        };
        auto combine = [&](int macrobatch_idx, int task_idx) {
            dispatch_combine_kernel<false>(g.y_routed_recv_buffer, g.y_routed, g.schedule_peer_rank, g.schedule_peer_token_idx,
                                           &g.y_routed_ready, nullptr, dispatch_combine_inputs_arrived, dispatch_combine_bitfield,
                                           num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, task_idx, 1, 0, 0, smem_base_addr);
        };
        for (int task_idx = comm_cta_idx; task_idx < num_dispatch_combine_tasks(0); task_idx += g.num_comm_sms)
            dispatch(0, task_idx);
        for (int macrobatch_idx = 0; macrobatch_idx < num_macrobatches; ++macrobatch_idx) {
            const int combine_tasks = num_dispatch_combine_tasks(macrobatch_idx);
            const int dispatch_tasks = macrobatch_idx + 1 < num_macrobatches ? num_dispatch_combine_tasks(macrobatch_idx + 1) : 0;
            for (int task_idx = comm_cta_idx; task_idx < combine_tasks; task_idx += g.num_comm_sms) {
                combine(macrobatch_idx, task_idx);
                if (task_idx < dispatch_tasks)
                    dispatch(macrobatch_idx + 1, task_idx);
            }
        }
        return;
    }

    // Swiglu tasks are CTA-local, GEMM is not
    auto is_cta_local_task = [&](int compute_cluster_idx) {
        const int minibatch_task_idx = (compute_cluster_idx - shared_tasks) % minibatch_tasks;
        if (compute_cluster_idx < 0) return false;
        else if (compute_cluster_idx < 2 * shared_gate_up_tasks) return false; // shared gate/up
        else if (compute_cluster_idx < 2 * shared_gate_up_tasks + shared_swiglu_tasks) return true; // shared swiglu
        else if (compute_cluster_idx < shared_tasks) return false; // shared down
        else if (minibatch_task_idx < 2 * minibatch_routed_gate_up_tasks) return false; // routed gate/up
        else if (minibatch_task_idx < 2 * minibatch_routed_gate_up_tasks + minibatch_routed_swiglu_tasks) return true; // routed swiglu
        else return false; // routed down
    };
    const int hidden_row_block_ready_required_count = (config::MLP_Mb / config::SWIGLU_Mb) * (g.hidden_shared.cols() / config::SWIGLU_Nb);

    for (int task_iter = 0; cluster_idx >= 0 && cluster_idx < true_num_clusters; ++task_iter) {
        const int clc_stage = task_iter % config::CLC_PIPE_DEPTH;
        if (warpgroup::groupid() == config::NUM_CONSUMERS && warpgroup::warpid() == 1 && warp::elect_leader()) { // warp not used by the gemms
            if (cta_rank == 0) {
                wait(schedule_finished[clc_stage], ((task_iter + config::CLC_PIPE_DEPTH) / config::CLC_PIPE_DEPTH) % 2);
                clc::schedule(clc_handle[clc_stage], schedule_arrived[clc_stage]);
            }
            tma::expect_bytes(schedule_arrived[clc_stage], sizeof(clc_handle[clc_stage]));
        }

        const int compute_cluster_idx = cluster_idx - comm_clusters;
        const bool current_is_cta_local = is_cta_local_task(compute_cluster_idx);

        if (compute_cluster_idx < shared_gate_up_tasks) {
            // Shared gate
            const int task_idx = compute_cluster_idx;
            expert_grouped_gemm<true>(g.x_shared, g.w_shared_gate, nullptr, nullptr, g.gate_shared,
                                      g.tokens_per_expert, nullptr, nullptr, &g.gate_up_tile_ready, nullptr, nullptr,
                                      d_tt, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                      num_tokens, macrobatch_size, g.minibatch_size, 0, 0, task_idx, cta_rank,
                                      0, 0, 0, 0, 0, smem_base_addr);
        } else if (compute_cluster_idx < shared_gate_up_tasks * 2) {
            // Shared up
            const int task_idx = compute_cluster_idx - shared_gate_up_tasks;
            expert_grouped_gemm<true>(g.x_shared, g.w_shared_up, nullptr, nullptr, g.up_shared,
                                      g.tokens_per_expert, nullptr, nullptr, &g.gate_up_tile_ready, nullptr, nullptr,
                                      d_tt, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                      num_tokens, macrobatch_size, g.minibatch_size, 0, 0, task_idx, cta_rank,
                                      0, 0, 0, 0, 0, smem_base_addr);
        } else if (compute_cluster_idx < shared_gate_up_tasks * 2 + shared_swiglu_tasks) {
            // Shared Swiglu
            const int task_idx = compute_cluster_idx - shared_gate_up_tasks * 2;
            swiglu_fwd<true>(g.gate_shared, g.up_shared, g.hidden_shared, g.gate_up_tile_ready, g.hidden_row_block_ready,
                             swiglu_inputs_arrived, swiglu_bitfield,
                             g.x_shared.rows(), macrobatch_size, g.minibatch_size, 0, 0, task_idx, cta_rank, 0, 0, smem_base_addr);
        } else if (compute_cluster_idx < shared_tasks) {
            // Shared down
            const int task_idx = compute_cluster_idx - shared_gate_up_tasks * 2 - shared_swiglu_tasks;
            expert_grouped_gemm<true>(g.hidden_shared, g.w_shared_down, nullptr, nullptr, g.y_shared,
                                      g.tokens_per_expert, nullptr, &g.hidden_row_block_ready, nullptr, nullptr, nullptr,
                                      d_tt, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                      num_tokens, macrobatch_size, g.minibatch_size, 0, 0, task_idx, cta_rank,
                                      0, 0, hidden_row_block_ready_required_count, 0, 0, smem_base_addr);
        } else {
            // Routed expert with macro/minibatching
            const int global_minibatch_idx = (compute_cluster_idx - shared_tasks) / minibatch_tasks;
            const int minibatch_task_idx = (compute_cluster_idx - shared_tasks) - global_minibatch_idx * minibatch_tasks;
            const int minibatches_per_macrobatch = macrobatch_size / g.minibatch_size;
            const int macrobatch_idx = global_minibatch_idx / minibatches_per_macrobatch;
            const int minibatch_idx = global_minibatch_idx - macrobatch_idx * minibatches_per_macrobatch;

            if (minibatch_task_idx < minibatch_routed_gate_up_tasks) {
                // Routed gate
                const int task_idx = minibatch_task_idx;
                expert_grouped_gemm<false>(g.x_routed, g.w_routed_gate, nullptr, nullptr, g.gate_routed,
                                           g.tokens_per_expert, &g.x_routed_ready, nullptr, &g.gate_up_tile_ready, nullptr, nullptr,
                                           d_tt, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                           num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, minibatch_idx, task_idx, cta_rank,
                                           0, 0, 0, shared_gate_up_tasks, 0, smem_base_addr);
            } else if (minibatch_task_idx < minibatch_routed_gate_up_tasks * 2) {
                // Routed up
                const int task_idx = minibatch_task_idx - minibatch_routed_gate_up_tasks;
                expert_grouped_gemm<false>(g.x_routed, g.w_routed_up, nullptr, nullptr, g.up_routed,
                                           g.tokens_per_expert, &g.x_routed_ready, nullptr, &g.gate_up_tile_ready, nullptr, nullptr,
                                           d_tt, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                           num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, minibatch_idx, task_idx, cta_rank,
                                           0, 0, 0, shared_gate_up_tasks, 0, smem_base_addr);
            } else if (minibatch_task_idx < minibatch_routed_gate_up_tasks * 2 + minibatch_routed_swiglu_tasks) {
                // Routed Swiglu
                const int task_idx = minibatch_task_idx - minibatch_routed_gate_up_tasks * 2;
                swiglu_fwd<false>(g.gate_routed, g.up_routed, g.hidden_routed, g.gate_up_tile_ready, g.hidden_row_block_ready,
                                  swiglu_inputs_arrived, swiglu_bitfield,
                                  num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, minibatch_idx, task_idx, cta_rank,
                                  shared_gate_up_tasks, shared_row_blocks, smem_base_addr);
            } else {
                // Routed down
                const int task_idx = minibatch_task_idx - minibatch_routed_gate_up_tasks * 2 - minibatch_routed_swiglu_tasks;
                expert_grouped_gemm<false>(g.hidden_routed, g.w_routed_down, nullptr, nullptr, g.y_routed,
                                           g.tokens_per_expert, nullptr, &g.hidden_row_block_ready, nullptr, &g.y_routed_ready, nullptr,
                                           d_tt, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                           num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, minibatch_idx, task_idx, cta_rank,
                                           0, shared_row_blocks, hidden_row_block_ready_required_count, 0, 0, smem_base_addr);
            }
        }

        wait(schedule_arrived[clc_stage], (task_iter / config::CLC_PIPE_DEPTH) % 2);
        const auto schedule = clc::query(clc_handle[clc_stage]);
        cluster_idx = schedule.success ? static_cast<int>(schedule.x / config::CLUSTER_SIZE) : -1;
        __syncwarp();
        warp::tma::cluster::arrive(schedule_finished[clc_stage], 0);

        // SWIGLU -> GEMM requires a cluster-wide sync
        const int next_compute_cluster_idx = cluster_idx - comm_clusters;
        if (current_is_cta_local && cluster_idx >= 0 && !is_cta_local_task(next_compute_cluster_idx))
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

static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, 
                           at::Tensor, at::Tensor, at::Tensor, at::Tensor>
dispatch_mlp_swiglu_combine_fwd(
    // Inputs and communication buffers
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,
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
    const at::Tensor &num_tokens,
    const at::Tensor &tokens_per_expert,

    // Metadata
    int topk,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size
) {
    const int num_local_tokens = x.size(0);
    const int schedule_capacity = schedule_peer_rank.size(0);
    const int hidden_dim = x.size(1);
    const int num_global_minibatches = (schedule_capacity + minibatch_size - 1) / minibatch_size;
    const int shared_row_blocks = num_local_tokens / config::MLP_Mb;
    const int routed_row_blocks = schedule_capacity / config::MLP_Mb;
    const int shared_gate_up_tasks = shared_row_blocks * (w_shared_gate.size(0) / config::MLP_Nb);
    const int routed_gate_up_tasks = routed_row_blocks * (w_routed_gate.size(1) / config::MLP_Nb);

    bf16 *x_routed_send_buffer_data[NUM_DEVICES];
    bf16 *y_routed_recv_buffer_data[NUM_DEVICES];
    for (int i = 0; i < NUM_DEVICES; ++i) {
        x_routed_send_buffer_data[i] = reinterpret_cast<bf16*>(x_ptrs[i]);
        y_routed_recv_buffer_data[i] = reinterpret_cast<bf16*>(combine_buffer_ptrs[i]);
    }

    at::Tensor x_routed = at::empty({macrobatch_size, hidden_dim}, x.options());
    at::Tensor gate_shared = at::empty({x.size(0), w_shared_gate.size(0)}, x.options());
    at::Tensor gate_routed = at::empty({macrobatch_size, w_routed_gate.size(1)}, x.options());
    at::Tensor up_shared = at::empty({x.size(0), w_shared_up.size(0)}, x.options());
    at::Tensor up_routed = at::empty({macrobatch_size, w_routed_up.size(1)}, x.options());
    at::Tensor hidden_shared = at::empty({x.size(0), w_shared_gate.size(0)}, x.options());
    at::Tensor hidden_routed = at::empty({macrobatch_size, w_routed_gate.size(1)}, x.options());
    at::Tensor y_shared = at::empty_like(x);
    at::Tensor y_routed = at::empty_like(x_routed);
    at::Tensor x_routed_ready = at::zeros({num_global_minibatches}, tokens_per_expert.options());
    at::Tensor gate_up_tile_ready = at::zeros({shared_gate_up_tasks + routed_gate_up_tasks}, tokens_per_expert.options());
    at::Tensor hidden_row_block_ready = at::zeros({shared_row_blocks + routed_row_blocks}, tokens_per_expert.options());
    at::Tensor y_routed_ready = at::zeros({num_global_minibatches}, tokens_per_expert.options());

    globals_fwd g {
        .x_shared = kittens::py::tensor_to_gl<activation_gl>(x),
        .x_routed = kittens::py::tensor_to_gl<activation_gl>(x_routed),
        .gate_shared = kittens::py::tensor_to_gl<activation_gl>(gate_shared),
        .gate_routed = kittens::py::tensor_to_gl<activation_gl>(gate_routed),
        .up_shared = kittens::py::tensor_to_gl<activation_gl>(up_shared),
        .up_routed = kittens::py::tensor_to_gl<activation_gl>(up_routed),
        .hidden_shared = kittens::py::tensor_to_gl<activation_gl>(hidden_shared),
        .hidden_routed = kittens::py::tensor_to_gl<activation_gl>(hidden_routed),
        .y_shared = kittens::py::tensor_to_gl<activation_gl>(y_shared),
        .y_routed = kittens::py::tensor_to_gl<activation_gl>(y_routed),
        .x_routed_send_buffer = activation_pgl{x_routed_send_buffer_data, nullptr, nullptr, static_cast<size_t>(num_local_tokens), static_cast<size_t>(hidden_dim)},
        .y_routed_recv_buffer = activation_pgl{y_routed_recv_buffer_data, nullptr, nullptr, static_cast<size_t>(num_local_tokens * topk), static_cast<size_t>(hidden_dim)},
        .w_shared_gate = kittens::py::tensor_to_gl<weight_gl>(w_shared_gate),
        .w_routed_gate = kittens::py::tensor_to_gl<weight_gl>(w_routed_gate),
        .w_shared_up = kittens::py::tensor_to_gl<weight_gl>(w_shared_up),
        .w_routed_up = kittens::py::tensor_to_gl<weight_gl>(w_routed_up),
        .w_shared_down = kittens::py::tensor_to_gl<weight_gl>(w_shared_down),
        .w_routed_down = kittens::py::tensor_to_gl<weight_gl>(w_routed_down),
        .schedule_peer_rank = kittens::py::tensor_to_gl<index_gl>(schedule_peer_rank),
        .schedule_peer_token_idx = kittens::py::tensor_to_gl<index_gl>(schedule_peer_token_idx),
        .num_tokens = kittens::py::tensor_to_gl<index_gl>(num_tokens),
        .tokens_per_expert = kittens::py::tensor_to_gl<index_gl>(tokens_per_expert),
        .gate_up_tile_ready = kittens::py::tensor_to_gl<index_gl>(gate_up_tile_ready),
        .hidden_row_block_ready = kittens::py::tensor_to_gl<index_gl>(hidden_row_block_ready),
        .x_routed_ready = kittens::py::tensor_to_gl<index_gl>(x_routed_ready),
        .y_routed_ready = kittens::py::tensor_to_gl<index_gl>(y_routed_ready),
        .topk = topk,
        .num_comm_sms = num_comm_sms,
        .macrobatch_size = macrobatch_size,
        .minibatch_size = minibatch_size
    };

    kittens::py::launch_kernel<config, globals_fwd, dispatch_mlp_swiglu_combine_fwd_kernel>(g);

    return {x_routed, gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, y_shared, y_routed};
}

static __device__ __forceinline__ void dispatch_mlp_swiglu_combine_bwd_kernel(const globals_bwd &g) {
    const int num_local_experts = g.w_routed_gate.depth();
    const int intermediate_dim_col_blocks = g.hidden_shared.cols() / config::MLP_Nb;
    const int hidden_dim_col_blocks = g.d_y_shared.cols() / config::MLP_Nb;

    int cluster_idx = clusterIdx().x;
    const int cta_rank = cluster_ctarank();

    const int shared_row_blocks = g.d_y_shared.rows() / config::MLP_Mb;
    const int shared_dgrad_down_tasks = shared_row_blocks * intermediate_dim_col_blocks;
    const int shared_swiglu_bwd_tiles = (g.hidden_shared.rows() / config::SWIGLU_Mb) * (g.hidden_shared.cols() / config::SWIGLU_Nb);
    const int shared_swiglu_bwd_tasks = (shared_swiglu_bwd_tiles + config::CLUSTER_SIZE * config::SWIGLU_BWD_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_BWD_PIPE_DEPTH);
    const int shared_dgrad_gate_up_tasks = shared_row_blocks * hidden_dim_col_blocks;
    const int shared_wgrad_tasks = intermediate_dim_col_blocks * hidden_dim_col_blocks; // per weight matrix
    const int shared_tasks = shared_dgrad_down_tasks + shared_swiglu_bwd_tasks + shared_dgrad_gate_up_tasks + 3 * shared_wgrad_tasks;

    const int minibatch_routed_row_blocks = g.minibatch_size / config::MLP_Mb;
    const int minibatch_routed_dgrad_down_tasks = minibatch_routed_row_blocks * intermediate_dim_col_blocks;
    const int minibatch_routed_swiglu_tiles = (g.minibatch_size / config::SWIGLU_Mb) * (g.hidden_routed.cols() / config::SWIGLU_Nb);
    const int minibatch_routed_swiglu_bwd_tasks = (minibatch_routed_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_BWD_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_BWD_PIPE_DEPTH);
    const int minibatch_routed_dgrad_gate_up_tasks = minibatch_routed_row_blocks * hidden_dim_col_blocks;
    const int minibatch_routed_bwd_tasks = minibatch_routed_dgrad_down_tasks + minibatch_routed_swiglu_bwd_tasks + minibatch_routed_dgrad_gate_up_tasks;

    const int wgrad_matrix_tasks = num_local_experts * intermediate_dim_col_blocks * hidden_dim_col_blocks;
    const int wgrad_tasks = 3 * wgrad_matrix_tasks;

    const int minibatch_routed_gate_up_tasks = minibatch_routed_row_blocks * intermediate_dim_col_blocks;
    const int minibatch_routed_swiglu_fwd_tasks = (minibatch_routed_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH);
    const int minibatch_routed_replay_tasks = 2 * minibatch_routed_gate_up_tasks + minibatch_routed_swiglu_fwd_tasks;

    const int comm_clusters = g.num_comm_sms / config::CLUSTER_SIZE;
    const int macrobatch_size = g.macrobatch_size;
    const int num_tokens = g.num_tokens[{0}];
    const int num_macrobatches = (num_tokens + macrobatch_size - 1) / macrobatch_size;

    auto macrobatch_idx_of = [&](int i) { return i == 0 ? num_macrobatches - 1 : i - 1; }; // TODO: change order
    auto num_minibatches_of = [&](int macrobatch_idx) { return (min(num_tokens - macrobatch_idx * macrobatch_size, macrobatch_size) + g.minibatch_size - 1) / g.minibatch_size; };
    auto num_dispatch_combine_tasks_of = [&](int macrobatch_idx) {
        const int macrobatch_tokens = min(macrobatch_size, num_tokens - macrobatch_idx * macrobatch_size);
        const int tiles = (macrobatch_tokens / config::DISPATCH_COMBINE_Mb) * (g.d_y_shared.cols() / config::DISPATCH_COMBINE_Nb);
        return (tiles + config::DISPATCH_COMBINE_PIPE_DEPTH - 1) / config::DISPATCH_COMBINE_PIPE_DEPTH;
    };
    auto routed_buffers_done_required_count_of = [&](int macrobatch_idx) {
        return config::CLUSTER_SIZE * (num_minibatches_of(macrobatch_idx) * minibatch_routed_bwd_tasks + wgrad_tasks) + num_dispatch_combine_tasks_of(macrobatch_idx);
    };

    const int minibatches_per_macrobatch = macrobatch_size / g.minibatch_size;
    const int last_macrobatch_idx = num_macrobatches - 1;
    const int last_num_minibatches = num_minibatches_of(last_macrobatch_idx);
    const int saved_macrobatch_tasks = last_num_minibatches * minibatch_routed_bwd_tasks + wgrad_tasks;
    const int replayed_macrobatch_tasks = minibatches_per_macrobatch * (minibatch_routed_replay_tasks + minibatch_routed_bwd_tasks) + wgrad_tasks;
    const int num_minibatches = (num_tokens + g.minibatch_size - 1) / g.minibatch_size;
    const int num_replay_minibatches = (num_macrobatches - 1) * minibatches_per_macrobatch;
    const int true_num_clusters = comm_clusters + shared_tasks + num_minibatches * minibatch_routed_bwd_tasks +
                                  num_replay_minibatches * minibatch_routed_replay_tasks + num_macrobatches * wgrad_tasks;
    if (cluster_idx >= true_num_clusters) return;

    warpgroup::increase_registers<256>();

    extern __shared__ int __shm[];
    const uint64_t smem_base_addr = (reinterpret_cast<uint64_t>(&__shm[0]) + 1023) & ~uint64_t(1023);

    uint32_t gemm_bitfield = 0xFFFF0000;
    uint32_t swiglu_fwd_bitfield = 0xFFFF0000;
    uint32_t swiglu_bwd_bitfield = 0xFFFF0000;
    uint32_t dispatch_combine_bitfield = 0xFFFF0000;

    __shared__ clc::handle clc_handle[config::CLC_PIPE_DEPTH];
    __shared__ clc::handle clc_drain_handle[config::CLC_DRAIN_PIPE_DEPTH];
    __shared__ semaphore schedule_arrived[config::CLC_PIPE_DEPTH], schedule_finished[config::CLC_PIPE_DEPTH];
    __shared__ semaphore drain_schedule_arrived[config::CLC_DRAIN_PIPE_DEPTH];
    __shared__ semaphore swiglu_fwd_inputs_arrived[config::SWIGLU_FWD_PIPE_DEPTH];
    __shared__ semaphore swiglu_bwd_inputs_arrived[config::SWIGLU_BWD_PIPE_DEPTH];
    __shared__ semaphore gemm_inputs_arrived[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_inputs_finished[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_outputs_arrived, gemm_outputs_finished;
    __shared__ semaphore dispatch_combine_inputs_arrived[config::DISPATCH_COMBINE_PIPE_DEPTH];

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < config::SWIGLU_FWD_PIPE_DEPTH; ++i) {
            init_semaphore(swiglu_fwd_inputs_arrived[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < config::SWIGLU_BWD_PIPE_DEPTH; ++i) {
            init_semaphore(swiglu_bwd_inputs_arrived[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < config::MLP_LOAD_PIPE_DEPTH; ++i) {
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
    everyone::tma::cluster::sync();

    if (cluster_idx < comm_clusters) {
        const int comm_cta_idx = cluster_idx * config::CLUSTER_SIZE + cta_rank;
        auto reverse_combine = [&](int i, int task_idx) {
            const int prev_macrobatch_idx = i > 0 ? macrobatch_idx_of(i - 1) : 0;
            dispatch_combine_kernel<true>(g.d_combine_buffer, g.d_y_routed, g.schedule_peer_rank, g.schedule_peer_token_idx,
                                          i > 0 ? &g.routed_buffers_done : nullptr, &g.d_y_routed_ready,
                                          dispatch_combine_inputs_arrived, dispatch_combine_bitfield,
                                          num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx_of(i), task_idx, 1,
                                          prev_macrobatch_idx, i > 0 ? routed_buffers_done_required_count_of(prev_macrobatch_idx) : 0, smem_base_addr);
        };
        auto reverse_dispatch = [&](int i, int task_idx) {
            dispatch_combine_kernel<false>(g.d_x_routed_buffer, g.d_x_routed, g.schedule_peer_rank, g.schedule_peer_token_idx,
                                           &g.d_x_routed_ready, num_macrobatches > 1 ? &g.routed_buffers_done : nullptr,
                                           dispatch_combine_inputs_arrived, dispatch_combine_bitfield,
                                           num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx_of(i), task_idx, 1,
                                           0, 0, smem_base_addr);
        };
        auto replay_dispatch = [&](int i, int task_idx) {
            const int prev_macrobatch_idx = macrobatch_idx_of(i - 1);
            dispatch_combine_kernel<true>(g.x_routed_send_buffer, g.x_routed, g.schedule_peer_rank, g.schedule_peer_token_idx,
                                          &g.routed_buffers_done, &g.replayed_x_routed_ready,
                                          dispatch_combine_inputs_arrived, dispatch_combine_bitfield,
                                          num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx_of(i), task_idx, g.topk,
                                          prev_macrobatch_idx, routed_buffers_done_required_count_of(prev_macrobatch_idx), smem_base_addr);
        };
        for (int task_idx = comm_cta_idx; task_idx < num_dispatch_combine_tasks_of(macrobatch_idx_of(0)); task_idx += g.num_comm_sms)
            reverse_combine(0, task_idx);
        for (int i = 0; i < num_macrobatches; ++i) {
            // All reverse-dispatch tasks must complete before this CTA moves on: the next macrobatch's pulls
            // wait on routed_buffers_done, which counts every rank's reverse-dispatch arrivals (including this CTA's)
            for (int task_idx = comm_cta_idx; task_idx < num_dispatch_combine_tasks_of(macrobatch_idx_of(i)); task_idx += g.num_comm_sms)
                reverse_dispatch(i, task_idx);
            if (i + 1 < num_macrobatches) {
                for (int task_idx = comm_cta_idx; task_idx < num_dispatch_combine_tasks_of(macrobatch_idx_of(i + 1)); task_idx += g.num_comm_sms) {
                    reverse_combine(i + 1, task_idx);
                    replay_dispatch(i + 1, task_idx);
                }
            }
        }
        return;
    }

    // Swiglu (forward and backward) tasks are CTA-local, GEMM is not
    auto is_cta_local_task = [&](int compute_cluster_idx) {
        if (compute_cluster_idx < 0) return false;
        else if (compute_cluster_idx < shared_dgrad_down_tasks) return false; // shared dgrad down
        else if (compute_cluster_idx < shared_dgrad_down_tasks + shared_swiglu_bwd_tasks) return true; // shared swiglu bwd
        else if (compute_cluster_idx < shared_tasks) return false; // shared dgrad/wgrad

        int idx = compute_cluster_idx - shared_tasks;
        int num_minibatches, macrobatch_task_idx;
        if (idx < saved_macrobatch_tasks) {
            num_minibatches = last_num_minibatches;
            macrobatch_task_idx = idx;
        } else {
            idx -= saved_macrobatch_tasks;
            num_minibatches = minibatches_per_macrobatch;
            macrobatch_task_idx = idx % replayed_macrobatch_tasks;
            if (macrobatch_task_idx < num_minibatches * minibatch_routed_replay_tasks)
                return macrobatch_task_idx % minibatch_routed_replay_tasks >= 2 * minibatch_routed_gate_up_tasks; // swiglu fwd replay
            macrobatch_task_idx -= num_minibatches * minibatch_routed_replay_tasks;
        }
        if (macrobatch_task_idx >= num_minibatches * minibatch_routed_bwd_tasks) return false; // wgrad
        const int minibatch_task_idx = macrobatch_task_idx % minibatch_routed_bwd_tasks;
        return minibatch_task_idx >= minibatch_routed_dgrad_down_tasks &&
               minibatch_task_idx < minibatch_routed_dgrad_down_tasks + minibatch_routed_swiglu_bwd_tasks; // swiglu bwd
    };

    const int d_gate_up_row_block_ready_required_count = (config::MLP_Mb / config::SWIGLU_Mb) * (g.hidden_shared.cols() / config::SWIGLU_Nb);
    const index_gl *buffer_done = num_macrobatches > 1 ? &g.routed_buffers_done : nullptr;

    for (int task_iter = 0; cluster_idx >= 0 && cluster_idx < true_num_clusters; ++task_iter) {
        const int clc_stage = task_iter % config::CLC_PIPE_DEPTH;
        if (warpgroup::groupid() == config::NUM_CONSUMERS && warpgroup::warpid() == 1 && warp::elect_leader()) { // warp not used by the gemms
            if (cta_rank == 0) {
                wait(schedule_finished[clc_stage], ((task_iter + config::CLC_PIPE_DEPTH) / config::CLC_PIPE_DEPTH) % 2);
                clc::schedule(clc_handle[clc_stage], schedule_arrived[clc_stage]);
            }
            tma::expect_bytes(schedule_arrived[clc_stage], sizeof(clc_handle[clc_stage]));
        }

        const int compute_cluster_idx = cluster_idx - comm_clusters;
        const bool current_is_cta_local = is_cta_local_task(compute_cluster_idx);

        if (compute_cluster_idx < shared_dgrad_down_tasks) {
            // Shared dgrad down: d_hidden_shared = d_y_shared @ w_shared_down_T^T
            const int task_idx = compute_cluster_idx;
            expert_grouped_gemm<true>(g.d_y_shared, g.w_shared_down_T, nullptr, nullptr, g.d_hidden_shared,
                                      g.tokens_per_expert, nullptr, nullptr, &g.d_hidden_ready, nullptr, nullptr,
                                      d_tt, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                      num_tokens, macrobatch_size, g.minibatch_size, 0, 0, task_idx, cta_rank,
                                      0, 0, 0, 0, 0, smem_base_addr);
        } else if (compute_cluster_idx < shared_dgrad_down_tasks + shared_swiglu_bwd_tasks) {
            // Shared Swiglu bwd: d_gate_shared, d_up_shared = swiglu_bwd(d_hidden_shared, gate_shared, up_shared)
            const int task_idx = compute_cluster_idx - shared_dgrad_down_tasks;
            swiglu_bwd<true>(g.d_hidden_shared, g.gate_shared, g.up_shared, g.d_gate_shared, g.d_up_shared,
                             g.d_hidden_ready, nullptr, g.d_gate_up_ready, nullptr,
                             swiglu_bwd_inputs_arrived, swiglu_bwd_bitfield,
                             g.gate_shared.rows(), macrobatch_size, g.minibatch_size, 0, 0, task_idx, cta_rank,
                             0, 0, 0, 0, smem_base_addr);
        } else if (compute_cluster_idx < shared_dgrad_down_tasks + shared_swiglu_bwd_tasks + shared_dgrad_gate_up_tasks) {
            // Shared dgrad gate+up: d_x_shared = d_gate_shared @ w_shared_gate_T^T + d_up_shared @ w_shared_up_T^T
            const int task_idx = compute_cluster_idx - shared_dgrad_down_tasks - shared_swiglu_bwd_tasks;
            expert_grouped_gemm<true>(g.d_gate_shared, g.w_shared_gate_T, &g.d_up_shared, &g.w_shared_up_T, g.d_x_shared,
                                      g.tokens_per_expert, nullptr, &g.d_gate_up_ready, nullptr, nullptr, nullptr,
                                      d_tt, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                      num_tokens, macrobatch_size, g.minibatch_size, 0, 0, task_idx, cta_rank,
                                      0, 0, d_gate_up_row_block_ready_required_count, 0, 0, smem_base_addr);
        } else if (compute_cluster_idx < shared_dgrad_down_tasks + shared_swiglu_bwd_tasks + shared_dgrad_gate_up_tasks + shared_wgrad_tasks) {
            // Shared wgrad down: d_w_shared_down += d_y_shared^T @ hidden_shared
            const int task_idx = compute_cluster_idx - shared_dgrad_down_tasks - shared_swiglu_bwd_tasks - shared_dgrad_gate_up_tasks;
            expert_grouped_gemm<true, true>(g.d_y_shared, g.hidden_shared, nullptr, nullptr, g.d_w_shared_down,
                                            g.tokens_per_expert, nullptr, nullptr, nullptr, nullptr, nullptr,
                                            d_tt, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                            num_tokens, macrobatch_size, g.minibatch_size, 0, 0, task_idx, cta_rank,
                                            0, 0, 0, 0, 0, smem_base_addr);
        } else if (compute_cluster_idx < shared_tasks) {
            // Shared wgrad gate/up: d_w_shared_gate/up += d_gate/up_shared^T @ x_shared
            const int idx = compute_cluster_idx - shared_dgrad_down_tasks - shared_swiglu_bwd_tasks - shared_dgrad_gate_up_tasks - shared_wgrad_tasks;
            const bool is_gate = idx < shared_wgrad_tasks;
            const int task_idx = idx % shared_wgrad_tasks;
            expert_grouped_gemm<true, true>(is_gate ? g.d_gate_shared : g.d_up_shared, g.x_shared, nullptr, nullptr,
                                            is_gate ? g.d_w_shared_gate : g.d_w_shared_up,
                                            g.tokens_per_expert, nullptr, &g.d_gate_up_ready, nullptr, nullptr, nullptr,
                                            d_tt, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                            num_tokens, macrobatch_size, g.minibatch_size, 0, 0, task_idx, cta_rank,
                                            0, 0, d_gate_up_row_block_ready_required_count, 0, 0, smem_base_addr);
        } else {
            // Routed tasks, one processed macrobatch block at a time
            const int routed_task_idx = compute_cluster_idx - shared_tasks;
            const bool replayed = routed_task_idx >= saved_macrobatch_tasks;
            int macrobatch_idx, num_minibatches, macrobatch_task_idx;
            if (replayed) {
                const int replayed_task_idx = routed_task_idx - saved_macrobatch_tasks;
                macrobatch_idx = replayed_task_idx / replayed_macrobatch_tasks;
                num_minibatches = minibatches_per_macrobatch;
                macrobatch_task_idx = replayed_task_idx - macrobatch_idx * replayed_macrobatch_tasks;
            } else {
                macrobatch_idx = last_macrobatch_idx;
                num_minibatches = last_num_minibatches;
                macrobatch_task_idx = routed_task_idx;
            }

            const int replay_task_end = replayed ? num_minibatches * minibatch_routed_replay_tasks : 0;
            const int bwd_task_end = replay_task_end + num_minibatches * minibatch_routed_bwd_tasks;
            if (macrobatch_task_idx < replay_task_end) {
                const int minibatch_idx = macrobatch_task_idx / minibatch_routed_replay_tasks;
                const int minibatch_task_idx = macrobatch_task_idx % minibatch_routed_replay_tasks;
                if (minibatch_task_idx < 2 * minibatch_routed_gate_up_tasks) {
                    // Replay gate/up: gate/up_routed = x_routed @ w_routed_gate/up^T
                    const bool is_gate = minibatch_task_idx < minibatch_routed_gate_up_tasks;
                    const int task_idx = minibatch_task_idx % minibatch_routed_gate_up_tasks;
                    expert_grouped_gemm<false>(g.x_routed, is_gate ? g.w_routed_gate : g.w_routed_up, nullptr, nullptr,
                                               is_gate ? g.gate_routed : g.up_routed,
                                               g.tokens_per_expert, &g.replayed_x_routed_ready, nullptr, &g.replayed_gate_up_ready, nullptr, nullptr,
                                               d_tt, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                               num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, minibatch_idx, task_idx, cta_rank,
                                               0, 0, 0, 0, 0, smem_base_addr);
                } else {
                    // Replay Swiglu: hidden_routed = silu(gate_routed) * up_routed
                    const int task_idx = minibatch_task_idx - 2 * minibatch_routed_gate_up_tasks;
                    swiglu_fwd<false>(g.gate_routed, g.up_routed, g.hidden_routed, g.replayed_gate_up_ready, g.replayed_hidden_ready,
                                      swiglu_fwd_inputs_arrived, swiglu_fwd_bitfield,
                                      num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, minibatch_idx,
                                      task_idx, cta_rank, 0, 0, smem_base_addr);
                }
            } else if (macrobatch_task_idx < bwd_task_end) {
                const int minibatch_bwd_task_idx = macrobatch_task_idx - replay_task_end;
                const int minibatch_idx = minibatch_bwd_task_idx / minibatch_routed_bwd_tasks;
                const int minibatch_task_idx = minibatch_bwd_task_idx % minibatch_routed_bwd_tasks;
                if (minibatch_task_idx < minibatch_routed_dgrad_down_tasks) {
                    // Dgrad down: d_hidden_routed = d_y_routed @ w_routed_down_T^T
                    const int task_idx = minibatch_task_idx;
                    expert_grouped_gemm<false>(g.d_y_routed, g.w_routed_down_T, nullptr, nullptr, g.d_hidden_routed,
                                               g.tokens_per_expert, &g.d_y_routed_ready, nullptr, &g.d_hidden_ready, nullptr, buffer_done,
                                               d_tt, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                               num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, minibatch_idx, task_idx, cta_rank,
                                               0, 0, 0, shared_dgrad_down_tasks, macrobatch_idx, smem_base_addr);
                } else if (minibatch_task_idx < minibatch_routed_dgrad_down_tasks + minibatch_routed_swiglu_bwd_tasks) {
                    // Swiglu bwd: d_gate_routed, d_up_routed = swiglu_bwd(d_hidden_routed, gate_routed, up_routed)
                    const int task_idx = minibatch_task_idx - minibatch_routed_dgrad_down_tasks;
                    swiglu_bwd<false>(g.d_hidden_routed, g.gate_routed, g.up_routed, g.d_gate_routed, g.d_up_routed,
                                      g.d_hidden_ready, replayed ? &g.replayed_gate_up_ready : nullptr, g.d_gate_up_ready, buffer_done,
                                      swiglu_bwd_inputs_arrived, swiglu_bwd_bitfield,
                                      num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, minibatch_idx,
                                      task_idx, cta_rank, shared_dgrad_down_tasks, 0, shared_row_blocks, macrobatch_idx, smem_base_addr);
                } else {
                    // Dgrad gate+up: d_x_routed = d_gate_routed @ w_routed_gate_T^T + d_up_routed @ w_routed_up_T^T
                    const int task_idx = minibatch_task_idx - minibatch_routed_dgrad_down_tasks - minibatch_routed_swiglu_bwd_tasks;
                    expert_grouped_gemm<false>(g.d_gate_routed, g.w_routed_gate_T, &g.d_up_routed, &g.w_routed_up_T, g.d_x_routed,
                                               g.tokens_per_expert, nullptr, &g.d_gate_up_ready, nullptr, &g.d_x_routed_ready, buffer_done,
                                               d_tt, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                               num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, minibatch_idx, task_idx, cta_rank,
                                               0, shared_row_blocks, d_gate_up_row_block_ready_required_count, 0, macrobatch_idx, smem_base_addr);
                }
            } else {
                const int wgrad_task_idx = macrobatch_task_idx - bwd_task_end;
                if (wgrad_task_idx < wgrad_matrix_tasks) {
                    // Wgrad down: d_w_routed_down += d_y_routed^T @ hidden_routed (per expert)
                    const int task_idx = wgrad_task_idx;
                    expert_grouped_gemm<false, true>(g.d_y_routed, g.hidden_routed, nullptr, nullptr, g.d_w_routed_down,
                                                     g.tokens_per_expert, &g.d_y_routed_ready, replayed ? &g.replayed_hidden_ready : nullptr,
                                                     nullptr, nullptr, buffer_done,
                                                     d_tt, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                                     num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, 0, task_idx, cta_rank,
                                                     g.d_y_shared.cols(), 0, d_gate_up_row_block_ready_required_count, 0, macrobatch_idx, smem_base_addr);
                } else {
                    // Wgrad gate/up: d_w_routed_gate/up += d_gate/up_routed^T @ x_routed (per expert)
                    const bool is_gate = wgrad_task_idx < 2 * wgrad_matrix_tasks;
                    const int task_idx = (wgrad_task_idx - wgrad_matrix_tasks) % wgrad_matrix_tasks;
                    expert_grouped_gemm<false, true>(is_gate ? g.d_gate_routed : g.d_up_routed, g.x_routed, nullptr, nullptr,
                                                     is_gate ? g.d_w_routed_gate : g.d_w_routed_up,
                                                     g.tokens_per_expert, replayed ? &g.replayed_x_routed_ready : nullptr,
                                                     &g.d_gate_up_ready, nullptr, nullptr, buffer_done,
                                                     d_tt, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                                     num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, 0, task_idx, cta_rank,
                                                     g.d_y_shared.cols(), shared_row_blocks, d_gate_up_row_block_ready_required_count,
                                                     0, macrobatch_idx, smem_base_addr);
                }
            }
        }

        wait(schedule_arrived[clc_stage], (task_iter / config::CLC_PIPE_DEPTH) % 2);
        const auto schedule = clc::query(clc_handle[clc_stage]);
        cluster_idx = schedule.success ? static_cast<int>(schedule.x / config::CLUSTER_SIZE) : -1;
        __syncwarp();
        warp::tma::cluster::arrive(schedule_finished[clc_stage], 0);

        // SWIGLU -> GEMM requires a cluster-wide sync
        const int next_compute_cluster_idx = cluster_idx - comm_clusters;
        if (current_is_cta_local && cluster_idx >= 0 && !is_cta_local_task(next_compute_cluster_idx))
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

static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
dispatch_mlp_swiglu_combine_bwd(
    // Output gradient
    const at::Tensor &d_y_shared,               // (num_local_tokens, H)

    // Communication buffers
    const at::Tensor &d_combine_buffer,         // (num_local_tokens * topk, H)
    const std::vector<int64_t> &d_combine_buffer_ptrs,
    const at::Tensor &d_x_routed_buffer,        // (num_local_tokens * topk, H)
    const std::vector<int64_t> &d_x_routed_buffer_ptrs,

    // Pre-transposed weights
    const at::Tensor &w_shared_gate_T,         // (H, I)
    const at::Tensor &w_routed_gate_T,         // (E, H, I)
    const at::Tensor &w_shared_up_T,           // (H, I)
    const at::Tensor &w_routed_up_T,           // (E, H, I)
    const at::Tensor &w_shared_down_T,         // (I, H)
    const at::Tensor &w_routed_down_T,         // (E, I, H)

    // Activations saved from the forward
    const at::Tensor &x_routed,                // (macrobatch_size, H)
    const at::Tensor &gate_shared,             // (num_local_tokens, I)
    const at::Tensor &gate_routed,             // (macrobatch_size, I)
    const at::Tensor &up_shared,               // (num_local_tokens, I)
    const at::Tensor &up_routed,               // (macrobatch_size, I)
    const at::Tensor &hidden_shared,           // (num_local_tokens, I)
    const at::Tensor &hidden_routed,           // (macrobatch_size, I)

    // Activations and weights for forward replay
    const at::Tensor &x,                       // (num_local_tokens, H)
    const std::vector<int64_t> &x_ptrs,
    const at::Tensor &w_routed_gate,           // (E, I, H)
    const at::Tensor &w_routed_up,             // (E, I, H)

    // Dispatch/combine schedule saved from the forward
    const at::Tensor &schedule_peer_rank,      // (schedule_capacity,)
    const at::Tensor &schedule_peer_token_idx, // (schedule_capacity,)
    const at::Tensor &num_tokens,              // (1,)
    const at::Tensor &tokens_per_expert,       // (E,)

    // Metadata
    int topk,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size
) {
    const int num_local_tokens = x.size(0);
    const int schedule_capacity = schedule_peer_rank.size(0);
    const int hidden_dim = x.size(1);
    const int intermediate_dim = w_routed_gate.size(1);
    const int num_local_experts = w_routed_gate.size(0);
    const int num_global_minibatches = (schedule_capacity + minibatch_size - 1) / minibatch_size;
    const int num_macrobatches = (schedule_capacity + macrobatch_size - 1) / macrobatch_size;
    const int shared_row_blocks = num_local_tokens / config::MLP_Mb;
    const int routed_row_blocks = schedule_capacity / config::MLP_Mb;
    const int intermediate_dim_col_blocks = intermediate_dim / config::MLP_Nb;

    bf16 *x_routed_send_buffer_data[NUM_DEVICES];
    bf16 *d_combine_buffer_data[NUM_DEVICES];
    bf16 *d_x_routed_buffer_data[NUM_DEVICES];
    for (int i = 0; i < NUM_DEVICES; ++i) {
        x_routed_send_buffer_data[i] = reinterpret_cast<bf16*>(x_ptrs[i]);
        d_combine_buffer_data[i] = reinterpret_cast<bf16*>(d_combine_buffer_ptrs[i]);
        d_x_routed_buffer_data[i] = reinterpret_cast<bf16*>(d_x_routed_buffer_ptrs[i]);
    }

    // Gradient tensors
    at::Tensor d_y_routed = at::empty({macrobatch_size, hidden_dim}, d_y_shared.options());
    at::Tensor d_hidden_shared = at::empty({num_local_tokens, intermediate_dim}, d_y_shared.options());
    at::Tensor d_hidden_routed = at::empty({macrobatch_size, intermediate_dim}, d_y_shared.options());
    at::Tensor d_gate_shared = at::empty_like(d_hidden_shared);
    at::Tensor d_gate_routed = at::empty_like(d_hidden_routed);
    at::Tensor d_up_shared = at::empty_like(d_hidden_shared);
    at::Tensor d_up_routed = at::empty_like(d_hidden_routed);
    at::Tensor d_x_shared = at::empty({num_local_tokens, hidden_dim}, d_y_shared.options());
    at::Tensor d_x_routed = at::empty({macrobatch_size, hidden_dim}, d_y_shared.options());
    at::Tensor d_w_shared_gate = at::zeros({intermediate_dim, hidden_dim}, d_y_shared.options());
    at::Tensor d_w_routed_gate = at::zeros_like(w_routed_gate);
    at::Tensor d_w_shared_up = at::zeros({intermediate_dim, hidden_dim}, d_y_shared.options());
    at::Tensor d_w_routed_up = at::zeros_like(w_routed_up);
    at::Tensor d_w_shared_down = at::zeros({hidden_dim, intermediate_dim}, d_y_shared.options());
    at::Tensor d_w_routed_down = at::zeros({num_local_experts, hidden_dim, intermediate_dim}, d_y_shared.options());

    // Counters
    at::Tensor d_y_routed_ready = at::zeros({num_global_minibatches}, tokens_per_expert.options());
    at::Tensor d_hidden_ready = at::zeros({(shared_row_blocks + routed_row_blocks) * intermediate_dim_col_blocks}, tokens_per_expert.options());
    at::Tensor d_gate_up_ready = at::zeros({shared_row_blocks + routed_row_blocks}, tokens_per_expert.options());
    at::Tensor d_x_routed_ready = at::zeros({num_global_minibatches}, tokens_per_expert.options());
    at::Tensor replayed_x_routed_ready = at::zeros({num_global_minibatches}, tokens_per_expert.options());
    at::Tensor replayed_gate_up_ready = at::zeros({routed_row_blocks * intermediate_dim_col_blocks}, tokens_per_expert.options());
    at::Tensor replayed_hidden_ready = at::zeros({routed_row_blocks}, tokens_per_expert.options());
    at::Tensor routed_buffers_done = at::zeros({num_macrobatches}, tokens_per_expert.options());

    globals_bwd g {
        .x_shared = kittens::py::tensor_to_gl<activation_gl>(x),
        .x_routed = kittens::py::tensor_to_gl<activation_gl>(x_routed),
        .gate_shared = kittens::py::tensor_to_gl<activation_gl>(gate_shared),
        .gate_routed = kittens::py::tensor_to_gl<activation_gl>(gate_routed),
        .up_shared = kittens::py::tensor_to_gl<activation_gl>(up_shared),
        .up_routed = kittens::py::tensor_to_gl<activation_gl>(up_routed),
        .hidden_shared = kittens::py::tensor_to_gl<activation_gl>(hidden_shared),
        .hidden_routed = kittens::py::tensor_to_gl<activation_gl>(hidden_routed),
        .d_y_shared = kittens::py::tensor_to_gl<activation_gl>(d_y_shared),
        .d_y_routed = kittens::py::tensor_to_gl<activation_gl>(d_y_routed),
        .d_hidden_shared = kittens::py::tensor_to_gl<activation_gl>(d_hidden_shared),
        .d_hidden_routed = kittens::py::tensor_to_gl<activation_gl>(d_hidden_routed),
        .d_gate_shared = kittens::py::tensor_to_gl<activation_gl>(d_gate_shared),
        .d_gate_routed = kittens::py::tensor_to_gl<activation_gl>(d_gate_routed),
        .d_up_shared = kittens::py::tensor_to_gl<activation_gl>(d_up_shared),
        .d_up_routed = kittens::py::tensor_to_gl<activation_gl>(d_up_routed),
        .d_x_shared = kittens::py::tensor_to_gl<activation_gl>(d_x_shared),
        .d_x_routed = kittens::py::tensor_to_gl<activation_gl>(d_x_routed),
        .x_routed_send_buffer = activation_pgl{x_routed_send_buffer_data, nullptr, nullptr, static_cast<size_t>(num_local_tokens), static_cast<size_t>(hidden_dim)},
        .d_combine_buffer = activation_pgl{d_combine_buffer_data, nullptr, nullptr, static_cast<size_t>(num_local_tokens * topk), static_cast<size_t>(hidden_dim)},
        .d_x_routed_buffer = activation_pgl{d_x_routed_buffer_data, nullptr, nullptr, static_cast<size_t>(num_local_tokens * topk), static_cast<size_t>(hidden_dim)},
        .w_routed_gate = kittens::py::tensor_to_gl<weight_gl>(w_routed_gate),
        .w_routed_up = kittens::py::tensor_to_gl<weight_gl>(w_routed_up),
        .w_shared_gate_T = kittens::py::tensor_to_gl<weight_gl>(w_shared_gate_T),
        .w_routed_gate_T = kittens::py::tensor_to_gl<weight_gl>(w_routed_gate_T),
        .w_shared_up_T = kittens::py::tensor_to_gl<weight_gl>(w_shared_up_T),
        .w_routed_up_T = kittens::py::tensor_to_gl<weight_gl>(w_routed_up_T),
        .w_shared_down_T = kittens::py::tensor_to_gl<weight_gl>(w_shared_down_T),
        .w_routed_down_T = kittens::py::tensor_to_gl<weight_gl>(w_routed_down_T),
        .d_w_shared_gate = kittens::py::tensor_to_gl<weight_gl>(d_w_shared_gate),
        .d_w_routed_gate = kittens::py::tensor_to_gl<weight_gl>(d_w_routed_gate),
        .d_w_shared_up = kittens::py::tensor_to_gl<weight_gl>(d_w_shared_up),
        .d_w_routed_up = kittens::py::tensor_to_gl<weight_gl>(d_w_routed_up),
        .d_w_shared_down = kittens::py::tensor_to_gl<weight_gl>(d_w_shared_down),
        .d_w_routed_down = kittens::py::tensor_to_gl<weight_gl>(d_w_routed_down),
        .schedule_peer_rank = kittens::py::tensor_to_gl<index_gl>(schedule_peer_rank),
        .schedule_peer_token_idx = kittens::py::tensor_to_gl<index_gl>(schedule_peer_token_idx),
        .num_tokens = kittens::py::tensor_to_gl<index_gl>(num_tokens),
        .tokens_per_expert = kittens::py::tensor_to_gl<index_gl>(tokens_per_expert),
        .d_y_routed_ready = kittens::py::tensor_to_gl<index_gl>(d_y_routed_ready),
        .d_hidden_ready = kittens::py::tensor_to_gl<index_gl>(d_hidden_ready),
        .d_gate_up_ready = kittens::py::tensor_to_gl<index_gl>(d_gate_up_ready),
        .d_x_routed_ready = kittens::py::tensor_to_gl<index_gl>(d_x_routed_ready),
        .replayed_x_routed_ready = kittens::py::tensor_to_gl<index_gl>(replayed_x_routed_ready),
        .replayed_gate_up_ready = kittens::py::tensor_to_gl<index_gl>(replayed_gate_up_ready),
        .replayed_hidden_ready = kittens::py::tensor_to_gl<index_gl>(replayed_hidden_ready),
        .routed_buffers_done = kittens::py::tensor_to_gl<index_gl>(routed_buffers_done),
        .topk = topk,
        .num_comm_sms = num_comm_sms,
        .macrobatch_size = macrobatch_size,
        .minibatch_size = minibatch_size
    };

    kittens::py::launch_kernel<config, globals_bwd, dispatch_mlp_swiglu_combine_bwd_kernel>(g);

    return {d_x_shared, d_x_routed, d_gate_shared, d_gate_routed, d_up_shared, d_up_routed, d_hidden_shared, d_hidden_routed, d_y_routed,
            d_w_shared_gate, d_w_routed_gate, d_w_shared_up, d_w_routed_up, d_w_shared_down, d_w_routed_down};
}

}; // struct dispatch_mlp_swiglu_combiner

struct utilities {

struct config_fwd_epilogue {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 256;
    static constexpr int NUM_WARPS = NUM_THREADS / WARP_THREADS;
};

struct globals_fwd_epilogue {
    static constexpr int Nb = 1024;
    static constexpr int TOKENS_PER_CTA = 2;

    using token_vec = sv_bf<Nb>;
    using activation_gl = gl<bf16, 1, 1, -1, -1, token_vec>;
    using weight_gl = gl<float, 1, 1, -1, -1>;

    activation_gl y_shared;        // (num_local_tokens, H)
    activation_gl combine_buffer;  // (num_local_tokens * topk, H)
    weight_gl topk_weights;        // (num_local_tokens, topk)
    activation_gl output;          // (num_local_tokens, H)

    __host__ inline dim3 grid() const { return dim3(y_shared.cols() / Nb, y_shared.rows() / TOKENS_PER_CTA); }
    __host__ inline int dynamic_shared_memory() const {
        return TOKENS_PER_CTA * ((topk_weights.cols() + 1) * sizeof(token_vec) + topk_weights.cols() * sizeof(float)) + 1024;
    }
};

static __device__ inline void fwd_epilogue_kernel(const globals_fwd_epilogue &g) {
    constexpr int TOKENS_PER_CTA = globals_fwd_epilogue::TOKENS_PER_CTA;
    using compute_group = group<config_fwd_epilogue::NUM_WARPS>;

    const int tid = threadIdx.x;
    const int topk = g.topk_weights.cols();
    const int num_tokens_per_stage = topk + 1;
    const int col_block_idx = blockIdx.x;
    const int first_token_idx = blockIdx.y * TOKENS_PER_CTA;

    extern __shared__ int __shm[];
    auto *token_vecs = reinterpret_cast<globals_fwd_epilogue::token_vec*>((reinterpret_cast<uint64_t>(&__shm[0]) + 1023) & ~uint64_t(1023));
    float *weights = reinterpret_cast<float*>(token_vecs + TOKENS_PER_CTA * num_tokens_per_stage); // (TOKENS_PER_CTA, topk)

    __shared__ semaphore inputs_arrived[TOKENS_PER_CTA];
    if (tid == 0) {
        #pragma unroll
        for (int stage = 0; stage < TOKENS_PER_CTA; ++stage) {
            init_semaphore(inputs_arrived[stage], 0, 1);
            tma::expect_bytes(inputs_arrived[stage], num_tokens_per_stage * sizeof(globals_fwd_epilogue::token_vec));
        }
    }
    for (int i = tid; i < TOKENS_PER_CTA * topk; i += blockDim.x)
        weights[i] = g.topk_weights[{first_token_idx + i / topk, i % topk}];
    __syncthreads();

    #pragma unroll
    for (int stage = 0; stage < TOKENS_PER_CTA; ++stage) {
        const int token_idx = first_token_idx + stage;
        if (tid == 0)
            tma::load_async(token_vecs[stage * num_tokens_per_stage], g.y_shared, {token_idx, col_block_idx}, inputs_arrived[stage]);
        else if (tid < num_tokens_per_stage)
            tma::load_async(token_vecs[stage * num_tokens_per_stage + tid], g.combine_buffer, {token_idx * topk + tid - 1, col_block_idx}, inputs_arrived[stage]);
    }

    #pragma unroll
    for (int stage = 0; stage < TOKENS_PER_CTA; ++stage) {
        globals_fwd_epilogue::token_vec *stage_vecs = token_vecs + stage * num_tokens_per_stage;
        rv_fl<globals_fwd_epilogue::Nb / config_fwd_epilogue::NUM_WARPS> accumulator, term;
        wait(inputs_arrived[stage], 0);
        compute_group::load(accumulator, stage_vecs[0]);
        for (int k = 0; k < topk; ++k) {
            compute_group::load(term, stage_vecs[1 + k]);
            compute_group::mul(term, term, weights[stage * topk + k]);
            compute_group::add(accumulator, accumulator, term);
        }
        compute_group::store(stage_vecs[0], accumulator);
        __syncthreads();
        if (tid == 0)
            tma::store_async(g.output, stage_vecs[0], {first_token_idx + stage, col_block_idx});
    }
}

static __host__ at::Tensor fwd_epilogue(
    const at::Tensor &y_shared,
    const at::Tensor &combine_buffer,
    const at::Tensor &topk_weights
) {
    at::Tensor output = at::empty_like(y_shared);
    globals_fwd_epilogue g {
        .y_shared = kittens::py::tensor_to_gl<globals_fwd_epilogue::activation_gl>(y_shared),
        .combine_buffer = kittens::py::tensor_to_gl<globals_fwd_epilogue::activation_gl>(combine_buffer),
        .topk_weights = kittens::py::tensor_to_gl<globals_fwd_epilogue::weight_gl>(topk_weights),
        .output = kittens::py::tensor_to_gl<globals_fwd_epilogue::activation_gl>(output)
    };
    kittens::py::launch_kernel<config_fwd_epilogue, globals_fwd_epilogue, fwd_epilogue_kernel>(g);
    return output;
}

struct config_bwd_prologue {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 256;
    static constexpr int NUM_WARPS = NUM_THREADS / WARP_THREADS;
};

struct globals_bwd_prologue {
    static constexpr int Nb = 1024;

    using token_vec = sv_bf<Nb>;
    using activation_gl = gl<bf16, 1, 1, -1, -1, token_vec>;
    using weight_gl = gl<float, 1, 1, -1, -1>;

    activation_gl d_output;         // (num_local_tokens, H)
    weight_gl topk_weights;         // (num_local_tokens, topk)
    activation_gl d_combine_buffer; // (num_local_tokens * topk, H)

    __host__ inline dim3 grid() const { return dim3(d_output.cols() / Nb, d_output.rows()); }
    __host__ inline int dynamic_shared_memory() const {
        return (topk_weights.cols() + 1) * sizeof(token_vec) + topk_weights.cols() * sizeof(float) + 1024;
    }
};

static __device__ inline void bwd_prologue_kernel(const globals_bwd_prologue &g) {
    using compute_group = group<config_bwd_prologue::NUM_WARPS>;

    const int tid = threadIdx.x;
    const int topk = g.topk_weights.cols();
    const int token_idx = blockIdx.y;
    const int col_block_idx = blockIdx.x;

    extern __shared__ int __shm[];
    auto *token_vecs = reinterpret_cast<globals_bwd_prologue::token_vec*>((reinterpret_cast<uint64_t>(&__shm[0]) + 1023) & ~uint64_t(1023));
    float *weights = reinterpret_cast<float*>(token_vecs + topk + 1); // (topk,)

    __shared__ semaphore inputs_arrived;
    if (tid == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        tma::expect_bytes(inputs_arrived, sizeof(globals_bwd_prologue::token_vec));
        tma::load_async(token_vecs[0], g.d_output, {token_idx, col_block_idx}, inputs_arrived);
    } else if (tid - 1 < topk) {
        weights[tid - 1] = g.topk_weights[{token_idx, tid - 1}];
    }
    __syncthreads();

    rv_fl<globals_bwd_prologue::Nb / config_bwd_prologue::NUM_WARPS> d_output, term;
    wait(inputs_arrived, 0);
    compute_group::load(d_output, token_vecs[0]);
    for (int k = 0; k < topk; ++k) {
        compute_group::mul(term, d_output, weights[k]);
        compute_group::store(token_vecs[1 + k], term);
    }
    __syncthreads();
    if (tid < topk)
        tma::store_async(g.d_combine_buffer, token_vecs[1 + tid], {token_idx * topk + tid, col_block_idx});
}

static __host__ at::Tensor bwd_prologue(const at::Tensor &d_output, const at::Tensor &topk_weights, const at::Tensor &d_combine_buffer) {
    globals_bwd_prologue g {
        .d_output = kittens::py::tensor_to_gl<globals_bwd_prologue::activation_gl>(d_output),
        .topk_weights = kittens::py::tensor_to_gl<globals_bwd_prologue::weight_gl>(topk_weights),
        .d_combine_buffer = kittens::py::tensor_to_gl<globals_bwd_prologue::activation_gl>(d_combine_buffer)
    };
    kittens::py::launch_kernel<config_bwd_prologue, globals_bwd_prologue, bwd_prologue_kernel>(g);
    return d_output; // == d_y_shared; d_combine_buffer needs to be symmetric memory, so is updated in-place
}

struct config_bwd_epilogue {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 256;
    static constexpr int NUM_WARPS = NUM_THREADS / WARP_THREADS;
};

struct globals_bwd_epilogue {
    static constexpr int Nb = 1024;

    using token_vec = sv_bf<Nb>;
    using activation_gl = gl<bf16, 1, 1, -1, -1, token_vec>;

    activation_gl d_x_shared;        // (num_local_tokens, H)
    activation_gl d_x_routed_buffer; // (num_local_tokens * topk, H)
    activation_gl d_x;               // (num_local_tokens, H)

    __host__ inline dim3 grid() const { return dim3(d_x_shared.cols() / Nb, d_x_shared.rows()); }
    __host__ inline int dynamic_shared_memory() const { return (d_x_routed_buffer.rows() / d_x_shared.rows() + 1) * sizeof(token_vec) + 1024; }
};

static __device__ inline void bwd_epilogue_kernel(const globals_bwd_epilogue &g) {
    using compute_group = group<config_bwd_epilogue::NUM_WARPS>;

    const int tid = threadIdx.x;
    const int topk = g.d_x_routed_buffer.rows() / g.d_x_shared.rows();
    const int num_vecs = topk + 1;
    const int token_idx = blockIdx.y;
    const int col_block_idx = blockIdx.x;

    extern __shared__ int __shm[];
    auto *token_vecs = reinterpret_cast<globals_bwd_epilogue::token_vec*>((reinterpret_cast<uint64_t>(&__shm[0]) + 1023) & ~uint64_t(1023));

    __shared__ semaphore inputs_arrived;
    if (tid == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        tma::expect_bytes(inputs_arrived, num_vecs * sizeof(globals_bwd_epilogue::token_vec));
    }
    __syncthreads();

    if (tid == 0)
        tma::load_async(token_vecs[0], g.d_x_shared, {token_idx, col_block_idx}, inputs_arrived);
    else if (tid < num_vecs)
        tma::load_async(token_vecs[tid], g.d_x_routed_buffer, {token_idx * topk + tid - 1, col_block_idx}, inputs_arrived);

    rv_fl<globals_bwd_epilogue::Nb / config_bwd_epilogue::NUM_WARPS> accumulator, term;
    wait(inputs_arrived, 0);
    compute_group::load(accumulator, token_vecs[0]);
    for (int k = 0; k < topk; ++k) {
        compute_group::load(term, token_vecs[1 + k]);
        compute_group::add(accumulator, accumulator, term);
    }
    compute_group::store(token_vecs[0], accumulator);
    __syncthreads();
    if (tid == 0)
        tma::store_async(g.d_x, token_vecs[0], {token_idx, col_block_idx});
}

static __host__ at::Tensor bwd_epilogue(const at::Tensor &d_x_shared, const at::Tensor &d_x_routed_buffer) {
    at::Tensor d_x = at::empty_like(d_x_shared);
    globals_bwd_epilogue g {
        .d_x_shared = kittens::py::tensor_to_gl<globals_bwd_epilogue::activation_gl>(d_x_shared),
        .d_x_routed_buffer = kittens::py::tensor_to_gl<globals_bwd_epilogue::activation_gl>(d_x_routed_buffer),
        .d_x = kittens::py::tensor_to_gl<globals_bwd_epilogue::activation_gl>(d_x)
    };
    kittens::py::launch_kernel<config_bwd_epilogue, globals_bwd_epilogue, bwd_epilogue_kernel>(g);
    return d_x;
}

}; // struct utilities

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("schedule", &scheduler::schedule, "",
          pybind11::arg("topk_all"), pybind11::arg("num_local_experts"), pybind11::arg("schedule_capacity"), pybind11::arg("rank"));
    m.def("dispatch_mlp_swiglu_combine_fwd", &dispatch_mlp_swiglu_combiner<4>::dispatch_mlp_swiglu_combine_fwd, "",
          pybind11::arg("x"), pybind11::arg("x_ptrs"),
          pybind11::arg("combine_buffer"), pybind11::arg("combine_buffer_ptrs"),
          pybind11::arg("w_shared_gate"), pybind11::arg("w_routed_gate"),
          pybind11::arg("w_shared_up"), pybind11::arg("w_routed_up"),
          pybind11::arg("w_shared_down"), pybind11::arg("w_routed_down"),
          pybind11::arg("schedule_peer_rank"), pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("num_tokens"), pybind11::arg("tokens_per_expert"),
          pybind11::arg("topk"), pybind11::arg("num_comm_sms"),
          pybind11::arg("macrobatch_size"), pybind11::arg("minibatch_size"));
    m.def("dispatch_mlp_swiglu_combine_bwd", &dispatch_mlp_swiglu_combiner<4>::dispatch_mlp_swiglu_combine_bwd, "",
          pybind11::arg("d_y_shared"),
          pybind11::arg("d_combine_buffer"), pybind11::arg("d_combine_buffer_ptrs"),
          pybind11::arg("d_x_routed_buffer"), pybind11::arg("d_x_routed_buffer_ptrs"),
          pybind11::arg("w_shared_gate_T"), pybind11::arg("w_routed_gate_T"),
          pybind11::arg("w_shared_up_T"), pybind11::arg("w_routed_up_T"),
          pybind11::arg("w_shared_down_T"), pybind11::arg("w_routed_down_T"),
          pybind11::arg("x_routed"),
          pybind11::arg("gate_shared"), pybind11::arg("gate_routed"),
          pybind11::arg("up_shared"), pybind11::arg("up_routed"),
          pybind11::arg("hidden_shared"), pybind11::arg("hidden_routed"),
          pybind11::arg("x"), pybind11::arg("x_ptrs"),
          pybind11::arg("w_routed_gate"), pybind11::arg("w_routed_up"),
          pybind11::arg("schedule_peer_rank"), pybind11::arg("schedule_peer_token_idx"),
          pybind11::arg("num_tokens"), pybind11::arg("tokens_per_expert"),
          pybind11::arg("topk"), pybind11::arg("num_comm_sms"),
          pybind11::arg("macrobatch_size"), pybind11::arg("minibatch_size"));
    m.def("fwd_epilogue", &utilities::fwd_epilogue, "",
          pybind11::arg("y_shared"), pybind11::arg("combine_buffer"), pybind11::arg("topk_weights"));
    m.def("bwd_prologue", &utilities::bwd_prologue, "",
          pybind11::arg("d_output"), pybind11::arg("topk_weights"), pybind11::arg("d_combine_buffer"));
    m.def("bwd_epilogue", &utilities::bwd_epilogue, "",
          pybind11::arg("d_x_shared"), pybind11::arg("d_x_routed_buffer"));
}
