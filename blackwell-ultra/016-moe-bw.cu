#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros.h>

using namespace kittens;

namespace moe_swigluer {

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

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int NUM_CONSUMERS = 1;
    static constexpr int NUM_PRODUCERS = 1;
    static constexpr int NUM_WARPS = (NUM_CONSUMERS + NUM_PRODUCERS) * WARPGROUP_WARPS; // 8
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS; // 256

    static constexpr int NUM_D_TILES = EPI_PIPE_DEPTH > 1 ? 2 : 1;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1024;

    static_assert(MINIBATCH_SIZE % Mb == 0,        "MINIBATCH_SIZE must be a multiple of Mb");
    static_assert(MINIBATCH_SIZE % SWIGLU_Mb == 0, "MINIBATCH_SIZE must be a multiple of SWIGLU_Mb");
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

    activation_gl x;
    activation_gl gate;
    activation_gl up;
    activation_gl hidden;
    activation_gl y;
    weight_gl w_gate;
    weight_gl w_up;
    weight_gl w_down;
    index_gl tokens_per_expert;
    index_gl counters;

    __host__ inline dim3 grid() const {
        const int num_minibatches = (x.rows() + C::MINIBATCH_SIZE - 1) / C::MINIBATCH_SIZE;
        const int gate_up_tasks = (C::MINIBATCH_SIZE / C::Mb) * (w_gate.rows() / C::Nb);
        const int swiglu_tiles = (C::MINIBATCH_SIZE / C::SWIGLU_Mb) * (hidden.cols() / C::SWIGLU_Nb);
        const int swiglu_tasks = (swiglu_tiles + C::CLUSTER_SIZE * C::SWIGLU_PIPE_DEPTH - 1) / (C::CLUSTER_SIZE * C::SWIGLU_PIPE_DEPTH);
        const int down_tasks = (C::MINIBATCH_SIZE / C::Mb) * (w_down.rows() / C::Nb);
        const int tasks_per_minibatch = 2 * gate_up_tasks + swiglu_tasks + down_tasks;
        return dim3(C::CLUSTER_SIZE * num_minibatches * tasks_per_minibatch);
    }
};

template <typename C>
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

    const int intermediate_col_blocks = g.w_gate.rows() / C::Nb;
    const int gate_up_tasks = (g.x.rows() / C::Mb) * intermediate_col_blocks;

    int num_tokens = 0;
    for (int expert_idx = 0; expert_idx < g.w_gate.depth(); ++expert_idx)
        num_tokens += g.tokens_per_expert[{expert_idx}];
    const int row_blocks = num_tokens / C::SWIGLU_Mb;
    const int col_blocks = g.hidden.cols() / C::SWIGLU_Nb;
    const int num_tiles = row_blocks * col_blocks;
    const int num_tiles_per_minibatch = (C::MINIBATCH_SIZE / C::SWIGLU_Mb) * col_blocks;
    const int minibatch_first_tile_idx = minibatch_idx * num_tiles_per_minibatch;
    const int minibatch_tile_end = min(num_tiles, minibatch_first_tile_idx + num_tiles_per_minibatch);
    const int first_tile_idx = minibatch_first_tile_idx + task_idx * C::CLUSTER_SIZE * C::SWIGLU_PIPE_DEPTH + cta_rank * C::SWIGLU_PIPE_DEPTH;
    if (first_tile_idx >= minibatch_tile_end)
        return;

    int first_row, first_col;
    if (threadIdx.x == 0) {
        first_row = first_tile_idx / col_blocks;
        first_col = first_tile_idx % col_blocks;
        #pragma unroll
        for (int stage = 0; stage < C::SWIGLU_PIPE_DEPTH; ++stage) {
            const int tile_idx = first_tile_idx + stage;
            if (tile_idx < minibatch_tile_end) {
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
                    asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}" : "=r"(gate_up_counter) : "l"(&g.counters[{parent_task_idx}]) : "memory");
                    if (gate_up_counter >= 2 * C::CLUSTER_SIZE) break;
                    __nanosleep(16);
                }
                asm volatile("{fence.acquire.gpu;}" ::: "memory");

                tma::load_async(a_smem[stage], g.gate, {row, col}, swiglu_inputs_arrived[stage]);
                tma::load_async(b_smem[stage], g.up,   {row, col}, swiglu_inputs_arrived[stage]);
            }
        }
    }

    using compute_group = group<C::NUM_WARPS>;
    #pragma unroll
    for (int stage = 0; stage < C::SWIGLU_PIPE_DEPTH; ++stage) {
        const int tile_idx = first_tile_idx + stage;
        if (tile_idx < minibatch_tile_end) {
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
                tma::store_async(g.hidden, a_smem[stage], {row, col});
            }
        }
    }

    if (threadIdx.x == 0) {
        tma::store_async_wait();
        #pragma unroll
        for (int stage = 0; stage < C::SWIGLU_PIPE_DEPTH; ++stage) {
            const int tile_idx = first_tile_idx + stage;
            if (tile_idx < minibatch_tile_end) {
                // This improves throughput
                int row = first_row;
                int col = first_col + stage;
                if (col >= col_blocks)
                    ++row;
                asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}" :: "l"(&g.counters[{gate_up_tasks + row / (C::Mb / C::SWIGLU_Mb)}]), "r"(1) : "memory");
            }
        }
    }
}

enum class expert_gemm_kind { GATE, UP, DOWN };

template <typename C>
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

    const typename G::activation_gl &a_gmem = kind == expert_gemm_kind::DOWN ? g.hidden : g.x;
    const typename G::weight_gl &b_gmem     = kind == expert_gemm_kind::GATE ? g.w_gate : (kind == expert_gemm_kind::UP ? g.w_up : g.w_down);
    const typename G::activation_gl &d_gmem = kind == expert_gemm_kind::GATE ? g.gate   : (kind == expert_gemm_kind::UP ? g.up : g.y);

    constexpr int minibatch_row_blocks = C::MINIBATCH_SIZE / C::Mb;
    const int minibatch_row_offset = minibatch_idx * minibatch_row_blocks;
    const int iters_per_task = a_gmem.cols() / C::Kb;
    const int col_blocks     = b_gmem.rows() / C::Nb;

    int row_block_offset = 0;
    int3 tile_coord = {-1, -1, -1};
    for (int expert_idx = 0; expert_idx < b_gmem.depth(); ++expert_idx) {
        const int expert_row_blocks = g.tokens_per_expert[{expert_idx}] / C::Mb;
        const int first_row_block = max(minibatch_row_offset, row_block_offset);
        const int row_blocks = max(0, min(minibatch_row_offset + minibatch_row_blocks, row_block_offset + expert_row_blocks) - first_row_block);
        const int num_tasks = row_blocks * col_blocks;
        if (task_idx < num_tasks) {
            const int2 swizzled = get_swizzled_2d_idx<C::SUPERGROUP_SIZE>(row_blocks, col_blocks, task_idx);
            tile_coord = {first_row_block + swizzled.x, swizzled.y, expert_idx};
            break;
        }
        task_idx -= num_tasks;
        row_block_offset += expert_row_blocks;
    }
    if (tile_coord.z < 0) return;

    if (warpgroup::groupid() == C::NUM_CONSUMERS) {
        if (warpgroup::warpid() == 3 && warp::elect_leader()) {
            if (kind == expert_gemm_kind::DOWN) {
                const int gate_up_tasks = (g.x.rows() / C::Mb) * (g.w_gate.rows() / C::Nb);
                const int swiglu_tiles_per_row_block = (C::Mb / C::SWIGLU_Mb) * (g.hidden.cols() / C::SWIGLU_Nb);
                int swiglu_counter;
                while (true) {
                    asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}" : "=r"(swiglu_counter) : "l"(&g.counters[{gate_up_tasks + tile_coord.x}]) : "memory");
                    if (swiglu_counter >= swiglu_tiles_per_row_block) break;
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
                tma::store_async_wait();
                asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}" :: "l"(&g.counters[{tile_coord.x * col_blocks + tile_coord.y}]), "r"(1) : "memory");
            }
        }
    }
}

template <typename C>
__device__ __forceinline__ void moe_swiglu_kernel(const globals<C> &g) {
    using G = globals<C>;

    warpgroup::increase_registers<256>();

    extern __shared__ int __shm[];
    const uint64_t smem_base_addr = (reinterpret_cast<uint64_t>(&__shm[0]) + 1023) & ~uint64_t(1023);

    int cluster_idx = clusterIdx().x;
    const int cta_rank = cluster_ctarank();
    const int minibatch_gate_up_tasks = (C::MINIBATCH_SIZE / C::Mb) * (g.w_gate.rows() / C::Nb);
    const int minibatch_down_tasks    = (C::MINIBATCH_SIZE / C::Mb) * (g.w_down.rows() / C::Nb);
    const int minibatch_swiglu_tiles  = (C::MINIBATCH_SIZE / C::SWIGLU_Mb) * (g.hidden.cols() / C::SWIGLU_Nb);
    const int minibatch_swiglu_tasks  = (minibatch_swiglu_tiles + C::CLUSTER_SIZE * C::SWIGLU_PIPE_DEPTH - 1) / (C::CLUSTER_SIZE * C::SWIGLU_PIPE_DEPTH);
    const int tasks_per_minibatch = 2 * minibatch_gate_up_tasks + minibatch_swiglu_tasks + minibatch_down_tasks;
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

        const int minibatch_idx  = cluster_idx / tasks_per_minibatch;
        const int minibatch_task_idx = cluster_idx - minibatch_idx * tasks_per_minibatch;

        if (minibatch_task_idx < minibatch_gate_up_tasks) {
            // Gate
            current_is_swiglu = false;
            const int task_idx = minibatch_task_idx;
            expert_grouped_gemm<C>(g, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished,
                                   gemm_bitfield, cta_rank, minibatch_idx, task_idx, smem_base_addr, expert_gemm_kind::GATE, d_tt);
        } else if (minibatch_task_idx < minibatch_gate_up_tasks * 2) {
            // Up
            current_is_swiglu = false;
            const int task_idx = minibatch_task_idx - minibatch_gate_up_tasks;
            expert_grouped_gemm<C>(g, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished,
                                   gemm_bitfield, cta_rank, minibatch_idx, task_idx, smem_base_addr, expert_gemm_kind::UP, d_tt);
        } else if (minibatch_task_idx < minibatch_gate_up_tasks * 2 + minibatch_swiglu_tasks) {
            // Swiglu
            current_is_swiglu = true;
            const int task_idx = minibatch_task_idx - minibatch_gate_up_tasks * 2;
            swiglu<C>(g, swiglu_inputs_arrived, swiglu_bitfield, cta_rank, minibatch_idx, task_idx, smem_base_addr);
        } else {
            // Down
            current_is_swiglu = false;
            const int task_idx = minibatch_task_idx - minibatch_gate_up_tasks * 2 - minibatch_swiglu_tasks;
            expert_grouped_gemm<C>(g, gemm_inputs_arrived, gemm_inputs_finished, gemm_outputs_arrived, gemm_outputs_finished,
                                   gemm_bitfield, cta_rank, minibatch_idx, task_idx, smem_base_addr, expert_gemm_kind::DOWN, d_tt);
        }

        wait(schedule_arrived[clc_stage], (task_iter / C::CLC_PIPE_DEPTH) % 2);
        const auto schedule = clc::query(clc_handle[clc_stage]);
        cluster_idx = schedule.success ? static_cast<int>(schedule.x / C::CLUSTER_SIZE) : -1;
        __syncwarp();
        warp::tma::cluster::arrive(schedule_finished[clc_stage], 0);

        // SWIGLU -> GEMM requires a cluster-wide sync
        if (current_is_swiglu && cluster_idx >= 0) {
            const int next_minibatch_task_idx = cluster_idx - (cluster_idx / tasks_per_minibatch) * tasks_per_minibatch;
            if (next_minibatch_task_idx < minibatch_gate_up_tasks * 2 || next_minibatch_task_idx >= minibatch_gate_up_tasks * 2 + minibatch_swiglu_tasks)
                everyone::tma::cluster::sync();
        }
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> moe_swiglu(
    const at::Tensor &x,
    const at::Tensor &w_gate,
    const at::Tensor &w_up,
    const at::Tensor &w_down,
    const at::Tensor &tokens_per_expert
) {
    using C = config;
    using G = globals<C>;

    at::Tensor gate = at::empty({x.size(0), w_gate.size(1)}, x.options());
    at::Tensor up = at::empty({x.size(0), w_up.size(1)}, x.options());
    at::Tensor hidden = at::empty({x.size(0), w_gate.size(1)}, x.options());
    at::Tensor y = at::empty_like(x);
    const int row_blocks = x.size(0) / C::Mb;
    const int gate_up_tasks = row_blocks * (w_gate.size(1) / C::Nb);
    at::Tensor counters = at::zeros({gate_up_tasks + row_blocks}, tokens_per_expert.options());

    G g {
        .x = kittens::py::tensor_to_gl<G::activation_gl>(x),
        .gate = kittens::py::tensor_to_gl<G::activation_gl>(gate),
        .up = kittens::py::tensor_to_gl<G::activation_gl>(up),
        .hidden = kittens::py::tensor_to_gl<G::activation_gl>(hidden),
        .y = kittens::py::tensor_to_gl<G::activation_gl>(y),
        .w_gate = kittens::py::tensor_to_gl<G::weight_gl>(w_gate),
        .w_up = kittens::py::tensor_to_gl<G::weight_gl>(w_up),
        .w_down = kittens::py::tensor_to_gl<G::weight_gl>(w_down),
        .tokens_per_expert = kittens::py::tensor_to_gl<G::index_gl>(tokens_per_expert),
        .counters = kittens::py::tensor_to_gl<G::index_gl>(counters)
    };

    kittens::py::launch_kernel<C, G, moe_swiglu_kernel<C>>(g);

    return {gate, up, hidden, y};
}

} // namespace moe_swigluer

PYBIND11_MODULE(_C, m) {
    m.def("moe_swiglu", &moe_swigluer::moe_swiglu, "MoE SwiGLU",
          pybind11::arg("x"), pybind11::arg("w_gate"), pybind11::arg("w_up"), pybind11::arg("w_down"),
          pybind11::arg("tokens_per_expert"));
}
