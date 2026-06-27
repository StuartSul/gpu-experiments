#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros.h>
#include <tuple>

using namespace kittens;

namespace moe_swigluer {

struct config {
    static constexpr int Mb = 256;
    static constexpr int Nb = 256;
    static constexpr int Kb = 64;
    static constexpr int SUPERGROUP_SIZE = 8;

    static constexpr int SWIGLU_Mb = 128;
    static constexpr int SWIGLU_Nb = 128;

    static constexpr int LOAD_PIPE_DEPTH = 5;
    static constexpr int EPI_PIPE_DEPTH = 4;

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int NUM_CONSUMERS = 1;
    static constexpr int NUM_PRODUCERS = 1;
    static constexpr int NUM_WARPS = (NUM_CONSUMERS + NUM_PRODUCERS) * WARPGROUP_WARPS; // 8
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS; // 256

    static constexpr int NUM_D_TILES = EPI_PIPE_DEPTH > 1 ? 2 : 1;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1024;
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
        const int row_blocks = x.rows() / C::Mb;
        const int gate_up_tasks = row_blocks * (w_gate.rows() / C::Nb);
        const int swiglu_tiles = (hidden.rows() / C::SWIGLU_Mb) * (hidden.cols() / C::SWIGLU_Nb);
        const int swiglu_tasks = (swiglu_tiles + C::CLUSTER_SIZE * 3 - 1) / (C::CLUSTER_SIZE * 3);
        const int down_tasks = row_blocks * (w_down.rows() / C::Nb);
        return dim3(C::CLUSTER_SIZE * (2 * gate_up_tasks + swiglu_tasks + down_tasks));
    }
};

template <typename C>
__device__ __forceinline__ void swiglu(
    const globals<C> &g,
    int task_idx,
    uint64_t smem_base_addr,
    semaphore (&inputs_arrived)[C::LOAD_PIPE_DEPTH]
) {
    using G = globals<C>;

    const int cta_rank = cluster_ctarank();
    const int intermediate_col_blocks = g.w_gate.rows() / C::Nb;
    const int projection_tasks = (g.x.rows() / C::Mb) * intermediate_col_blocks;
    const int row_blocks = g.hidden.rows() / G::swiglu_tile::rows;
    const int col_blocks = g.hidden.cols() / G::swiglu_tile::cols;
    const int num_tiles = row_blocks * col_blocks;
    const int first_tile_idx = task_idx * C::CLUSTER_SIZE * 3 + cta_rank * 3;
    if (first_tile_idx >= num_tiles)
        return;

    typename G::swiglu_tile (&a_smem)[3] = *reinterpret_cast<typename G::swiglu_tile (*)[3]>(smem_base_addr);
    typename G::swiglu_tile (&b_smem)[3] = *reinterpret_cast<typename G::swiglu_tile (*)[3]>(smem_base_addr + sizeof(a_smem));

    int first_row, first_col;
    if (threadIdx.x == 0) {
        first_row = first_tile_idx / col_blocks;
        first_col = first_tile_idx % col_blocks;
        #pragma unroll
        for (int stage = 0; stage < 3; ++stage) {
            const int tile_idx = first_tile_idx + stage;
            if (tile_idx < num_tiles) {
                int row = first_row;
                int col = first_col + stage;
                if (col >= col_blocks) {
                    ++row;
                    col -= col_blocks;
                }
                const int parent_task_idx = (row / 2) * intermediate_col_blocks + col / 2;
                while (atomicAdd(&g.counters[{parent_task_idx}], 0) < C::CLUSTER_SIZE)
                    __nanosleep(64);
                while (atomicAdd(&g.counters[{projection_tasks + parent_task_idx}], 0) < C::CLUSTER_SIZE)
                    __nanosleep(64);
                init_semaphore(inputs_arrived[stage], 0, 1);
                tma::expect_bytes(inputs_arrived[stage], sizeof(a_smem[stage]) + sizeof(b_smem[stage]));
                tma::load_async(a_smem[stage], g.gate, {row, col}, inputs_arrived[stage]);
                tma::load_async(b_smem[stage], g.up, {row, col}, inputs_arrived[stage]);
            }
        }
    }
    __syncthreads();

    using compute_group = group<C::NUM_WARPS>;
    rt_fl<C::Mb / 2 / C::NUM_WARPS, C::Nb / 2> gate, up, denominator;
    #pragma unroll
    for (int stage = 0; stage < 3; ++stage) {
        const int tile_idx = first_tile_idx + stage;
        if (tile_idx < num_tiles) {
            wait(inputs_arrived[stage], 0);
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
        __threadfence();
        #pragma unroll
        for (int stage = 0; stage < 3; ++stage) {
            const int tile_idx = first_tile_idx + stage;
            if (tile_idx < num_tiles) {
                int row = first_row;
                int col = first_col + stage;
                if (col >= col_blocks)
                    ++row;
                atomicAdd(&g.counters[{2 * projection_tasks + row / 2}], 1);
            }
        }
    }
}

template <typename C>
__device__ __forceinline__ void expert_grouped_gemm(
    const globals<C> &g,
    int task_idx,
    int kind,
    uint64_t smem_base_addr,
    uint32_t &tmem_addr,
    semaphore &tmem_provisioned,
    semaphore &tmem_finished,
    semaphore (&inputs_arrived)[C::LOAD_PIPE_DEPTH],
    semaphore (&inputs_finished)[C::LOAD_PIPE_DEPTH],
    semaphore &outputs_arrived,
    semaphore &outputs_finished,
    uint32_t &bitfield
) {
    using G = globals<C>;

    typename G::a_tile (&a_smem)[C::LOAD_PIPE_DEPTH] = *reinterpret_cast<typename G::a_tile (*)[C::LOAD_PIPE_DEPTH]>(smem_base_addr);
    typename G::b_tile (&b_smem)[C::LOAD_PIPE_DEPTH] = *reinterpret_cast<typename G::b_tile (*)[C::LOAD_PIPE_DEPTH]>(smem_base_addr + sizeof(a_smem));
    typename G::d_tile (&d_smem)[C::NUM_D_TILES] = *reinterpret_cast<typename G::d_tile (*)[C::NUM_D_TILES]>(smem_base_addr + sizeof(a_smem) + sizeof(b_smem));

    const bool is_gate = kind == 0;
    const bool is_up = kind == 1;
    const bool is_down = kind == 2;
    const int cta_rank = cluster_ctarank();
    const int projection_tasks = (g.x.rows() / C::Mb) * (g.w_gate.rows() / C::Nb);

    const typename G::activation_gl &a_gl = is_down ? g.hidden : g.x;
    const typename G::weight_gl &b_gl = is_gate ? g.w_gate : (is_up ? g.w_up : g.w_down);
    const typename G::activation_gl &d_gl = is_gate ? g.gate : (is_up ? g.up : g.y);
    const int iters_per_task = a_gl.cols() / C::Kb;
    const int col_blocks = b_gl.rows() / C::Nb;

    int expert_task_idx = task_idx;
    int row_block_offset = 0;
    int3 tile_coord = {-1, -1, -1};
    for (int expert_idx = 0; expert_idx < b_gl.depth(); ++expert_idx) {
        const int row_blocks = g.tokens_per_expert[{expert_idx}] / C::Mb;
        const int num_tasks = row_blocks * col_blocks;
        if (expert_task_idx < num_tasks) {
            const int2 swizzled = get_swizzled_2d_idx<C::SUPERGROUP_SIZE>(row_blocks, col_blocks, expert_task_idx);
            tile_coord = {row_block_offset + swizzled.x, swizzled.y, expert_idx};
            break;
        }
        expert_task_idx -= num_tasks;
        row_block_offset += row_blocks;
    }
    if (tile_coord.z < 0)
        return;

    if (is_down) {
        const int swiglu_tiles_per_row_block = (C::Mb / G::swiglu_tile::rows) * (g.hidden.cols() / G::swiglu_tile::cols);
        if (threadIdx.x == 0) {
            while (atomicAdd(&g.counters[{2 * projection_tasks + tile_coord.x}], 0) < swiglu_tiles_per_row_block)
                __nanosleep(64);
        }
        __syncthreads();
    }

    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_alloc{};
    using d_tt_t = tt<float, C::Mb / 2, C::Nb>;

    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(tmem_finished, 0, 1);
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        init_semaphore(outputs_arrived, 0, 1);
        init_semaphore(outputs_finished, 0, C::CLUSTER_SIZE);
    }
    everyone::tma::cluster::arrive_aligned();

    if (warpgroup::groupid() == C::NUM_CONSUMERS) {
        warpgroup::increase_registers<256>();

        if (warpgroup::warpid() == 3 && warp::elect_leader()) {
            int input_ring = 0;
            everyone::tma::cluster::wait();
            for (int idx = 0; idx < iters_per_task; ++idx) {
                wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                tma::cluster::load_async(a_smem[input_ring], a_gl, {tile_coord.x * 2 + cta_rank, idx}, inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                tma::cluster::load_async(b_smem[input_ring], b_gl, {tile_coord.z, tile_coord.y * 2 + cta_rank, idx}, inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                update_phasebit<1>(bitfield, input_ring);
                input_ring = ring_advance<C::LOAD_PIPE_DEPTH>(input_ring);
            }
        } else if (cta_rank == 0 && warpgroup::warpid() == 0 && warp::elect_leader()) {
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_alloc.set_addr(tmem_addr);
            d_tt_t d_tt = tm_alloc.template allocate<d_tt_t>(0);
            int input_ring = 0;
            wait(outputs_finished, 1);
            for (int idx = 0; idx < iters_per_task; ++idx) {
                tma::expect_bytes(inputs_arrived[input_ring], C::CLUSTER_SIZE * sizeof(typename G::a_tile) + 2 * sizeof(typename G::b_tile));
                wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
                if (idx == 0) mm2_ABt (d_tt, a_smem[input_ring], b_smem[input_ring], inputs_finished[input_ring]);
                else          mma2_ABt(d_tt, a_smem[input_ring], b_smem[input_ring], inputs_finished[input_ring]);
                update_phasebit<0>(bitfield, input_ring);
                input_ring = ring_advance<C::LOAD_PIPE_DEPTH>(input_ring);
            }
            detail::tcgen05::commit<C::CLUSTER_SIZE>(outputs_arrived);
        }
    } else {
        using epilogue_group = group<WARPGROUP_WARPS>;
        warpgroup::increase_registers<256>();
        everyone::tma::cluster::wait_aligned();
        if (epilogue_group::warpid() == 0) {
            tm_alloc.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_alloc.set_addr(tmem_addr);
        d_tt_t d_tt = tm_alloc.template allocate<d_tt_t>(0);
        wait(outputs_arrived, 0);
        rt_bf<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> d_reg[C::EPI_PIPE_DEPTH];
        #pragma unroll
        for (int i = 0; i < C::EPI_PIPE_DEPTH; ++i)
            warpgroup::load_async(d_reg[i], d_tt.template subtile<tt<float, C::Mb / 2, C::Nb / C::EPI_PIPE_DEPTH>>(0, C::Nb / C::EPI_PIPE_DEPTH * i));
        tensor_load_wait();
        warpgroup::sync(1);
        warpgroup::tma::cluster::arrive(outputs_finished, 0);
        #pragma unroll
        for (int i = 0; i < C::EPI_PIPE_DEPTH; ++i) {
            warpgroup::tma::store_async_read_wait<C::NUM_D_TILES - 1>();
            warpgroup::sync(1);
            warpgroup::store(d_smem[i % C::NUM_D_TILES], d_reg[i]);
            warpgroup::sync(1);
            warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(d_gl, d_smem[i % C::NUM_D_TILES], {2 * tile_coord.x + cta_rank, C::EPI_PIPE_DEPTH * tile_coord.y + i});
        }
        epilogue_group::sync(4);
        if (epilogue_group::warpid() == 0) {
            if (warp::elect_leader()) {
                tma::store_async_wait();
                if (!is_down) {
                    __threadfence();
                    const int counter_task_idx = tile_coord.x * col_blocks + tile_coord.y;
                    atomicAdd(&g.counters[{(is_gate ? 0 : projection_tasks) + counter_task_idx}], 1);
                }
                tma::cluster::arrive(tmem_finished, 1 - cta_rank);
            }
            wait(tmem_finished, 0);
            tm_alloc.deprovision();
        }
    }
}

template <typename C>
__device__ __forceinline__ void moe_swiglu_kernel(const globals<C> &g) {
    using G = globals<C>;

    extern __shared__ int __shm[];
    const uint64_t smem_base_addr = (reinterpret_cast<uint64_t>(&__shm[0]) + 1023) & ~uint64_t(1023);

    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned, tmem_finished;
    __shared__ semaphore inputs_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived, outputs_finished;

    uint32_t bitfield = 0xFFFF0000;

    const int cluster_idx = clusterIdx().x;
    const int row_blocks = g.x.rows() / C::Mb;
    const int projection_tasks = row_blocks * (g.w_gate.rows() / C::Nb);
    const int swiglu_tiles = (g.hidden.rows() / G::swiglu_tile::rows) * (g.hidden.cols() / G::swiglu_tile::cols);
    const int swiglu_tasks = (swiglu_tiles + C::CLUSTER_SIZE * 3 - 1) / (C::CLUSTER_SIZE * 3);

    if (cluster_idx < projection_tasks) {
        expert_grouped_gemm<C>(
            g, cluster_idx, 0, smem_base_addr, tmem_addr, tmem_provisioned, tmem_finished,
            inputs_arrived, inputs_finished, outputs_arrived, outputs_finished, bitfield
        );
        return;
    }
    if (cluster_idx < 2 * projection_tasks) {
        expert_grouped_gemm<C>(
            g, cluster_idx - projection_tasks, 1, smem_base_addr, tmem_addr, tmem_provisioned,
            tmem_finished, inputs_arrived, inputs_finished, outputs_arrived,
            outputs_finished, bitfield
        );
        return;
    }
    if (cluster_idx < 2 * projection_tasks + swiglu_tasks) {
        swiglu<C>(g, cluster_idx - 2 * projection_tasks, smem_base_addr, inputs_arrived);
        return;
    }
    expert_grouped_gemm<C>(
        g, cluster_idx - 2 * projection_tasks - swiglu_tasks, 2, smem_base_addr, tmem_addr,
        tmem_provisioned, tmem_finished, inputs_arrived, inputs_finished,
        outputs_arrived, outputs_finished, bitfield
    );
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
    at::Tensor counters = at::zeros({2 * gate_up_tasks + row_blocks}, tokens_per_expert.options());

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
