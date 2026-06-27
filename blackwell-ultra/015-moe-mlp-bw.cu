#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

namespace moe_mlper {

struct config {
    static constexpr int Mb = 256;
    static constexpr int Nb = 256;
    static constexpr int Kb = 64;
    static constexpr int SUPERGROUP_SIZE = 8;

    static constexpr int LOAD_PIPE_DEPTH = 5;
    static constexpr int EPI_PIPE_DEPTH = 4;

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int NUM_CONSUMERS = 1;
    static constexpr int NUM_PRODUCERS = 1;
    static constexpr int NUM_WARPS = (NUM_CONSUMERS + NUM_PRODUCERS) * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int NUM_D_TILES = EPI_PIPE_DEPTH > 1 ? 2 : 1;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1024;
};

template <typename C>
struct globals {
    using a_tile = st_bf<C::Mb / 2, C::Kb>;
    using b_tile = st_bf<C::Nb / 2, C::Kb>;
    using d_tile = st_bf<C::Mb / 2, C::Nb / C::EPI_PIPE_DEPTH>;

    using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<bf16, 1, -1, -1, -1, b_tile>;
    using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;
    using index_gl = gl<int, 1, 1, 1, -1>;

    a_gl a;                       // (M, K)
    b_gl b;                       // (E, N, K)
    index_gl tokens_per_expert;   // (E,)
    d_gl d;                       // (M, N)

    __host__ __inline__ dim3 grid() const { return dim3(d.rows() / (C::Mb / 2) * d.cols() / C::Nb); }
};

template <typename C>
__device__ inline void moe_mlp_kernel(const globals<C> &g) {
    using G = globals<C>;

    const int cta_rank = cluster_ctarank();
    const int cluster_idx = clusterIdx().x;
    const int iters_per_task = g.a.cols() / C::Kb;
    const int col_blocks = g.d.cols() / C::Nb;

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);

    typename G::a_tile (&a_smem)[C::LOAD_PIPE_DEPTH] = al.allocate<G::a_tile, C::LOAD_PIPE_DEPTH>();
    typename G::b_tile (&b_smem)[C::LOAD_PIPE_DEPTH] = al.allocate<G::b_tile, C::LOAD_PIPE_DEPTH>();
    typename G::d_tile (&d_smem)[C::NUM_D_TILES] = al.allocate<G::d_tile, C::NUM_D_TILES>();

    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_alloc{};
    using d_tt_t = tt<float, C::Mb / 2, C::Nb>;

    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned, tmem_finished;
    __shared__ semaphore inputs_arrived[C::LOAD_PIPE_DEPTH], inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived, outputs_finished;
    uint32_t bitfield = 0xFFFF0000;

    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(tmem_finished, 0, 1);
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; i++) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        init_semaphore(outputs_arrived, 0, 1);
        init_semaphore(outputs_finished, 0, C::CLUSTER_SIZE);
    }
    everyone::tma::cluster::arrive_aligned();

    int task_idx = cluster_idx;
    int row_block_offset = 0;
    int3 tile_coord = {-1, -1, -1};
    for (int expert_idx = 0; expert_idx < g.b.depth(); expert_idx++) {
        const int row_blocks = g.tokens_per_expert[{expert_idx}] / C::Mb;
        const int num_tasks = row_blocks * col_blocks;
        if (task_idx < num_tasks) {
            const int2 swizzled = get_swizzled_2d_idx<C::SUPERGROUP_SIZE>(row_blocks, col_blocks, task_idx);
            tile_coord = {row_block_offset + swizzled.x, swizzled.y, expert_idx};
            break;
        }
        task_idx -= num_tasks;
        row_block_offset += row_blocks;
    }

    if (warpgroup::groupid() == C::NUM_CONSUMERS) {
        warpgroup::increase_registers<256>();

        if (warpgroup::warpid() == 3 && warp::elect_leader()) {
            int input_ring = 0;
            everyone::tma::cluster::wait();
            for (int idx = 0; idx < iters_per_task; idx++) {
                wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                tma::cluster::load_async(a_smem[input_ring], g.a, {tile_coord.x * 2 + cta_rank, idx}, inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                tma::cluster::load_async(b_smem[input_ring], g.b, {tile_coord.z, tile_coord.y * 2 + cta_rank, idx}, inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                update_phasebit<1>(bitfield, input_ring);
                input_ring = ring_advance<C::LOAD_PIPE_DEPTH>(input_ring);
            }
        } else if (cta_rank == 0 && warpgroup::warpid() == 0 && warp::elect_leader()) {
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_alloc.set_addr(tmem_addr);
            d_tt_t d_tt = tm_alloc.template allocate<d_tt_t>(0);
            int input_ring = 0;
            wait(outputs_finished, 1); // noop, left here for future expansion
            for (int idx = 0; idx < iters_per_task; idx++) {
                tma::expect_bytes(inputs_arrived[input_ring], C::CLUSTER_SIZE * sizeof(G::a_tile) + 2 * sizeof(G::b_tile));
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
        for (int i = 0; i < C::EPI_PIPE_DEPTH; i++)
            warpgroup::load_async(d_reg[i], d_tt.template subtile<tt<float, C::Mb / 2, C::Nb / C::EPI_PIPE_DEPTH>>(0, C::Nb / C::EPI_PIPE_DEPTH * i));
        tensor_load_wait();
        warpgroup::sync(1);
        warpgroup::tma::cluster::arrive(outputs_finished, 0);
        #pragma unroll
        for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
            warpgroup::tma::store_async_read_wait<C::NUM_D_TILES - 1>();
            warpgroup::sync(1);
            warpgroup::store(d_smem[i % C::NUM_D_TILES], d_reg[i]);
            warpgroup::sync(1);
            warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(g.d, d_smem[i % C::NUM_D_TILES], {2 * tile_coord.x + cta_rank, C::EPI_PIPE_DEPTH * tile_coord.y + i});
        }
        epilogue_group::sync(4);
        if (epilogue_group::warpid() == 0) {
            if (warp::elect_leader()) tma::cluster::arrive(tmem_finished, 1 - cta_rank);
            wait(tmem_finished, 0);
            tm_alloc.deprovision();
        }
    }
}

void moe_mlp(
    const at::Tensor &a, const at::Tensor &b, at::Tensor &d, const at::Tensor &tokens_per_expert
) {
    using C = config;
    using G = globals<C>;

    G g {
        .a = kittens::py::tensor_to_gl<G::a_gl>(a),
        .b = kittens::py::tensor_to_gl<G::b_gl>(b),
        .tokens_per_expert = kittens::py::tensor_to_gl<G::index_gl>(tokens_per_expert),
        .d = kittens::py::tensor_to_gl<G::d_gl>(d),
    };

    kittens::py::launch_kernel<C, G, moe_mlp_kernel<C>>(g);
}

} // namespace moe_mlper

PYBIND11_MODULE(_C, m) {
    m.def("moe_mlp", &moe_mlper::moe_mlp, "MoE MLP matrix multiply", 
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("d"), pybind11::arg("tokens_per_expert"));
}
