#include "kittens.cuh"

using namespace kittens;

template <int _Mb, int _Nb, int _Kb, int _SUPERGROUP_SIZE, bool _OVERLAP_MMA_EPI, int _LOAD_PIPE_DEPTH, int _EPI_PIPE_DEPTH>
struct config {
    static_assert(_Mb == 256, "Mb must be 256");
    static_assert(_Nb >= 16 && _Nb <= 256 && _Nb % 16 == 0, "Nb must be 16, 32, ..., 256");
    static_assert(_Kb >= 16 && _Kb % 16 == 0, "Kb must be a multiple of 16");
    static_assert(_SUPERGROUP_SIZE >= 1 && _SUPERGROUP_SIZE <= 16, "SUPERGROUP_SIZE must be 1-16");
    static_assert(_LOAD_PIPE_DEPTH >= 1 && _LOAD_PIPE_DEPTH <= 16, "LOAD_PIPE_DEPTH must be 1-16");
    static_assert(_EPI_PIPE_DEPTH >= 1 && _EPI_PIPE_DEPTH <= 16, "EPI_PIPE_DEPTH must be 1-16");

    static constexpr int Mb = _Mb; // per MMA
    static constexpr int Nb = _Nb; // per MMA
    static constexpr int Kb = _Kb;
    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;

    static constexpr bool OVERLAP_MMA_EPI = _OVERLAP_MMA_EPI;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int MMA_PIPE_DEPTH = OVERLAP_MMA_EPI ? 2 : 1;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr int CLC_PIPE_DEPTH = 1;

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int NUM_CONSUMERS = OVERLAP_MMA_EPI ? 1 : 2;
    static constexpr int NUM_PRODUCERS = 1;
    static constexpr int NUM_WARPS = (NUM_CONSUMERS + NUM_PRODUCERS) * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int NUM_D_TILES = EPI_PIPE_DEPTH > 1 ? 2 : 1;
};

template <typename C> using a_tile = st_bf<C::Mb/2, C::Kb>;
template <typename C> using b_tile = st_bf<C::Nb/2, C::Kb>;
template <typename C> using d_tile = st_bf<C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>;

template <typename C> using a_gl = gl<bf16, -1, -1, -1, -1, a_tile<C>>;
template <typename C> using b_gl = gl<bf16, -1, -1, -1, -1, b_tile<C>>;
template <typename C> using d_gl = gl<bf16, -1, -1, -1, -1, d_tile<C>>;

template <typename C>
__launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel(const __grid_constant__ a_gl<C> a, const __grid_constant__ b_gl<C> b, const __grid_constant__ d_gl<C> d) {
    if (threadIdx.x == 0) {
        a.template prefetch_tma<a_tile<C>>();
        b.template prefetch_tma<b_tile<C>>();
        d.template prefetch_tma<d_tile<C>>();
    }

    const int cta_rank = cluster_ctarank();
    const int iters_per_task = a.cols() / C::Kb;
    const int rblks = d.rows() / (C::Mb * C::NUM_CONSUMERS);
    const int cblks = d.cols() / C::Nb;

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    a_tile<C> (&a_smem)[C::LOAD_PIPE_DEPTH][C::NUM_CONSUMERS] = al.allocate<a_tile<C>, C::LOAD_PIPE_DEPTH, C::NUM_CONSUMERS>();
    b_tile<C> (&b_smem)[C::LOAD_PIPE_DEPTH] = al.allocate<b_tile<C>, C::LOAD_PIPE_DEPTH>();
    d_tile<C> (&d_smem)[C::NUM_CONSUMERS][C::NUM_D_TILES] = al.allocate<d_tile<C>, C::NUM_CONSUMERS, C::NUM_D_TILES>();

    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_alloc{};
    using d_tt_t = tt<float, C::Mb/2, C::Nb>;

    __shared__ uint32_t tmem_addr;
    __shared__ clc::handle clc_handle[C::CLC_PIPE_DEPTH];
    __shared__ semaphore tmem_provisioned, tmem_finished, schedule_arrived[C::CLC_PIPE_DEPTH], schedule_finished[C::CLC_PIPE_DEPTH];
    __shared__ semaphore inputs_arrived[C::LOAD_PIPE_DEPTH], inputs_finished[C::LOAD_PIPE_DEPTH], outputs_arrived[C::NUM_CONSUMERS], outputs_finished[C::MMA_PIPE_DEPTH];
    uint32_t bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s

    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(tmem_finished, 0, 1);
        #pragma unroll
        for (int i = 0; i < C::CLC_PIPE_DEPTH; i++) {
            init_semaphore(schedule_arrived[i], 0, 1);
            init_semaphore(schedule_finished[i], 0, (2+C::NUM_CONSUMERS)*C::CLUSTER_SIZE+C::NUM_CONSUMERS);
        }
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; i++) {
            init_semaphore(inputs_arrived[i], 0, C::NUM_CONSUMERS);
            init_semaphore(inputs_finished[i], 0, C::NUM_CONSUMERS);
        }
        #pragma unroll
        for (int i = 0; i < C::NUM_CONSUMERS; i++) {
            init_semaphore(outputs_arrived[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < C::MMA_PIPE_DEPTH; i++) {
            init_semaphore(outputs_finished[i], 0, C::CLUSTER_SIZE*C::NUM_CONSUMERS);
        }
    }
    everyone::tma::cluster::arrive_aligned();

    if (warpgroup::groupid() == C::NUM_CONSUMERS) {
        warpgroup::decrease_registers<56>();

        if (warpgroup::warpid() == 3 && warp::elect_leader()) {
            int input_ring = 0;
            int2 tile_coord = get_swizzled_2d_idx<C::SUPERGROUP_SIZE>(rblks, cblks, blockIdx.x/C::CLUSTER_SIZE);
            pdl::wait();
            everyone::tma::cluster::wait();
            for (int task_iter = 0; true; task_iter++) {
                for (int idx = 0; idx < iters_per_task; idx++) {
                    wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                    #pragma unroll
                    for (int i = 0; i < C::NUM_CONSUMERS; i++)
                        tma::cluster::load_async(a_smem[input_ring][i], a, {(tile_coord.x*2+cta_rank)*C::NUM_CONSUMERS+i, idx}, inputs_arrived[input_ring], (uint16_t)(1<<cta_rank), 0);
                    tma::cluster::load_async(b_smem[input_ring], b, {tile_coord.y*2+cta_rank, idx}, inputs_arrived[input_ring], (uint16_t)(1<<cta_rank), 0);
                    update_phasebit<1>(bitfield, input_ring);
                    input_ring=ring_advance<C::LOAD_PIPE_DEPTH>(input_ring);
                }
                wait(schedule_arrived[task_iter%C::CLC_PIPE_DEPTH], (task_iter/C::CLC_PIPE_DEPTH)%2);
                auto schedule = clc::query(clc_handle[task_iter%C::CLC_PIPE_DEPTH]);
                tma::cluster::arrive(schedule_finished[task_iter%C::CLC_PIPE_DEPTH], 0);
                if (schedule.success) tile_coord = get_swizzled_2d_idx<C::SUPERGROUP_SIZE>(rblks, cblks, schedule.x/C::CLUSTER_SIZE);
                else break;
            }
        } else if (warpgroup::warpid() == 2 && warp::elect_leader()) {
            everyone::tma::cluster::wait();
            for (int task_iter = 0; true; task_iter++) {
                if (cta_rank == 0) {
                    wait(schedule_finished[task_iter%C::CLC_PIPE_DEPTH], ((task_iter+C::CLC_PIPE_DEPTH)/C::CLC_PIPE_DEPTH)%2);
                    clc::schedule(clc_handle[task_iter%C::CLC_PIPE_DEPTH], schedule_arrived[task_iter%C::CLC_PIPE_DEPTH]);
                }
                tma::expect_bytes(schedule_arrived[task_iter%C::CLC_PIPE_DEPTH], sizeof(clc_handle[task_iter%C::CLC_PIPE_DEPTH]));
                wait(schedule_arrived[task_iter%C::CLC_PIPE_DEPTH], (task_iter/C::CLC_PIPE_DEPTH)%2);
                auto schedule = clc::query(clc_handle[task_iter%C::CLC_PIPE_DEPTH]);
                tma::cluster::arrive(schedule_finished[task_iter%C::CLC_PIPE_DEPTH], 0);
                if (!schedule.success) break;
            }
        } else if (cta_rank == 0 && warpgroup::warpid() < C::NUM_CONSUMERS && warp::elect_leader()) {
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_alloc.set_addr(tmem_addr);
            d_tt_t d_tt[C::MMA_PIPE_DEPTH];
            #pragma unroll
            for (int i = 0; i < C::MMA_PIPE_DEPTH; i++) {
                if constexpr(C::Mb == 256) d_tt[i] = tm_alloc.template allocate<d_tt_t>(   (i+warpgroup::warpid())*C::Nb);
                else                       d_tt[i] = tm_alloc.template allocate<d_tt_t>(0, (i+warpgroup::warpid())*C::Nb);
            }
            int input_ring = 0;
            for (int task_iter = 0; true; task_iter++) {
                wait(schedule_arrived[task_iter%C::CLC_PIPE_DEPTH], (task_iter/C::CLC_PIPE_DEPTH)%2);
                auto schedule = clc::query(clc_handle[task_iter%C::CLC_PIPE_DEPTH]);
                tma::cluster::arrive(schedule_finished[task_iter%C::CLC_PIPE_DEPTH], 0);
                wait(outputs_finished[task_iter%C::MMA_PIPE_DEPTH], ((task_iter+C::MMA_PIPE_DEPTH)/C::MMA_PIPE_DEPTH)%2);
                for(int idx = 0; idx < iters_per_task; idx++) {
                    tma::expect_bytes(inputs_arrived[input_ring], (C::CLUSTER_SIZE*C::NUM_CONSUMERS*sizeof(a_tile<C>) + 2*sizeof(b_tile<C>))/C::NUM_CONSUMERS);
                    wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
                    if (idx == 0) mm2_ABt (d_tt[task_iter%C::MMA_PIPE_DEPTH], a_smem[input_ring][warpgroup::warpid()], b_smem[input_ring], inputs_finished[input_ring]);
                    else          mma2_ABt(d_tt[task_iter%C::MMA_PIPE_DEPTH], a_smem[input_ring][warpgroup::warpid()], b_smem[input_ring], inputs_finished[input_ring]);
                    update_phasebit<0>(bitfield, input_ring);
                    input_ring=ring_advance<C::LOAD_PIPE_DEPTH>(input_ring);
                }
                detail::tcgen05::commit<C::CLUSTER_SIZE>(outputs_arrived[warpgroup::warpid()]);
                if (!schedule.success) break;
            }
        }
    }
    else {
        using epilogue_group = group<WARPGROUP_WARPS*C::NUM_CONSUMERS>;
        if constexpr (!C::OVERLAP_MMA_EPI)
            warpgroup::increase_registers<224>();
        everyone::tma::cluster::wait_aligned();
        if (epilogue_group::warpid() == 0) {
            tm_alloc.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_alloc.set_addr(tmem_addr);
        d_tt_t d_tt[C::MMA_PIPE_DEPTH];
        #pragma unroll
        for (int i = 0; i < C::MMA_PIPE_DEPTH; i++) {
            if constexpr(C::Mb == 256) d_tt[i] = tm_alloc.template allocate<d_tt_t>(   (i+warpgroup::groupid())*C::Nb);
            else                       d_tt[i] = tm_alloc.template allocate<d_tt_t>(0, (i+warpgroup::groupid())*C::Nb);
        }
        int2 tile_coord, next_tile_coord = get_swizzled_2d_idx<C::SUPERGROUP_SIZE>(rblks, cblks, blockIdx.x/C::CLUSTER_SIZE);
        for(int task_iter = 0; true; task_iter++) {
            tile_coord = next_tile_coord;
            wait(schedule_arrived[task_iter%C::CLC_PIPE_DEPTH], (task_iter/C::CLC_PIPE_DEPTH)%2);
            auto schedule = clc::query(clc_handle[task_iter%C::CLC_PIPE_DEPTH]);
            warpgroup::sync(warpgroup::groupid()+1);
            warpgroup::tma::cluster::arrive(schedule_finished[task_iter%C::CLC_PIPE_DEPTH], 0);
            if (schedule.success) next_tile_coord = get_swizzled_2d_idx<C::SUPERGROUP_SIZE>(rblks, cblks, schedule.x/C::CLUSTER_SIZE);
            wait(outputs_arrived[warpgroup::groupid()], task_iter%2);
            if constexpr (C::OVERLAP_MMA_EPI) {
                rt_bf<C::Mb/8, C::Nb/C::EPI_PIPE_DEPTH> d_reg;
                #pragma unroll
                for(int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                    warpgroup::load_async(d_reg, d_tt[task_iter%C::MMA_PIPE_DEPTH].template subtile<tt<float, C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>>(0, C::Nb/C::EPI_PIPE_DEPTH*i));
                    if (i == C::EPI_PIPE_DEPTH - 1) {
                        tensor_load_wait();
                        warpgroup::sync(warpgroup::groupid()+1);
                        if (!schedule.success) warpgroup::pdl::arrive();
                        warpgroup::tma::cluster::arrive(outputs_finished[task_iter%C::MMA_PIPE_DEPTH], 0);
                    }
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                    warpgroup::sync(warpgroup::groupid()+1);
                    warpgroup::store(d_smem[warpgroup::groupid()][i%C::NUM_D_TILES], d_reg);
                    warpgroup::sync(warpgroup::groupid()+1);
                    warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(d, d_smem[warpgroup::groupid()][i%C::NUM_D_TILES], {(2*tile_coord.x+cta_rank)*C::NUM_CONSUMERS+warpgroup::groupid(), C::EPI_PIPE_DEPTH*tile_coord.y+i});
                }
            } else {
                rt_bf<C::Mb/8, C::Nb/C::EPI_PIPE_DEPTH> d_reg[C::EPI_PIPE_DEPTH];
                #pragma unroll
                for(int i = 0; i < C::EPI_PIPE_DEPTH; i++)
                    warpgroup::load_async(d_reg[i], d_tt[task_iter%C::MMA_PIPE_DEPTH].template subtile<tt<float, C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>>(0, C::Nb/C::EPI_PIPE_DEPTH*i));
                tensor_load_wait();
                warpgroup::sync(warpgroup::groupid()+1);
                if (!schedule.success) warpgroup::pdl::arrive();
                warpgroup::tma::cluster::arrive(outputs_finished[task_iter%C::MMA_PIPE_DEPTH], 0);
                #pragma unroll
                for(int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                    warpgroup::sync(warpgroup::groupid()+1);
                    warpgroup::store(d_smem[warpgroup::groupid()][i%C::NUM_D_TILES], d_reg[i]);
                    warpgroup::sync(warpgroup::groupid()+1);
                    warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(d, d_smem[warpgroup::groupid()][i%C::NUM_D_TILES], {(2*tile_coord.x+cta_rank)*C::NUM_CONSUMERS+warpgroup::groupid(), C::EPI_PIPE_DEPTH*tile_coord.y+i});
                }
            }
            if (!schedule.success) break;
        }
        epilogue_group::sync(4);
        if (epilogue_group::warpid() == 0) {
            if (warp::elect_leader()) tma::cluster::arrive(tmem_finished, 1-cta_rank);
            wait(tmem_finished, 0);
            tm_alloc.deprovision();
        }
    }
}
