/*
    WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING
    The forward kernel has race condition.
    It's a rewrite of (poorly written) TK B200 MHA implementation anyways.
    I would rather gnore this file.
    Start with 51-bf16-mha-opt-D128.cu instead.

    Benchmarks (B=8 N=8192 H=128 D=128)

    Original ThunderKittens B200 MHA Implementation:
      - FWD: 273.33 TFLOP/s
      - BWD Prep: 6298.60 GB/s
      - BWD: 650.11 TFLOP/s

    This version:
      - FWD:
            Two consumers: 858.63 TFLOP/s
            One consumer : 825.12 TFLOP/s
            --> Having two consumers do not dramatically improve perf. In fact, I think there are more opportunities from free TMEM
      - BWD Prep: 7184.93 GB/s
      - BWD: 608.22 TFLOP/s

    Not the best, but a reasonable starting point.
*/

#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace bf16_mha_fwd {

struct config {
    static constexpr int CLUSTER_SIZE = 2;

    static constexpr int NUM_BLOCKS = 148;
    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int NUM_CONSUMERS = 2;
    static constexpr int WARPGROUPS_PER_CONSUMER = 2;
    static constexpr int NUM_PRODUCERS = 1;
    static constexpr int NUM_WARPGROUPS = NUM_CONSUMERS * WARPGROUPS_PER_CONSUMER + NUM_PRODUCERS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int PRODUCER_REGISTERS = 64;
    static constexpr int CONSUMER_REGISTERS = 104;
};

struct globals {
    static constexpr int BLOCK_SIZE = 128;

    static constexpr int QK_DIM = 128;
    static constexpr int VO_DIM = 128;

    static constexpr int PIPELINE_STAGES = 2;

    using Q_tile = st_bf<BLOCK_SIZE, QK_DIM>;
    using K_tile = st_bf<BLOCK_SIZE / 2, QK_DIM>;
    using V_tile = st_bf<BLOCK_SIZE, VO_DIM / 2>;
    using L_vec  = col_vec<st_fl<BLOCK_SIZE, VO_DIM>>;
    using O_tile = st_bf<BLOCK_SIZE, VO_DIM>;
    using A_tile = st_bf<BLOCK_SIZE, BLOCK_SIZE>;

    using Q_gl = gl<bf16,  -1, -1, -1, -1, Q_tile>; // B, H, N, D
    using K_gl = gl<bf16,  -1, -1, -1, -1, K_tile>;
    using V_gl = gl<bf16,  -1, -1, -1, -1, V_tile>;
    using L_gl = gl<float, -1, -1, -1, -1, L_vec>;
    using O_gl = gl<bf16,  -1, -1, -1, -1, O_tile>;

    Q_gl Q;
    K_gl K;
    V_gl V;
    L_gl L;
    O_gl O;

    __host__ inline int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
    __host__ inline dim3 grid()  { return dim3(config::NUM_BLOCKS); }
    __host__ inline dim3 block() { return dim3(config::NUM_THREADS); }

    struct task_info {
        int batch_idx;
        int head_idx;
        int Q_block_idx;
        int KV_block_start;
        int KV_block_end;
    };
};

__device__ static inline globals::task_info get_task_info(const globals &G, int task_idx) {
    constexpr int Q_block_size = 2 * config::NUM_CONSUMERS * globals::BLOCK_SIZE;
    const int num_QO_blocks = G.K.rows() / globals::BLOCK_SIZE;

    const int B = G.Q.batch();
    const int H = G.Q.depth();
    const int num_Q_blocks = (G.Q.rows() + Q_block_size - 1) / Q_block_size;

    globals::task_info task_info;
    task_info.batch_idx = task_idx / (H * num_Q_blocks);
    task_idx -= task_info.batch_idx * H * num_Q_blocks;
    task_info.head_idx  = task_idx / num_Q_blocks;
    task_idx -= task_info.head_idx * num_Q_blocks;
    task_info.Q_block_idx = task_idx;

    task_info.KV_block_start = 0;
    task_info.KV_block_end = num_QO_blocks;

    if (task_info.batch_idx >= B)
        return { -1, -1, -1 };
    else
        return task_info;
}

struct rescale_add {
    template<typename T> static __device__ inline T op(const T &a, const T &b) {
        if constexpr (std::is_same_v<T, float2>) {
            constexpr float2 scale = {1.44269504089f*0.08838834764f, 1.44269504089f*0.08838834764f};
            float2 c;
            asm volatile("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(*(uint64_t*)&c) : "l"(*(uint64_t*)&a), "l"(*(uint64_t*)&scale), "l"(*(uint64_t*)&b));
            return c;
        }
        else {
            static_assert(sizeof(T) == 999, "Currently unsupported type");
        }
    }
};

__global__ __cluster_dims__(config::CLUSTER_SIZE) __launch_bounds__(config::NUM_THREADS, 1)
static void kernel(const __grid_constant__ globals G) {
    // Declare shared memory
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);

    // Allocate shared memory
    static_assert(
        sizeof(globals::Q_tile) * config::NUM_CONSUMERS +
        sizeof(globals::K_tile) * globals::PIPELINE_STAGES +
        sizeof(globals::V_tile) * globals::PIPELINE_STAGES +
        sizeof(globals::L_vec) * config::NUM_CONSUMERS +
        sizeof(globals::A_tile) * config::NUM_CONSUMERS <= config::DYNAMIC_SHARED_MEMORY
    );
    globals::Q_tile (&Q_smem)[config::NUM_CONSUMERS]    = sm_allocator.allocate<globals::Q_tile, config::NUM_CONSUMERS>();
    globals::K_tile (&K_smem)[globals::PIPELINE_STAGES] = sm_allocator.allocate<globals::K_tile, globals::PIPELINE_STAGES>();
    globals::V_tile (&V_smem)[globals::PIPELINE_STAGES] = sm_allocator.allocate<globals::V_tile, globals::PIPELINE_STAGES>();
    globals::L_vec  (&L_smem)[config::NUM_CONSUMERS]    = sm_allocator.allocate<globals::L_vec, config::NUM_CONSUMERS>();
    globals::O_tile (&O_smem)[config::NUM_CONSUMERS]    = *reinterpret_cast<globals::O_tile(*)[config::NUM_CONSUMERS]>(&Q_smem[0]);
    globals::A_tile (&A_smem)[config::NUM_CONSUMERS]    = sm_allocator.allocate<globals::A_tile, config::NUM_CONSUMERS>();

    // Allocate tensor memory
    tensor_allocator<1, config::CLUSTER_SIZE> tm_allocator {};
    using tm_t = tt<float, globals::BLOCK_SIZE, globals::BLOCK_SIZE>;
    tm_t QK_tm[2] = {
        tm_allocator.allocate<tm_t>(globals::BLOCK_SIZE * 0),
        tm_allocator.allocate<tm_t>(globals::BLOCK_SIZE * 1)
    };
    tm_t AV_tm[2] = {
        tm_allocator.allocate<tm_t>(globals::BLOCK_SIZE * 2),
        tm_allocator.allocate<tm_t>(globals::BLOCK_SIZE * 3)
    };

    // Set up mbarriers
    __shared__ semaphore Q_arrived;
    __shared__ semaphore K_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore V_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore K_finished[globals::PIPELINE_STAGES];
    __shared__ semaphore V_finished[globals::PIPELINE_STAGES];
    __shared__ semaphore A_unloaded[config::NUM_CONSUMERS];
    __shared__ semaphore A_loaded[config::NUM_CONSUMERS];
    __shared__ semaphore QK_finished[config::NUM_CONSUMERS];
    __shared__ semaphore AV_finished[config::NUM_CONSUMERS];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < globals::PIPELINE_STAGES; ++i) {
            init_semaphore(K_arrived[i], 0, config::CLUSTER_SIZE);
            init_semaphore(V_arrived[i], 0, config::CLUSTER_SIZE);
            init_semaphore(K_finished[i], 0, 1);
            init_semaphore(V_finished[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < config::NUM_CONSUMERS; ++i) {
            init_semaphore(Q_arrived, 0, config::CLUSTER_SIZE * config::NUM_CONSUMERS);
            init_semaphore(A_unloaded[i], 0, config::CLUSTER_SIZE);
            init_semaphore(A_loaded[i], 0, config::CLUSTER_SIZE);
            init_semaphore(QK_finished[i], 0, 1);
            init_semaphore(AV_finished[i], 0, 1);
        }
    }
    everyone::tma::cluster::sync();

    // Pipeline configuration
    const int cluster_id = clusterIdx().x;
    const int cta_id = cluster_ctarank();

    // Constants
    constexpr float SQRT_D_INV = 0.08838834764f; // 1 / sqrt(128)
    constexpr float NEG_SQRT_D = -11.313708499f; // -sqrt(128)
    constexpr float LOG2E = 1.44269504089f;
    constexpr float NEG_LOGE2 = -0.69314718056f;

    // Main divergence
    if (warpgroup::groupid() == config::NUM_WARPGROUPS - 1) {
        // Producer group
        // warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();
        const int warp_id = warpgroup::warpid();
        const int lane_id = warp::laneid();

        // Declare stage and phasebits for semaphore waits
        int stage = 0;
        uint32_t phasebits = 0xFFFF0000;

        if (warp_id == 0 && lane_id == 0) {
            // K loader
            for (int task_idx = cluster_id; true; task_idx += gridDim.x / 2) {
                globals::task_info task_info = get_task_info(G, task_idx);
                if (task_info.batch_idx == -1) break;

                for (int i = task_info.KV_block_start; i < task_info.KV_block_end; ++i) {
                    wait(K_finished[stage], get_phasebit<1>(phasebits, stage));
                    update_phasebit<1>(phasebits, stage);

                    tma::cluster::expect(K_arrived[stage], 0, K_smem[stage]);
                    tma::cluster::load_async(K_smem[stage], G.K, 
                                             {task_info.batch_idx, task_info.head_idx, 2 * i + cta_id, 0},
                                             K_arrived[stage], (uint16_t)(1 << cta_id), 0);

                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }
            }
        } else if (warp_id == 1 && lane_id == 0) {
            // V loader
            for (int task_idx = cluster_id; true; task_idx += gridDim.x / 2) {
                globals::task_info task_info = get_task_info(G, task_idx);
                if (task_info.batch_idx == -1) break;

                for (int i = task_info.KV_block_start; i < task_info.KV_block_end; ++i) {
                    wait(V_finished[stage], get_phasebit<1>(phasebits, stage));
                    update_phasebit<1>(phasebits, stage);

                    tma::cluster::expect(V_arrived[stage], 0, V_smem[stage]);
                    tma::cluster::load_async(V_smem[stage], G.V, 
                                             {task_info.batch_idx, task_info.head_idx, i, cta_id},
                                             V_arrived[stage], (uint16_t)(1 << cta_id), 0);

                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }
            }
        } else if (cta_id == 0 && warp_id == 2 && lane_id == 0) {
            // QK launcher
            for (int task_idx = cluster_id; true; task_idx += gridDim.x / 2) {
                globals::task_info task_info = get_task_info(G, task_idx);
                if (task_info.batch_idx == -1) break;

                tma::cluster::wait(Q_arrived, get_phasebit<0>(phasebits, globals::PIPELINE_STAGES));
                update_phasebit<0>(phasebits, globals::PIPELINE_STAGES);

                for (int i = task_info.KV_block_start; i < task_info.KV_block_end; ++i) {
                    tma::cluster::wait(K_arrived[stage], get_phasebit<0>(phasebits, stage));
                    update_phasebit<0>(phasebits, stage);

                    #pragma unroll
                    for (int consumer_id = 0; consumer_id < config::NUM_CONSUMERS; ++consumer_id) {
                        tma::cluster::wait(A_unloaded[consumer_id], get_phasebit<1>(phasebits, globals::PIPELINE_STAGES + consumer_id));
                        update_phasebit<1>(phasebits, globals::PIPELINE_STAGES + consumer_id);
                        mm2_ABt(QK_tm[consumer_id], Q_smem[consumer_id], K_smem[stage], QK_finished[consumer_id]);
                    }

                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }
            }
        } else if (cta_id == 0 && warp_id == 3 && lane_id == 0) {
            // AV launcher
            for (int task_idx = cluster_id; true; task_idx += gridDim.x / 2) {
                globals::task_info task_info = get_task_info(G, task_idx);
                if (task_info.batch_idx == -1) break;

                for (int i = task_info.KV_block_start; i < task_info.KV_block_end; ++i) {
                    tma::cluster::wait(V_arrived[stage], get_phasebit<0>(phasebits, stage));
                    update_phasebit<0>(phasebits, stage);

                    #pragma unroll
                    for (int consumer_id = 0; consumer_id < config::NUM_CONSUMERS; ++consumer_id) {
                        tma::cluster::wait(A_loaded[consumer_id], get_phasebit<0>(phasebits, globals::PIPELINE_STAGES + consumer_id));
                        update_phasebit<0>(phasebits, globals::PIPELINE_STAGES + consumer_id);
                        mma2_AB(AV_tm[consumer_id], A_smem[consumer_id], V_smem[stage], AV_finished[consumer_id]);
                    }

                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }
            }
        }
    } else {
        // Consumer group
        // warpgroup::increase_registers<config::CONSUMER_REGISTERS>();
        using all_consumers = group<config::NUM_CONSUMERS * config::WARPGROUPS_PER_CONSUMER * WARPGROUP_WARPS>;
        using consumer = group<config::WARPGROUPS_PER_CONSUMER * WARPGROUP_WARPS>;
        const int consumer_id = consumer::groupid();
        constexpr int ROWS_PER_WARP = globals::BLOCK_SIZE / config::WARPGROUPS_PER_CONSUMER / WARPGROUP_WARPS;

        // Declare stage and phasebits for semaphore waits
        int K_stage = 0;
        int V_stage = 0;
        uint32_t phasebits = 0xFFFF0000;

        for (int task_idx = cluster_id; true; task_idx += gridDim.x / 2) {
            globals::task_info task_info = get_task_info(G, task_idx);
            if (task_info.batch_idx == -1) break;

            // Load Q
            if (consumer::laneid() == 0) {
                tma::cluster::expect(Q_arrived, 0, Q_smem[consumer_id]);
                tma::cluster::load_async(Q_smem[consumer_id], G.Q,
                                         {task_info.batch_idx, task_info.head_idx, 
                                          2 * config::NUM_CONSUMERS * task_info.Q_block_idx + 
                                          config::NUM_CONSUMERS * cta_id + consumer_id, 0},
                                         Q_arrived, (uint16_t)(1 << cta_id), 0);
            }

            rt_fl<ROWS_PER_WARP, globals::VO_DIM> O_reg;
            col_vec<rt_fl<ROWS_PER_WARP, globals::BLOCK_SIZE>> max_vec, norm_vec;
            warp::zero(O_reg);
            warp::neg_infty(max_vec);
            warp::zero(norm_vec);

            for (int i = task_info.KV_block_start; i < task_info.KV_block_end; ++i) {
                tma::cluster::wait(QK_finished[consumer_id], get_phasebit<0>(phasebits, globals::PIPELINE_STAGES + 0));
                update_phasebit<0>(phasebits, globals::PIPELINE_STAGES + 0);
                if (all_consumers::laneid() == 0) arrive(K_finished[K_stage]);
                K_stage = (K_stage + 1) % globals::PIPELINE_STAGES;

                rt_fl<ROWS_PER_WARP, globals::BLOCK_SIZE> A_fl_reg;
                consumer::load_async(A_fl_reg, QK_tm[consumer_id]);
                tensor_load_wait();
                consumer::sync(1 + consumer_id);
                if (consumer::laneid() == 0)
                    tma::cluster::arrive(A_unloaded[consumer_id], 0);

                // Perform softmax
                rt_bf<ROWS_PER_WARP, globals::BLOCK_SIZE> A_bf_reg;
                col_vec<rt_fl<ROWS_PER_WARP, globals::BLOCK_SIZE>> max_vec_last_scaled, max_vec_scaled;
                warp::mul(max_vec_last_scaled, max_vec, LOG2E * SQRT_D_INV);
                warp::row_max(max_vec, A_fl_reg, max_vec);
                warp::mul(max_vec_scaled, max_vec, -LOG2E * SQRT_D_INV);
                warp::row_map<rescale_add>(A_fl_reg, A_fl_reg, max_vec_scaled);
                warp::exp2(A_fl_reg, A_fl_reg);
                warp::add(max_vec_last_scaled, max_vec_scaled, max_vec_last_scaled);
                warp::exp2(max_vec_last_scaled, max_vec_last_scaled);
                warp::mul(norm_vec, max_vec_last_scaled, norm_vec);
                warp::row_sum(norm_vec, A_fl_reg, norm_vec);
                warp::copy(A_bf_reg, A_fl_reg);

                if (i > task_info.KV_block_start) {
                    tma::cluster::wait(AV_finished[consumer_id], get_phasebit<0>(phasebits, globals::PIPELINE_STAGES + 1));
                    update_phasebit<0>(phasebits, globals::PIPELINE_STAGES + 1);
                    if (all_consumers::laneid() == 0) arrive(V_finished[V_stage]);
                    V_stage = (V_stage + 1) % globals::PIPELINE_STAGES;
                }

                consumer::load_async(O_reg, AV_tm[consumer_id]);
                consumer::store(A_smem[consumer_id], A_bf_reg);
                warp::mul_row(O_reg, O_reg, max_vec_last_scaled);
                consumer::store_async(AV_tm[consumer_id], O_reg);
                tensor_store_wait();
                consumer::sync(1 + consumer_id);
                if (consumer::laneid() == 0)
                    tma::cluster::arrive(A_loaded[consumer_id], 0);
            }

            // Wait for the last AV to finished
            tma::cluster::wait(AV_finished[consumer_id], get_phasebit<0>(phasebits, globals::PIPELINE_STAGES + 1));
            update_phasebit<0>(phasebits, globals::PIPELINE_STAGES + 1);
            if (all_consumers::laneid() == 0) arrive(V_finished[V_stage]);
            V_stage = (V_stage + 1) % globals::PIPELINE_STAGES;

            consumer::load_async(O_reg, AV_tm[consumer_id]);
            warp::div_row(O_reg, O_reg, norm_vec);
            consumer::store(O_smem[consumer_id], O_reg);
            consumer::sync(1 + consumer_id);
            if (consumer::laneid() == 0) {
                tma::store_async(G.O, O_smem[consumer_id], 
                                 {task_info.batch_idx, task_info.head_idx, 
                                  2 * config::NUM_CONSUMERS * task_info.Q_block_idx + 
                                  config::NUM_CONSUMERS * cta_id + consumer_id, 0});
            }

            warp::mul(max_vec, max_vec, SQRT_D_INV);
            warp::log(norm_vec, norm_vec);
            warp::add(norm_vec, norm_vec, max_vec);
            warp::mul(norm_vec, norm_vec, NEG_SQRT_D);

            consumer::store(L_smem[consumer_id], norm_vec);
            consumer::sync(1 + consumer_id);
            if (consumer::laneid() == 0) {
                tma::store_async(G.L, L_smem[consumer_id], 
                                 {task_info.batch_idx, task_info.head_idx, 0, 
                                  2 * config::NUM_CONSUMERS * task_info.Q_block_idx + 
                                  config::NUM_CONSUMERS * cta_id + consumer_id});
                tma::store_async_read_wait();
            }
            consumer::sync(1 + consumer_id);
        }
    }

    // For TM deallocation
    everyone::tma::cluster::sync();
}

} // namespace bf16_mha_fwd

namespace bf16_mha_bwd_prep {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;
    static constexpr int NUM_THREADS = 128;
};

struct globals {
    static constexpr int BLOCK_SIZE = 128;
    static constexpr int VO_DIM = 128;

    using O_grad_tile = st_bf<BLOCK_SIZE, VO_DIM>;
    using O_tile = st_bf<BLOCK_SIZE, VO_DIM>;
    using D_vec = col_vec<st_fl<BLOCK_SIZE, VO_DIM>>;

    using O_grad_gl = gl<bf16, -1, -1, -1, -1, O_grad_tile>;
    using O_gl = gl<bf16, -1, -1, -1, -1, O_tile>;
    using D_gl = gl<float, -1, -1, -1, -1, D_vec>;

    O_grad_gl O_grad;
    O_gl O;
    D_gl D;

    __host__ inline int dynamic_shared_memory() { return sizeof(O_grad_tile) + sizeof(O_tile) + sizeof(D_vec) + 1024; }
    __host__ inline dim3 grid()  { return dim3(O_grad.rows() / BLOCK_SIZE, O_grad.depth(), O_grad.batch()); }
    __host__ inline dim3 block() { return dim3(config::NUM_THREADS); }
};

__global__ __launch_bounds__(config::NUM_THREADS, 1)
void kernel(const __grid_constant__ globals G) {
    // Declare shared memory
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);

    // Allocate shared memory
    globals::O_grad_tile &O_grad_smem = sm_allocator.allocate<globals::O_grad_tile>();
    globals::O_tile &O_smem = sm_allocator.allocate<globals::O_tile>();
    globals::D_vec &D_smem = sm_allocator.allocate<globals::D_vec>();

    // Retrieve indices
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int QO_block_idx = blockIdx.x;

    // Set up mbarriers
    __shared__ semaphore inputs_arrived;
    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        tma::expect_bytes(inputs_arrived, sizeof(globals::O_grad_tile) + sizeof(globals::O_tile));
        tma::load_async(O_grad_smem, G.O_grad, {batch_idx, head_idx, QO_block_idx, 0}, inputs_arrived);
        tma::load_async(O_smem, G.O, {batch_idx, head_idx, QO_block_idx, 0}, inputs_arrived);
    }
    __syncthreads();

    // Wait and load
    rt_fl<globals::BLOCK_SIZE / WARPGROUP_WARPS, globals::VO_DIM> O_grad_reg;
    rt_fl<globals::BLOCK_SIZE / WARPGROUP_WARPS, globals::VO_DIM> O_reg;
    wait(inputs_arrived, 0);
    warpgroup::load(O_grad_reg, O_grad_smem);
    warpgroup::load(O_reg, O_smem);

    // Compute
    col_vec<rt_fl<globals::BLOCK_SIZE / WARPGROUP_WARPS, globals::VO_DIM>> D_reg;
    warp::mul(O_grad_reg, O_grad_reg, O_reg);
    warp::row_sum(D_reg, O_grad_reg, D_reg);

    // Store to SM
    warpgroup::store(D_smem, D_reg);
    __syncthreads();

    // Store to GMEM
    if (threadIdx.x == 0) {
        tma::store_async(G.D, D_smem, {batch_idx, head_idx, 0, QO_block_idx});
    }
}

} // namespace bf16_mha_bwd_prep

namespace bf16_mha_bwd {

struct config {
    static constexpr int CLUSTER_SIZE = 1;

    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int NUM_CONSUMERS = 2;
    static constexpr int NUM_PRODUCERS = 1;
    static constexpr int NUM_WARPGROUPS = NUM_CONSUMERS + NUM_PRODUCERS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};

struct globals {
    static constexpr int BLOCK_SIZE = 64;

    static constexpr int QK_DIM = 128;
    static constexpr int VO_DIM = 128;

    static constexpr int PIPELINE_STAGES = 2;

    using Q_tile  = st_bf<BLOCK_SIZE, QK_DIM>;
    using K_tile  = st_bf<BLOCK_SIZE, QK_DIM>;
    using V_tile  = st_bf<BLOCK_SIZE, VO_DIM>;
    using O_grad_tile = st_bf<BLOCK_SIZE, VO_DIM>;
    using Q_grad_tile = st_fl<BLOCK_SIZE, QK_DIM>;
    using K_grad_tile = st_fl<BLOCK_SIZE, QK_DIM>;
    using V_grad_tile = st_fl<BLOCK_SIZE, VO_DIM>;
    using L_vec  = row_vec<st_fl<BLOCK_SIZE, BLOCK_SIZE>>;
    using D_vec  = row_vec<st_fl<BLOCK_SIZE, BLOCK_SIZE>>;
    using SP_tile = st_bf<BLOCK_SIZE, BLOCK_SIZE>; 

    using Q_gl = gl<bf16,  -1, -1, -1, -1, Q_tile>;
    using K_gl = gl<bf16,  -1, -1, -1, -1, K_tile>;
    using V_gl = gl<bf16,  -1, -1, -1, -1, V_tile>;
    using O_grad_gl = gl<bf16,  -1, -1, -1, -1, O_grad_tile>;
    using Q_grad_gl = gl<float, -1, -1, -1, -1, Q_grad_tile>;
    using K_grad_gl = gl<float, -1, -1, -1, -1, K_grad_tile>;
    using V_grad_gl = gl<float, -1, -1, -1, -1, V_grad_tile>;
    using L_gl = gl<float, -1, -1, -1, -1, L_vec>;
    using D_gl = gl<float, -1, -1, -1, -1, D_vec>; 

    Q_gl Q;
    K_gl K;
    V_gl V;
    O_grad_gl O_grad;
    Q_grad_gl Q_grad;
    K_grad_gl K_grad;
    V_grad_gl V_grad;
    L_gl L;
    D_gl D;

    __host__ inline int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
    __host__ inline dim3 grid()  { return dim3(K.rows() / (BLOCK_SIZE * 2), Q.depth(), Q.batch()); }
    __host__ inline dim3 block() { return dim3(config::NUM_THREADS); }
};

__global__ __launch_bounds__(config::NUM_THREADS, 1)
void kernel(const __grid_constant__ globals G) {
    // Set up indices
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int KV_block_idx = blockIdx.x;
    const int num_QO_blocks = G.Q.rows() / globals::BLOCK_SIZE; // Q is 64x128, KV is 128x128
    const int warpgroup_id = warpgroup::groupid();

    // Declare shared memory
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);

    // Allocate shared memory
    static_assert(
        sizeof(globals::Q_tile) * globals::PIPELINE_STAGES +
        sizeof(globals::K_tile) * config::NUM_CONSUMERS +
        sizeof(globals::V_tile) * config::NUM_CONSUMERS +
        sizeof(globals::O_grad_tile) * globals::PIPELINE_STAGES +
        sizeof(globals::Q_grad_tile) +
        sizeof(globals::L_vec) * globals::PIPELINE_STAGES +
        sizeof(globals::D_vec) * globals::PIPELINE_STAGES +
        sizeof(globals::SP_tile) * config::NUM_CONSUMERS <= config::DYNAMIC_SHARED_MEMORY
    );
    globals::Q_tile (&Q_smem)[globals::PIPELINE_STAGES] = sm_allocator.allocate<globals::Q_tile, globals::PIPELINE_STAGES>();
    globals::K_tile (&K_smem)[config::NUM_CONSUMERS] = sm_allocator.allocate<globals::K_tile, config::NUM_CONSUMERS>();
    globals::V_tile (&V_smem)[config::NUM_CONSUMERS] = sm_allocator.allocate<globals::V_tile, config::NUM_CONSUMERS>();
    globals::O_grad_tile (&O_grad_smem)[globals::PIPELINE_STAGES] = sm_allocator.allocate<globals::O_grad_tile, globals::PIPELINE_STAGES>(); 
    globals::Q_grad_tile (&Q_grad_smem) = sm_allocator.allocate<globals::Q_grad_tile>();
    globals::K_grad_tile (&K_grad_smem)[config::NUM_CONSUMERS] = *reinterpret_cast<globals::K_grad_tile(*)[config::NUM_CONSUMERS]>(&Q_smem[0].data[0]);
    globals::V_grad_tile (&V_grad_smem)[config::NUM_CONSUMERS] = *reinterpret_cast<globals::V_grad_tile(*)[config::NUM_CONSUMERS]>(&V_smem[0].data[0]);
    globals::L_vec (&L_smem)[globals::PIPELINE_STAGES] = sm_allocator.allocate<globals::L_vec, globals::PIPELINE_STAGES>();
    globals::D_vec (&D_smem)[globals::PIPELINE_STAGES] = sm_allocator.allocate<globals::D_vec, globals::PIPELINE_STAGES>(); 
    globals::SP_tile (&dS_smem)[config::NUM_CONSUMERS] = sm_allocator.allocate<globals::SP_tile, config::NUM_CONSUMERS>();

    // Allocate tensor memory
    tensor_allocator<1, config::CLUSTER_SIZE> tm_allocator {};

    // Set up mbarriers and launch initial loads
    __shared__ semaphore Q_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore KV_arrived[config::NUM_CONSUMERS];
    __shared__ semaphore LD_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore O_grad_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore Q_grad_ready;
    __shared__ semaphore matmul_finished[config::NUM_CONSUMERS];
    __shared__ semaphore compute_done[globals::PIPELINE_STAGES];
    if (threadIdx.x == 0) {
        // Initialize mbarriers
        #pragma unroll
        for (int i = 0; i < globals::PIPELINE_STAGES; i++) {
            init_semaphore(Q_arrived[i], 0, 1);
            init_semaphore(LD_arrived[i], 0, 1);
            init_semaphore(O_grad_arrived[i], 0, 1); 
            init_semaphore(compute_done[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < config::NUM_CONSUMERS; i++) {
            init_semaphore(KV_arrived[i], 0, 1);
            init_semaphore(matmul_finished[i], 0, 1);
        }
        init_semaphore(Q_grad_ready, 0, 1);

        // Load K and V
        #pragma unroll
        for (int i = 0; i < config::NUM_CONSUMERS; i++) {
            tma::expect_bytes(KV_arrived[i], sizeof(K_smem[i]) + sizeof(V_smem[i]));
            tma::load_async(K_smem[i], G.K, {batch_idx, head_idx, (KV_block_idx * config::NUM_CONSUMERS) + i, 0}, KV_arrived[i]);
            tma::load_async(V_smem[i], G.V, {batch_idx, head_idx, (KV_block_idx * config::NUM_CONSUMERS) + i, 0}, KV_arrived[i]);
        }
    }
    __syncthreads();

    // Main divergence
    if (warpgroup_id == config::NUM_CONSUMERS) {
        // Loader/storer group
        warpgroup::decrease_registers<24>();
        const int warp_id = warpgroup::warpid();
        const int lane_id = warp::laneid();

        // Declare stage and phasebits for semaphore waits
        int stage = 0;
        uint32_t phasebits = 0xFFFF0000;

        if (warp_id == 0 && lane_id == 0) { // Loader (TODO: separate L/D/Q/dO and apply per signaling)
            for (int i = 0; i < num_QO_blocks; i++) {
                wait(compute_done[stage], get_phasebit<1>(phasebits, stage));
                update_phasebit<1>(phasebits, stage);

                // Load L and D
                tma::expect_bytes(LD_arrived[stage], sizeof(L_smem[stage]) + sizeof(D_smem[stage]));
                tma::load_async(L_smem[stage], G.L, {batch_idx, head_idx, 0, i}, LD_arrived[stage]);
                tma::load_async(D_smem[stage], G.D, {batch_idx, head_idx, 0, i}, LD_arrived[stage]);

                // Load Q
                tma::expect_bytes(Q_arrived[stage], sizeof(Q_smem[stage]));
                tma::load_async(Q_smem[stage], G.Q, {batch_idx, head_idx, i, 0}, Q_arrived[stage]);

                // Load O_grad
                tma::expect_bytes(O_grad_arrived[stage], sizeof(O_grad_smem[stage]));
                tma::load_async(O_grad_smem[stage], G.O_grad, {batch_idx, head_idx, i, 0}, O_grad_arrived[stage]);

                stage = (stage + 1) % globals::PIPELINE_STAGES;
            }
        } else if (warp_id == 1 && lane_id == 0) { // Storer (TODO: is this even necessary)
            for (int i = 0; i < num_QO_blocks; i++) {
                wait(compute_done[stage], get_phasebit<0>(phasebits, stage));
                update_phasebit<0>(phasebits, stage);

                tma::store_add_async(G.Q_grad, Q_grad_smem, {batch_idx, head_idx, i, 0});
                tma::store_async_read_wait();
                arrive(Q_grad_ready);

                stage = (stage + 1) % globals::PIPELINE_STAGES;
            }
        }
    } else {
        // Consumer group
        if (warpgroup_id == 0)
            warpgroup::increase_registers<256>();
        else
            warpgroup::increase_registers<224>();

        // Constants
        constexpr float SQRT_D_INV = 0.08838834764f; // 1 / sqrt(128)
        constexpr float LOG2E = 1.44269504089f;

        // Declare stage and phasebits for semaphore waits
        int stage = 0;
        uint32_t phasebits = 0xFFFF0000;

        // Declare K and V TMEM
        auto K_grad_tm = tm_allocator.allocate<tt<float, 64, 128>>(warpgroup_id, 0);
        auto V_grad_tm = tm_allocator.allocate<tt<float, 64, 128>>(warpgroup_id, 128);

        // Wait for K & V to be loaded
        wait(KV_arrived[warpgroup_id], 0);

        for (int i = 0; i < num_QO_blocks; i++) {
            // Wait for L & D to be loaded
            wait(LD_arrived[stage], get_phasebit<0>(phasebits, stage));

            // Broadcast the L vec row (1x64) to all of the rows in the S^T tile (64x64)
            // This makes S_t = -LSE * sqrt(d)
            rt_fl<16, 64> SP_t_reg;
            #pragma unroll
            for(int ii = 0; ii < 4; ii++) {
                int base_col = 16 * ii + 2 * (warp::laneid() % 4);
                SP_t_reg.tiles[0][ii].data[0] = *(float2*)&L_smem[stage][base_col + 0];
                SP_t_reg.tiles[0][ii].data[1] = *(float2*)&L_smem[stage][base_col + 0];
                SP_t_reg.tiles[0][ii].data[2] = *(float2*)&L_smem[stage][base_col + 8];
                SP_t_reg.tiles[0][ii].data[3] = *(float2*)&L_smem[stage][base_col + 8];
            }

            // S_t = (QK^T)^T - LSE * sqrt(d)
            auto S_t_tm = tm_allocator.allocate<tt<float, 64, 64>>(warpgroup_id, 256);
            warpgroup::store_async(S_t_tm, SP_t_reg);
            tensor_store_wait();
            wait(Q_arrived[stage], get_phasebit<0>(phasebits, stage));
            warpgroup::mma_ABt(S_t_tm, K_smem[warpgroup_id], Q_smem[stage]);

            // dP_t = (dO @ V^T)^T
            auto dP_t_tm = tm_allocator.allocate<tt<float, 64, 64>>(warpgroup_id, 320);
            wait(O_grad_arrived[stage], get_phasebit<0>(phasebits, stage));
            warpgroup::mm_ABt(dP_t_tm, V_smem[warpgroup_id], O_grad_smem[stage], matmul_finished[warpgroup_id]);

            // Wait for the matmuls to finish and load
            rt_fl<16, 64> dSP_t_reg;
            wait(matmul_finished[warpgroup_id], get_phasebit<0>(phasebits, globals::PIPELINE_STAGES));
            update_phasebit<0>(phasebits, globals::PIPELINE_STAGES);
            warpgroup::load_async(SP_t_reg, S_t_tm);
            warpgroup::load_async(dSP_t_reg, dP_t_tm);

            // S_t = ( (QK^T)^T / sqrt(d) - LSE ) * log2(e)
            warp::mul(SP_t_reg, SP_t_reg, SQRT_D_INV * LOG2E);
            // P_t = S_t = exp( (QK^T)^T / sqrt(d) - LSE )
            warp::exp2(SP_t_reg, SP_t_reg);

            // Move P_t to TMEM
            auto &P_t_bf_tm = reinterpret_cast<tt<bf16, 64, 64>&>(S_t_tm);
            rt_bf<16, 64> SP_t_bf_reg;
            warp::copy(SP_t_bf_reg, SP_t_reg);
            warpgroup::store_async(P_t_bf_tm, SP_t_bf_reg);

            // dP_t = (dO @ V^T)^T - D
            #pragma unroll
            for(int ii = 0; ii < 4; ii++) {
                int base_col = 16 * ii + 2 * (warp::laneid() % 4);
                dSP_t_reg.tiles[0][ii].data[0] = base_ops::sub::template op<float2>(dSP_t_reg.tiles[0][ii].data[0], *(float2*)&D_smem[stage][base_col + 0]);
                dSP_t_reg.tiles[0][ii].data[1] = base_ops::sub::template op<float2>(dSP_t_reg.tiles[0][ii].data[1], *(float2*)&D_smem[stage][base_col + 0]);
                dSP_t_reg.tiles[0][ii].data[2] = base_ops::sub::template op<float2>(dSP_t_reg.tiles[0][ii].data[2], *(float2*)&D_smem[stage][base_col + 8]);
                dSP_t_reg.tiles[0][ii].data[3] = base_ops::sub::template op<float2>(dSP_t_reg.tiles[0][ii].data[3], *(float2*)&D_smem[stage][base_col + 8]);
            }
            // dS_t = P_t * ( (dO @ V^T)^T - D )
            warp::mul(dSP_t_reg, SP_t_reg, dSP_t_reg);
            // dS_t = P_t * ( (dO @ V^T)^T - D ) / sqrt(d)
            warp::mul(dSP_t_reg, dSP_t_reg, SQRT_D_INV);

            // dV = P^T @ O_grad
            tensor_store_wait();
            warpgroup::sync(warpgroup_id + 2);
            if (i == 0)
                warpgroup::mm_AB(V_grad_tm, P_t_bf_tm, O_grad_smem[stage]);
            else
                warpgroup::mma_AB(V_grad_tm, P_t_bf_tm, O_grad_smem[stage]);

            // dK = dS^T @ Q
            auto &dS_t_bf_tm = reinterpret_cast<tt<bf16, 64, 64>&>(dP_t_tm);
            rt_bf<16, 64> dSP_t_bf_reg;
            warp::copy(dSP_t_bf_reg, dSP_t_reg);
            warpgroup::store_async(dS_t_bf_tm, dSP_t_bf_reg);
            tensor_store_wait();
            warpgroup::sync(warpgroup_id + 2);
            if (i == 0)
                warpgroup::mm_AB(K_grad_tm, dS_t_bf_tm, Q_smem[stage], matmul_finished[warpgroup_id]);
            else
                warpgroup::mma_AB(K_grad_tm, dS_t_bf_tm, Q_smem[stage], matmul_finished[warpgroup_id]);

            warpgroup::store(dS_smem[warpgroup_id], dSP_t_reg);
            wait(matmul_finished[warpgroup_id], get_phasebit<0>(phasebits, globals::PIPELINE_STAGES));
            update_phasebit<0>(phasebits, globals::PIPELINE_STAGES);
            group<config::NUM_CONSUMERS * WARPGROUP_WARPS>::sync(1);

            if (warpgroup_id == 0) {
                // dQ = dS_t @ K
                auto Q_grad_tm = tm_allocator.allocate<tt<float, 64, 128>>(0, 384); // just used by warpgroup 0
                warpgroup::mm_AtB(Q_grad_tm, dS_smem[0], K_smem[0]);
                warpgroup::mma_AtB(Q_grad_tm, dS_smem[1], K_smem[1], matmul_finished[0]);

                // Wait for matrix multiply to complete and SMEM to be empty
                wait(matmul_finished[0], get_phasebit<0>(phasebits, globals::PIPELINE_STAGES));
                update_phasebit<0>(phasebits, globals::PIPELINE_STAGES);
                wait(Q_grad_ready, get_phasebit<1>(phasebits, globals::PIPELINE_STAGES));
                update_phasebit<1>(phasebits, globals::PIPELINE_STAGES);

                // Store Q_grad and signal
                rt_fl<16, 128> Q_grad_reg;
                warpgroup::load_async(Q_grad_reg, Q_grad_tm);
                warpgroup::store(Q_grad_smem, Q_grad_reg);
                warpgroup::sync(warpgroup_id + 2);
                warpgroup::arrive(compute_done[stage]);
            }

            update_phasebit<0>(phasebits, stage);
            stage = (stage + 1) % globals::PIPELINE_STAGES;
        }

        // Load finished K_grad and V_grad
        rt_fl<16, 128> K_grad_reg;
        rt_fl<16, 128> V_grad_reg;
        warpgroup::load_async(K_grad_reg, K_grad_tm);
        warpgroup::load_async(V_grad_reg, V_grad_tm);

        // Store V_grad
        warpgroup::store(V_grad_smem[warpgroup_id], V_grad_reg);
        warpgroup::sync(2 + warpgroup_id);
        warpgroup::tma::store_add_async(G.V_grad, V_grad_smem[warpgroup_id], {batch_idx, head_idx, (KV_block_idx * config::NUM_CONSUMERS) + warpgroup_id, 0});

        // Store K_grad
        wait(Q_grad_ready, get_phasebit<1>(phasebits, globals::PIPELINE_STAGES));
        update_phasebit<1>(phasebits, globals::PIPELINE_STAGES); // TODO: remove
        group<config::NUM_CONSUMERS * WARPGROUP_WARPS>::sync(1);
        warpgroup::store(K_grad_smem[warpgroup_id], K_grad_reg);
        warpgroup::sync(2 + warpgroup_id);
        warpgroup::tma::store_add_async(G.K_grad, K_grad_smem[warpgroup_id], {batch_idx, head_idx, (KV_block_idx * config::NUM_CONSUMERS) + warpgroup_id, 0});
    }
}

} // namespace bf16_mha_bwd

constexpr int NUM_CONSUMERS = (2); 
constexpr int WARPGROUPS_PER_CONSUMER = (2);
constexpr int NUM_PRODUCERS = (1);

constexpr int NUM_WARPS = (NUM_CONSUMERS*WARPGROUPS_PER_CONSUMER + NUM_PRODUCERS) * 4;
constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;

/**
 * @brief Makes a square register tile anti-causal by zeroing elements below the main diagonal.
 *
 * This function modifies a square register tile in-place to make it anti-causal. All elements
 * below the main diagonal are set to zero, while elements on or above the main diagonal
 * are left unchanged.
 *
 * @tparam T The data type of the register tile elements.
 * @tparam _size The size (height and width) of the square register tile.
 * @tparam layout The current layout of the register tile.
 * @param tile[in,out] Reference to the register tile to be made causal.
 */
template<ducks::rt::row_layout RT>
__device__ static inline void make_causal_t(RT &dst, const RT &src, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
    #ifdef KITTENS_HOPPER
    static_assert(!std::is_same_v<typename RT::dtype, fp8e4m3_4> && !std::is_same_v<typename RT::dtype, fp8e5m2_4>, "Unsupported type for make_causal");
    #endif
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            if(j > i) { // above the diagonal, copy
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = src.tiles[i][j].data[k];
                }
            }
            else if(j < i) { // below the diagonal, zero
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = packed_val;
                }
            }
            else { // on the diagonal, interesting!
                constexpr uint32_t MASK_X = 0x88CCEEF; 
                constexpr uint32_t MASK_Y = 0x88CCEEFF;

                dst.tiles[i][j].data[1] = packed_val;              // below diagonal, zero
                dst.tiles[i][j].data[2] = src.tiles[i][j].data[2]; // above diagonal, copy

                // on the diagonal or above
                if((MASK_X >> laneid()) & 1) {
                    dst.tiles[i][j].data[0].x = src.tiles[i][j].data[0].x;
                    dst.tiles[i][j].data[3].x = src.tiles[i][j].data[3].x;
                }
                // below the diagonal
                else {
                    dst.tiles[i][j].data[0].x = val;
                    dst.tiles[i][j].data[3].x = val;
                }

                // on the diagonal or above
                if((MASK_Y >> laneid()) & 1) {
                    dst.tiles[i][j].data[0].y = src.tiles[i][j].data[0].y;
                    dst.tiles[i][j].data[3].y = src.tiles[i][j].data[3].y;
                }
                // below the diagonal
                else {
                    dst.tiles[i][j].data[0].y = val;
                    dst.tiles[i][j].data[3].y = val;
                }
                
            }
            __syncwarp();
        }
    }
}

struct rescale_add {
    template<typename T> static __device__ inline T op(const T &a, const T &b) {
        if constexpr (std::is_same_v<T, float2>) {
            constexpr float2 scale = {1.44269504089f*0.08838834764f, 1.44269504089f*0.08838834764f};
            float2 c;
            asm volatile("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(*(uint64_t*)&c) : "l"(*(uint64_t*)&a), "l"(*(uint64_t*)&scale), "l"(*(uint64_t*)&b));
            return c;
        }
        else {
            static_assert(sizeof(T) == 999, "Currently unsupported type");
        }
    }
};
template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void rescale_add_row(T &dst, const T &src, const V &row_values) {
    warp::row_map<rescale_add, T, V>(dst, src, row_values);
}

template<int D> struct fwd_attend_ker_tile_dims {};
template<> struct fwd_attend_ker_tile_dims<128> {
    constexpr static int tile_width = (128);
    constexpr static int qo_height  = (128);
    constexpr static int kv_height  = (128);
    constexpr static int stages     = (2); 
};

template<int D=128> struct fwd_globals {
    using q_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::qo_height,   fwd_attend_ker_tile_dims<D>::tile_width>  ;
    using k_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::kv_height/2, fwd_attend_ker_tile_dims<D>::tile_width>  ; // since we're using a two-SM dispatch, split on N dim.
    using v_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::kv_height,   fwd_attend_ker_tile_dims<D>::tile_width/2>; // since we're using a two-SM dispatch, split on N dim.
    using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<D>::qo_height,   fwd_attend_ker_tile_dims<D>::tile_width>> ;
    using o_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::qo_height,   fwd_attend_ker_tile_dims<D>::tile_width>  ;

    using q_gl = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16,  -1, -1, -1, -1, v_tile>;
    using l_gl = gl<float, -1, -1, -1, -1, l_col_vec>;
    using o_gl = gl<bf16,  -1, -1, -1, -1, o_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    l_gl l;
    o_gl o;

    int dynamic_shared_memory() { return 226000; }
    dim3 grid()  { return dim3(148); }
    dim3 block() { return dim3(NUM_THREADS); }
};

template<int D=128> struct softmax_registers {
    static constexpr int rows = fwd_attend_ker_tile_dims<D>::qo_height/(WARPGROUPS_PER_CONSUMER*WARPGROUP_WARPS), cols = fwd_attend_ker_tile_dims<D>::kv_height;
    rt_fl<rows, cols> att_block;
    rt_bf<rows, cols> att_block_mma;
    col_vec<rt_fl<rows, cols>> max_vec, norm_vec, max_vec_last_scaled, max_vec_scaled;
    __device__ inline void init() {
        warp::neg_infty(max_vec);
        warp::zero(norm_vec);
    }
};
template<int D> __device__ static inline void softmax(softmax_registers<D> &regs) {
    regs.max_vec_last_scaled = regs.max_vec * 1.44269504089f*0.08838834764f;
    warp::max<axis::COL>(regs.max_vec, regs.att_block, regs.max_vec);
    regs.max_vec_scaled = regs.max_vec * -1.44269504089f*0.08838834764f;
    rescale_add_row(regs.att_block, regs.att_block, regs.max_vec_scaled);
    regs.att_block = warp::exp2(regs.att_block);
    regs.max_vec_last_scaled += regs.max_vec_scaled;
    regs.max_vec_last_scaled = warp::exp2(regs.max_vec_last_scaled);
    regs.norm_vec *= regs.max_vec_last_scaled;
    warp::sum<axis::COL>(regs.norm_vec, regs.att_block, regs.norm_vec);
    warp::copy(regs.att_block_mma, regs.att_block);
}

__device__ static inline int get_iters_per_task(const fwd_globals<> &g) {
    return g.k.rows() / fwd_globals<>::v_tile::rows;
}
__device__ static inline int3 get_task_idx(const fwd_globals<> &g, int task_iter) {
    int cluster_x = clusterIdx().x, ctarank = cluster_ctarank();
    int task_id = task_iter * (gridDim.x/2) + cluster_x;
    constexpr int q_rows_per_task = 2 * NUM_CONSUMERS*fwd_globals<>::q_tile::rows;
    int seq_q = (g.q.rows() + q_rows_per_task - 1)/(q_rows_per_task);
    int3 task_idx;
    task_idx.x = task_id / (seq_q*g.k.depth());
    task_id -= task_idx.x * seq_q * g.k.depth();
    task_idx.y  = task_id / seq_q;
    task_id -= task_idx.y  * seq_q;
    task_idx.z   = task_id;
    if(task_idx.x >= g.q.batch()) return { -1, -1, -1 };
    return task_idx;
}

template<int D=128, bool causal=false> __global__ __cluster_dims__(2) __launch_bounds__(NUM_THREADS, 1)
void fwd_attend_ker(const __grid_constant__ fwd_globals<D> g) {
    static_assert(!causal, "Causal attention not supported yet");
    static_assert(D==128, "Only D=128 is supported");

    using K = fwd_attend_ker_tile_dims<D>;
    using G = fwd_globals<D>;
    using consumer = group<WARPGROUP_WARPS * WARPGROUPS_PER_CONSUMER>;

    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpgroupid = warpgroup::groupid();
    int consumerid = warpgroupid == NUM_CONSUMERS * WARPGROUPS_PER_CONSUMER ? 0 : consumer::groupid();
    int ctarank = cluster_ctarank();
    int iters_per_task = get_iters_per_task(g);

    using q_tile    = G::q_tile;
    using k_tile    = G::k_tile;
    using v_tile    = G::v_tile;
    using l_col_vec = G::l_col_vec;
    using o_tile    = G::o_tile;
    
    q_tile    (&q_smem)[NUM_CONSUMERS] = al.allocate<q_tile, NUM_CONSUMERS>();
    k_tile    (&k_smem)[K::stages]     = al.allocate<k_tile, K::stages>();
    v_tile    (&v_smem)[K::stages]     = al.allocate<v_tile, K::stages>();
    l_col_vec (&l_smem)[NUM_CONSUMERS] = al.allocate<l_col_vec, NUM_CONSUMERS>();
    auto      (*o_smem)                = reinterpret_cast<o_tile(*)>(&q_smem);
    st_bf<128,128> (&att_smem)[NUM_CONSUMERS] = al.allocate<st_bf<128,128>, NUM_CONSUMERS>();

    tensor_allocator<1, 2> tm_alloc{};
    using att_tm_fl_t = tt<float, K::qo_height, K::kv_height>;
    using att_tm_bf_t = tt<bf16,  K::qo_height, K::kv_height>;
    using o_tm_fl_t   = tt<float, K::qo_height, K::tile_width>;
    att_tm_fl_t att_tm    = tm_alloc.allocate<att_tm_fl_t>(consumerid*K::kv_height);
    o_tm_fl_t   o_tm      = tm_alloc.allocate<o_tm_fl_t>  ((NUM_CONSUMERS*K::kv_height) + consumerid*K::tile_width);
    att_tm_bf_t att_bf_tm = reinterpret_cast<att_tm_bf_t&>(att_tm);

    __shared__ kittens::semaphore q_smem_arrived[NUM_CONSUMERS],
                                  k_smem_arrived[K::stages], v_smem_arrived[K::stages],
                                  k_smem_finished[K::stages], v_smem_finished[K::stages],
                                  attn_unloaded[NUM_CONSUMERS],
                                  attn_mma_stored[NUM_CONSUMERS],
                                  qk_matmul_done[NUM_CONSUMERS], av_matmul_done[NUM_CONSUMERS];
    uint32_t bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s

    if (threadIdx.x == 0) {
        for(int i = 0; i < K::stages; i++) {
            init_semaphore(k_smem_arrived[i], 0, 2);
            init_semaphore(v_smem_arrived[i], 0, 2);
            init_semaphore(k_smem_finished[i], 0, NUM_CONSUMERS); 
            init_semaphore(v_smem_finished[i], 0, NUM_CONSUMERS); 
        }
        for(int i = 0; i < NUM_CONSUMERS; i++) {
            init_semaphore(q_smem_arrived[i], 0, 2); 
            init_semaphore(attn_unloaded[i], 0, 2*8);
            init_semaphore(attn_mma_stored[i], 0, 2*8);
            init_semaphore(qk_matmul_done[i], 0, 1);
            init_semaphore(av_matmul_done[i], 0, 1);
        }
    }

    everyone::tma::cluster::sync();
    
    if(warpgroupid == NUM_CONSUMERS*WARPGROUPS_PER_CONSUMER) {
        warpgroup::decrease_registers<64>();
        if(ctarank == 0 && warpgroup::warpid() == 0 && warp::laneid() == 0) { // launch the QK MMA's
            int input_ring = 0; // tracking which input block is being loaded
            for(int task_iter = 0; true; task_iter++) {
                int3 batchheadrow = get_task_idx(g, task_iter);
                if(batchheadrow.x == -1) break;
                for(int idx = 0; idx < iters_per_task; idx++) {
                    tma::cluster::wait(k_smem_arrived[input_ring], prototype::get_phasebit<0>(bitfield, input_ring));
                    #pragma unroll
                    for(int i = 0; i < NUM_CONSUMERS; i++) {
                        if(idx == 0) tma::cluster::wait(q_smem_arrived[i], task_iter%2);    // make sure Q is loaded
                        tma::cluster::wait(attn_unloaded[i], prototype::get_phasebit<1>(bitfield, i)); // make sure ready to launch the next one.
                        auto att_tm_i = att_tm.template subtile<att_tm_fl_t>(0, i*K::kv_height);
                        mm2_ABt(att_tm_i, q_smem[i], k_smem[input_ring], qk_matmul_done[i]);
                        prototype::update_phasebit<1>(bitfield, i);
                    }
                    prototype::update_phasebit<0>(bitfield, input_ring);
                    input_ring=prototype::ring_advance<K::stages>(input_ring);
                }
            }
        }
        else if(ctarank == 0 && warpgroup::warpid() == 1 && warp::laneid() == 0) { // launch the AV MMA's
            int input_ring = 0; // tracking which input block is being loaded
            for(int task_iter = 0; true; task_iter++) {
                int3 batchheadrow = get_task_idx(g, task_iter);
                if(batchheadrow.x == -1) break;
                for(int idx = 0; idx < iters_per_task; idx++) {
                    tma::cluster::wait(v_smem_arrived[input_ring], prototype::get_phasebit<0>(bitfield, input_ring));
                    #pragma unroll
                    for(int i = 0; i < NUM_CONSUMERS; i++) {
                        tma::cluster::wait(attn_mma_stored[i], prototype::get_phasebit<0>(bitfield, K::stages+i));
                        auto o_tm_i = o_tm.template subtile<o_tm_fl_t>(0, i*K::tile_width);
                        mma2_AB(o_tm_i, att_smem[i], v_smem[input_ring], av_matmul_done[i]);
                        prototype::update_phasebit<0>(bitfield, K::stages+i);
                    }
                    prototype::update_phasebit<0>(bitfield, input_ring);
                    input_ring=prototype::ring_advance<K::stages>(input_ring);
                }
            }
        }
        else if(warpgroup::warpid() == 2 && warp::laneid() == 0) { // K loader
            int input_ring = 0; // tracking which input block is being loaded
            for(int task_iter = 0; true; task_iter++) {
                int3 batchheadrow = get_task_idx(g, task_iter);
                if(batchheadrow.x == -1) break;
                for (int idx = 0; idx < iters_per_task; idx++) {
                    kittens::wait(k_smem_finished[input_ring], prototype::get_phasebit<1>(bitfield, input_ring));
                    tma::cluster::expect(k_smem_arrived[input_ring], 0, k_smem[input_ring]);
                    tma::cluster::load_async(k_smem[input_ring], g.k, {batchheadrow.x, batchheadrow.y, 2*idx + ctarank, 0},
                                            k_smem_arrived[input_ring], (uint16_t)(1<<ctarank), 0);
                    prototype::update_phasebit<1>(bitfield, input_ring);
                    input_ring=prototype::ring_advance<K::stages>(input_ring);
                }
            }
        }
        else if(warpgroup::warpid() == 3 && warp::laneid() == 0) { // V loader
            int input_ring = 0; // tracking which input block is being loaded
            for(int task_iter = 0; true; task_iter++) {
                int3 batchheadrow = get_task_idx(g, task_iter);
                if(batchheadrow.x == -1) break;
                for (int idx = 0; idx < iters_per_task; idx++) {
                    kittens::wait(v_smem_finished[input_ring], prototype::get_phasebit<1>(bitfield, input_ring));
                    tma::cluster::expect(v_smem_arrived[input_ring], 0, v_smem[input_ring]);
                    tma::cluster::load_async(v_smem[input_ring], g.v, {batchheadrow.x, batchheadrow.y, idx, ctarank},
                                             v_smem_arrived[input_ring], (uint16_t)(1<<ctarank), 0);
                    prototype::update_phasebit<1>(bitfield, input_ring);
                    input_ring=prototype::ring_advance<K::stages>(input_ring);
                }
            }
        }
    }
    else {
        warpgroup::increase_registers<104>();
        using all_consumers = group<NUM_CONSUMERS*WARPGROUPS_PER_CONSUMER*WARPGROUP_WARPS>;
        using all_barrier = kittens::barrier<all_consumers::GROUP_WARPS>;
        all_barrier bar(15);
        
        softmax_registers<D> sr;
        rt_fl<K::qo_height/(WARPGROUPS_PER_CONSUMER*WARPGROUP_WARPS), K::tile_width> o_reg;
        int k_input_ring = 0, v_input_ring = 0;

        for(int task_iter = 0; true; task_iter++) {
            int3 batchheadrow = get_task_idx(g, task_iter);
            if(batchheadrow.x == -1) break;
            // Load Q matrices
            if(consumer::laneid() == 0) {
                tma::cluster::expect(q_smem_arrived[consumerid], 0, q_smem[consumerid]);
                tma::cluster::load_async(q_smem[consumerid], g.q,
                    {batchheadrow.x, batchheadrow.y, 2*NUM_CONSUMERS*batchheadrow.z + NUM_CONSUMERS*ctarank + consumerid, 0}, q_smem_arrived[consumerid], (uint16_t)(1<<ctarank), 0);
            }

            // Initialize register state
            sr.init();
            warp::zero(o_reg);

            for(int idx = 0; idx < iters_per_task; idx++) {

                // wait for QK matmul
                tma::cluster::wait(qk_matmul_done[consumerid], prototype::get_phasebit<0>(bitfield, K::stages+0));
                prototype::update_phasebit<0>(bitfield, K::stages+0);
                if(consumer::laneid() == 0) arrive(k_smem_finished[k_input_ring]);
                k_input_ring=prototype::ring_advance<K::stages>(k_input_ring); // Advance the ring to the next input block

                consumer::load_async(sr.att_block, att_tm);
                tensor_load_wait();
                if(laneid() == 0) tma::cluster::arrive(attn_unloaded[consumerid], 0); // signal that we're ready to launch the next QK matmul for this consumer.
                
                // Do softmax and o register rescaling.
                softmax(sr);

                // Do the O rescaling, store the attention matrix, and signal to launch the next AV matmul.
                if(idx>0) { // Don't wait or signal on the first iteration.
                    tma::cluster::wait(av_matmul_done[consumerid], prototype::get_phasebit<0>(bitfield, K::stages+1)); // O must be ready for us to use.
                    if(consumer::laneid() == 0) arrive(v_smem_finished[v_input_ring]); // since that matmul finished, we can next the load the next block.
                    prototype::update_phasebit<0>(bitfield, K::stages+1);
                    v_input_ring=prototype::ring_advance<K::stages>(v_input_ring); // Advance the ring to the next input block
                }
                consumer::load_async(o_reg, o_tm);
                consumer::store(att_smem[consumerid], sr.att_block_mma);
                warp::mul_row(o_reg, o_reg, sr.max_vec_last_scaled);
                consumer::store_async(o_tm, o_reg);
                tensor_store_wait();
                if(laneid() == 0) tma::cluster::arrive(attn_mma_stored[consumerid], 0);
                consumer::sync(consumerid);
            }

            // Wait for the last AV matmul to finish and store the output.
            tma::cluster::wait(av_matmul_done[consumerid], prototype::get_phasebit<0>(bitfield, K::stages+1)); // O must be ready for us to use.
            if(consumer::laneid() == 0) arrive(v_smem_finished[v_input_ring]); // since that matmul finished, we can next the load the next block.
            prototype::update_phasebit<0>(bitfield, K::stages+1);
            v_input_ring=prototype::ring_advance<K::stages>(v_input_ring); // Advance the ring to the next input block

            consumer::load_async(o_reg, o_tm);
            warp::div_row(o_reg, o_reg, sr.norm_vec);
            consumer::store(o_smem[consumerid], o_reg);
            consumer::sync(consumerid);
            if(consumer::laneid() == 0) {
                tma::store_async(g.o, o_smem[consumerid], {batchheadrow.x, batchheadrow.y, 2*NUM_CONSUMERS*batchheadrow.z + NUM_CONSUMERS*ctarank + consumerid, 0});
            }

            warp::mul(sr.max_vec_scaled, sr.max_vec_scaled, -0.69314718056f);
            warp::log(sr.norm_vec, sr.norm_vec);
            warp::add(sr.norm_vec, sr.norm_vec, sr.max_vec_scaled);

            if constexpr (D == 64) { warp::mul(sr.norm_vec, sr.norm_vec, -8.0f); }
            else                   { warp::mul(sr.norm_vec, sr.norm_vec, -11.313708499f); }

            consumer::store(l_smem[consumerid], sr.norm_vec);
            consumer::sync(consumerid);

            if(consumer::laneid() == 0) {
                tma::store_async(g.l, l_smem[consumerid], {batchheadrow.x, batchheadrow.y, 0, 2*NUM_CONSUMERS*batchheadrow.z + NUM_CONSUMERS*ctarank + consumerid});
            }
            
            tma::store_async_read_wait();
            consumer::sync(consumerid);
        }
    }

    everyone::tma::cluster::sync();
}

// ---------------------------------------------------------------------------------------------------
// ----------------------------------- Backward preparation kernel -----------------------------------
// ---------------------------------------------------------------------------------------------------

template<int D>
struct bwd_prep_globals {
    using og_tile = st_bf<4*16, D>;
    using o_tile  = st_bf<4*16, D>;
    using d_tile  = col_vec<st_fl<4*16, D>>;

    using og_gl = gl<bf16,  -1, -1, -1, -1, og_tile>;
    using o_gl  = gl<bf16,  -1, -1, -1, -1, o_tile>;
    using d_gl  = gl<float, -1, -1, -1, -1, d_tile>;

    og_gl og;
    o_gl  o;
    d_gl  d;

    int dynamic_shared_memory() { return 226000 / ((D == 64) ? 2 : 1); }
    dim3 grid()  { return dim3(og.rows()/256, og.depth(), og.batch()); }
    dim3 block() { return dim3(128); }
};

template<int D>
__global__  __launch_bounds__(4*kittens::WARP_THREADS, (D == 64) ? 2 : 1)
void bwd_attend_prep_ker(const __grid_constant__ bwd_prep_globals<D> g) {
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);

    int warpid = kittens::warpid();

    using og_tile = st_bf<4*16, D>;
    using o_tile  = st_bf<4*16, D>;
    using d_tile  = col_vec<st_fl<4*16, D>>;

    og_tile (&og_smem)[4] = al.allocate<og_tile, 4>();
    o_tile  (&o_smem) [4] = al.allocate<o_tile , 4>();
    d_tile  (&d_smem) [4] = al.allocate<d_tile , 4>();
    
    rt_fl<4*16, D> og_reg, o_reg; 
    col_vec<rt_fl<4*16, D>> d_reg;

    __shared__ kittens::semaphore smem_semaphore;

    if (threadIdx.x == 0) {
        init_semaphore(smem_semaphore, 0, 1);
        tma::expect_bytes(smem_semaphore, sizeof(og_smem[0]) * 4 * 2);
        #pragma unroll
        for (int w = 0; w < 4; w++) {
            coord<o_tile> tile_idx = {blockIdx.z, blockIdx.y, (blockIdx.x * 4) + w, 0};
            tma::load_async(o_smem[w],  g.o,  tile_idx, smem_semaphore);
            tma::load_async(og_smem[w], g.og, tile_idx, smem_semaphore);
        }
    }
    __syncthreads();

    wait(smem_semaphore, 0);
    warp::load(o_reg, o_smem[warpid]);
    warp::load(og_reg, og_smem[warpid]);
    warp::mul(og_reg, og_reg, o_reg);
    warp::sum<axis::COL>(d_reg, og_reg);
    warp::store(d_smem[warpid], d_reg);
    __syncthreads();

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int w = 0; w < 4; w++) {
            coord<d_tile> tile_idx = {blockIdx.z, blockIdx.y, 0, (blockIdx.x * 4) + w};
            tma::store_async(g.d, d_smem[w], tile_idx);
        }
        tma::store_async_wait();
    }
}

template<int D> struct bwd_attend_ker_tile_dims {};
template<> struct bwd_attend_ker_tile_dims<64> {
    constexpr static int tile_width = (64);
    constexpr static int tile_h     = (4*16);
    constexpr static int tile_h_qo  = (4*16);
    constexpr static int blocks_sm = 1;
};
template<> struct bwd_attend_ker_tile_dims<128> {
    constexpr static int tile_width = (128);
    constexpr static int tile_h     = (4*16);
    constexpr static int tile_h_qo  = (4*16);
    constexpr static int blocks_sm = 1; 
};

constexpr int BWD_CONSUMER_WARPGROUPS = (2); 
constexpr int BWD_PRODUCER_WARPGROUPS = (1); 
constexpr int BWD_NUM_WARPGROUPS      = (BWD_CONSUMER_WARPGROUPS+BWD_PRODUCER_WARPGROUPS); 
constexpr int BWD_NUM_WORKERS         = (BWD_NUM_WARPGROUPS*kittens::WARPGROUP_WARPS); 

template<int D>
struct bwd_globals {
    using G = bwd_attend_ker_tile_dims<D>;

    using q_tile  =         st_bf<G::tile_h_qo, G::tile_width>;
    using k_tile  =         st_bf<G::tile_h,    G::tile_width>;
    using v_tile  =         st_bf<G::tile_h,    G::tile_width>;
    using og_tile =         st_bf<G::tile_h_qo, G::tile_width>;
    using qg_tile =         st_fl<G::tile_h_qo, G::tile_width>;
    using kg_tile =         st_fl<G::tile_h,    G::tile_width>;
    using vg_tile =         st_fl<G::tile_h,    G::tile_width>;
    using l_tile  = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using d_tile  = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;

    using q_gl  = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_gl  = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_gl  = gl<bf16,  -1, -1, -1, -1, v_tile>;

    using og_gl = gl<bf16,  -1, -1, -1, -1, og_tile>;

    using qg_gl = gl<float, -1, -1, -1, -1, qg_tile>;
    using kg_gl = gl<float, -1, -1, -1, -1, kg_tile>;
    using vg_gl = gl<float, -1, -1, -1, -1, vg_tile>;

    using l_gl  = gl<float, -1, -1, -1, -1, l_tile>;
    using d_gl  = gl<float, -1, -1, -1, -1, d_tile>; 

    q_gl  q;
    k_gl  k;
    v_gl  v;
    og_gl og;
    qg_gl qg;
    kg_gl kg;
    vg_gl vg;
    l_gl  l;
    d_gl  d;

    const int N;
    const int hr;

    int dynamic_shared_memory() { return 226000; }
    dim3 grid()  { return dim3(q.rows()/128, q.depth(), q.batch()); }
    dim3 block() { return dim3(BWD_NUM_WORKERS*32); }
};

__device__ static inline void
stream_tile(auto &reg_tile, auto &smem_vec, int tic) {
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        int base_col = 16*i + 2*(kittens::laneid()%4);
        reg_tile.tiles[0][i].data[0] = *(float2*)&smem_vec[tic][base_col + 0];
        reg_tile.tiles[0][i].data[1] = *(float2*)&smem_vec[tic][base_col + 0];
        reg_tile.tiles[0][i].data[2] = *(float2*)&smem_vec[tic][base_col + 8];
        reg_tile.tiles[0][i].data[3] = *(float2*)&smem_vec[tic][base_col + 8];
    }
}

__device__ static inline void
stream_sub_tile(auto &reg_tile, auto &smem_vec, int tic) {
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        int base_col = 16*i + 2*(laneid()%4);
        reg_tile.tiles[0][i].data[0] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[0], *(float2*)&smem_vec[tic][base_col + 0]);
        reg_tile.tiles[0][i].data[1] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[1], *(float2*)&smem_vec[tic][base_col + 0]);
        reg_tile.tiles[0][i].data[2] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[2], *(float2*)&smem_vec[tic][base_col + 8]);
        reg_tile.tiles[0][i].data[3] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[3], *(float2*)&smem_vec[tic][base_col + 8]);
    }
}

template<int tile_h_qo, int tile_h>
__device__ static inline void 
causal_mask(auto &reg_tile, int qo_idx) {
    int q_blk = (qo_idx) * (tile_h_qo/kittens::TILE_ROW_DIM<bf16>);
    int k_blk = (blockIdx.x * BWD_CONSUMER_WARPGROUPS * (tile_h/kittens::TILE_ROW_DIM<bf16>)) 
                + ((kittens::warpid()/kittens::WARPGROUP_WARPS) * (tile_h/kittens::TILE_ROW_DIM<bf16>)) 
                + (kittens::warpid() % kittens::WARPGROUP_WARPS);

    for (int j = 0; j < (tile_h_qo/kittens::TILE_ROW_DIM<bf16>); j++) {
        int q_idx = q_blk + j;
        auto &attn_subtile = reinterpret_cast<rt_fl<16, 16>&>(reg_tile.tiles[0][j]);
        if      (q_idx  < k_blk) { warp::neg_infty(attn_subtile); }
        else if (q_idx == k_blk) { make_causal_t(attn_subtile, attn_subtile, kittens::base_types::constants<float>::neg_infty()); }
    }
}

// The tensor memory used by a warpgroup.
struct wg_tmem_t {
    tt<float, 64, 128> &kg;
    tt<float, 64, 128> &vg;
    tt<float, 64, 64>  &sb;
    tt<float, 64, 64>  &dp;
    tt<bf16,  64, 64>  &pb_bf;
    tt<bf16,  64, 64>  &dp_bf;
    semaphore *mma_sem;
};

template<bool is_causal, int tile_h_qo, int tile_h, int tile_width, int D>
__device__ static inline void
compute_bwd_loop(
        wg_tmem_t &wg_tmem,
        kittens::semaphore *vec_b, kittens::semaphore *q_b, kittens::semaphore *o_b, 
        rt_fl<16, 64> &s_block_t, rt_fl<16, 64> &dp_block_t, 
        rt_fl<16, 64> &p_block_t, rt_fl<16, 64> &ds_block_t,  
        rt_bf<16, 64> &p_block_t_mma,  rt_bf<16, 64> &ds_block_t_mma,
        rt_fl<16, tile_width> &kg_reg, rt_fl<16, tile_width> &vg_reg,
        auto &q_smem, auto &k_smem, auto &v_smem, 
        auto &og_smem, auto &ds_smem, auto &l_smem, auto &d_smem,
        int qo_idx, int q_start, int tic, int toc) 
{
    wait(vec_b[tic], ((qo_idx - q_start)/2)%2);
    stream_tile(s_block_t, l_smem, tic);
    warpgroup::store_async(wg_tmem.sb, s_block_t);
    wait(q_b[tic], ((qo_idx - q_start)/2)%2);

    // warpgroup::mma_ABt(s_block_t, k_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], q_smem[tic]);
    tensor_store_wait();
    warpgroup::sync(warpgroup::groupid()+4);
    if(warpgroup::laneid() == 0) {
        mma_ABt(wg_tmem.sb, k_smem[warpgroup::groupid()], q_smem[tic], *wg_tmem.mma_sem);
    }

    wait(o_b[tic], ((qo_idx - q_start)/2)%2);
    // warpgroup::mm_ABt(dp_block_t, v_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], og_smem[tic]);
    if(warpgroup::laneid() == 1) {
        mm_ABt(wg_tmem.dp, v_smem[warpgroup::groupid()], og_smem[tic], *wg_tmem.mma_sem);
    }
    // warpgroup::mma_async_wait();
    wait(*wg_tmem.mma_sem, 0);
    warpgroup::load_async(s_block_t, wg_tmem.sb);
    warpgroup::load_async(dp_block_t, wg_tmem.dp);

    if constexpr (D == 64) { warp::mul(s_block_t, s_block_t, 1.44269504089f*0.125f); }
    else                   { warp::mul(s_block_t, s_block_t, 1.44269504089f*0.08838834764f); }

    if constexpr (is_causal) { causal_mask<tile_h_qo, tile_h>(s_block_t, qo_idx); }

    warp::exp2(s_block_t, s_block_t);
    warp::copy(p_block_t, s_block_t);
    warp::copy(p_block_t_mma, s_block_t);
    warpgroup::store_async(wg_tmem.pb_bf, p_block_t_mma);
    stream_sub_tile(dp_block_t, d_smem, tic);
    warp::mul(ds_block_t, p_block_t, dp_block_t);

    if constexpr (D == 64) { warp::mul(ds_block_t, ds_block_t, 0.125f); }
    else                   { warp::mul(ds_block_t, ds_block_t, 0.08838834764f); }

    // warpgroup::mma_AB(vg_reg, p_block_t_mma, og_smem[tic]);
    tensor_store_wait();
    warpgroup::sync(warpgroup::groupid()+4);
    if(warpgroup::laneid() == 0) {
        mma_AB(wg_tmem.vg, wg_tmem.pb_bf, og_smem[tic], *wg_tmem.mma_sem);
    }
    
    warp::copy(ds_block_t_mma, ds_block_t);
    warpgroup::store_async(wg_tmem.dp_bf, ds_block_t_mma);
    tensor_store_wait();
    warpgroup::sync(warpgroup::groupid()+4);
    warpgroup::store(ds_smem[warpgroup::groupid()], ds_block_t);
    // warpgroup::mma_AB(kg_reg, ds_block_t_mma, q_smem[tic]);
    if(warpgroup::laneid() == 0) {
        mma_AB(wg_tmem.kg, wg_tmem.dp_bf, q_smem[tic], *wg_tmem.mma_sem);
    }
    // warpgroup::mma_async_wait();
    wait(*wg_tmem.mma_sem, 1);
    group<8>::sync(10); 
}

template<typename kg_tile, typename vg_tile>
__device__ static inline void 
kv_store(auto &kg_smem, auto &kg_reg, 
         auto &vg_smem, auto &vg_reg, 
         auto &dst, auto &bar, int kv_head_idx, int toc) 
{
    group<8>::sync(10); 
    warpgroup::store(kg_smem[warpgroup::groupid()], kg_reg);
    group<4>::sync(warpgroup::groupid()+4);
    
    if (kittens::warpid() % 4 == 0 && warp::laneid() == 0) {
        coord<kg_tile> tile_idx = {blockIdx.z, kv_head_idx, (blockIdx.x * BWD_CONSUMER_WARPGROUPS) + warpgroup::groupid(), 0};
        tma::store_add_async(dst.kg, kg_smem[warpgroup::groupid()], tile_idx);
    }

    wait(bar, toc);
    warpgroup::store(vg_smem[warpgroup::groupid()], vg_reg);
    group<4>::sync(warpgroup::groupid()+4);

    if (warpgroup::laneid() == 0) {
        coord<vg_tile> tile_idx = {blockIdx.z, kv_head_idx, (blockIdx.x * BWD_CONSUMER_WARPGROUPS) + warpgroup::groupid(), 0};
        tma::store_add_async(dst.vg, vg_smem[warpgroup::groupid()], tile_idx);
        tma::store_async_wait(); 
    }
}

template<int D, bool is_causal>
__global__ __launch_bounds__(BWD_NUM_WORKERS*kittens::WARP_THREADS, bwd_attend_ker_tile_dims<D>::blocks_sm)
void bwd_attend_ker(const __grid_constant__ bwd_globals<D> g) {
    static_assert(D == 128);
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    const int N = g.N, hr = g.hr;
    using G = bwd_attend_ker_tile_dims<D>;
    
    using kg_tile   = st_fl<G::tile_h, G::tile_width>;
    using vg_tile   = st_fl<G::tile_h, G::tile_width>;
    using k_tile    = st_bf<G::tile_h, G::tile_width>;
    using v_tile    = st_bf<G::tile_h, G::tile_width>;
    using q_tile    = st_bf<G::tile_h_qo, G::tile_width>;
    using og_tile   = st_bf<G::tile_h_qo, G::tile_width>;
    using qg_tile   = st_fl<G::tile_h_qo, G::tile_width>;
    using l_tile    = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using d_tile    = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using attn_tile = st_bf<G::tile_h_qo, G::tile_h>; 

    k_tile  (&k_smem) [BWD_CONSUMER_WARPGROUPS] = al.allocate<k_tile, BWD_CONSUMER_WARPGROUPS>();
    v_tile  (&v_smem) [BWD_CONSUMER_WARPGROUPS] = al.allocate<v_tile, BWD_CONSUMER_WARPGROUPS>();

    q_tile  (&q_smem) [2] = al.allocate<q_tile,  2>(); 
    og_tile (&og_smem)[2] = al.allocate<og_tile, 2>(); 
    qg_tile (&qg_smem)    = al.allocate<qg_tile>();

    l_tile   (&l_smem)[2] = al.allocate<l_tile, 2>();
    d_tile   (&d_smem)[2] = al.allocate<d_tile, 2>();
    kg_tile (*kg_smem)    = reinterpret_cast<kg_tile*>(&k_smem[0].data[0]); 
    vg_tile (*vg_smem)    = reinterpret_cast<vg_tile*>(&q_smem[0].data[0]); 

    attn_tile (&ds_smem)[BWD_CONSUMER_WARPGROUPS] = al.allocate<attn_tile, BWD_CONSUMER_WARPGROUPS>();

    const int warpid      = kittens::warpid();
    const int warpgroupid = warpid/kittens::WARPGROUP_WARPS;
    const int qo_blocks   = N / (G::tile_h_qo);
    const int kv_head_idx = (blockIdx.y) / hr; 

    __shared__ kittens::semaphore kv_b, q_b[2], o_b[2], vec_b[2];
    __shared__ kittens::semaphore compute_done[2], qg_ready;
    __shared__ kittens::semaphore mma_sem[2];

    int tic = 0, toc = 1;
    const int q_start = (is_causal) ? (blockIdx.x * 2) : (0);

    if (threadIdx.x == 0) {
        init_semaphore(kv_b,  0, 1);
        init_semaphore(qg_ready, 1, 0);
        #pragma unroll
        for (int s = 0; s < 2; s++) {
            init_semaphore(q_b[s],   0, 1);
            init_semaphore(o_b[s],   0, 1); 
            init_semaphore(vec_b[s], 0, 1);
            init_semaphore(compute_done[s], 1, 0);
            init_semaphore(mma_sem[s], 0, 2);
        }

        tma::expect_bytes(kv_b, (sizeof(k_smem[0]) + sizeof(v_smem[0])) * BWD_CONSUMER_WARPGROUPS);
        #pragma unroll
        for (int w = 0; w < BWD_CONSUMER_WARPGROUPS; w++) {
            coord<k_tile> tile_idx = {blockIdx.z, kv_head_idx, (blockIdx.x * BWD_CONSUMER_WARPGROUPS) + w, 0};
            tma::load_async(k_smem[w], g.k, tile_idx, kv_b);
            tma::load_async(v_smem[w], g.v, tile_idx, kv_b);
        }

        coord<q_tile> tile_idx = {blockIdx.z, blockIdx.y, q_start, 0};
        tma::expect_bytes(q_b[tic],   sizeof(q_smem[0]));
        tma::load_async(q_smem[tic],  g.q,  tile_idx, q_b[tic]);
        tma::expect_bytes(o_b[tic],   sizeof(og_smem[0]));
        tma::load_async(og_smem[tic], g.og, tile_idx, o_b[tic]);

        coord<l_tile> vec_idx = {blockIdx.z, blockIdx.y, 0, q_start};
        tma::expect_bytes(vec_b[tic], sizeof(l_smem[0]) + sizeof(d_smem[0]));
        tma::load_async(l_smem[tic], g.l, vec_idx, vec_b[tic]);
        tma::load_async(d_smem[tic], g.d, vec_idx, vec_b[tic]);
    }

    tensor_allocator<1, 1> tm_alloc{};
    auto kg_tt = tm_alloc.allocate<tt<float, 64, 128>>(warpgroupid, 0);
    auto vg_tt = tm_alloc.allocate<tt<float, 64, 128>>(warpgroupid, 128);
    auto sb_tt = tm_alloc.allocate<tt<float, 64, 64>>(warpgroupid, 256);
    auto &pb_tt_bf = reinterpret_cast<tt<bf16, 64, 64>&>(sb_tt);
    auto dp_tt = tm_alloc.allocate<tt<float, 64, 64>>(warpgroupid, 320);
    auto &dp_tt_bf = reinterpret_cast<tt<bf16, 64, 64>&>(dp_tt);
    auto qg_tt = tm_alloc.allocate<tt<float, 64, 128>>(0, 384); // Just used by warpgroupid 0.

    if(warpgroupid < 2) {
        rt_fl<16, 128> z;
        warp::zero(z);
        warpgroup::store_async(kg_tt, z);
        warpgroup::store_async(vg_tt, z);
        tensor_store_wait();
    }

    wg_tmem_t wg_tmem{kg_tt, vg_tt, sb_tt, dp_tt, pb_tt_bf, dp_tt_bf, &mma_sem[warpgroupid]};

    __syncthreads(); 

    if (warpgroupid == BWD_NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<24>();

        if (warpid % kittens::WARPGROUP_WARPS == 0 && warp::laneid() == 0) {
            for (auto qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {
                if (qo_idx + 1 < qo_blocks) {
                    coord<q_tile> tile_idx = {blockIdx.z, blockIdx.y, qo_idx + 1, 0};
                    tma::expect_bytes(q_b[toc],   sizeof(q_smem[0])); 
                    tma::load_async(q_smem[toc], g.q,  tile_idx, q_b[toc]);
                    tma::expect_bytes(o_b[toc],   sizeof(og_smem[0]));
                    tma::load_async(og_smem[toc], g.og, tile_idx, o_b[toc]);

                    coord<l_tile> vec_idx = {blockIdx.z, blockIdx.y, 0, qo_idx + 1};
                    tma::expect_bytes(vec_b[toc], sizeof(l_smem[0]) + sizeof(d_smem[0]));
                    tma::load_async(l_smem[toc], g.l, vec_idx, vec_b[toc]);
                    tma::load_async(d_smem[toc], g.d, vec_idx, vec_b[toc]);
                }
                
                wait(compute_done[tic], ((qo_idx - q_start)/(2))%2);
            }
        }
        else if(warpid % WARPGROUP_WARPS == 1 && warp::laneid() == 0) {
            for (auto qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {
                wait(compute_done[tic], ((qo_idx - q_start)/(2))%2);
                
                coord<qg_tile> tile_idx = {blockIdx.z, blockIdx.y, qo_idx, 0};
                tma::store_add_async(g.qg, qg_smem, tile_idx);
                tma::store_async_wait();
                
                if(laneid() == 0) arrive(qg_ready); 
            }
        }
    }
    else {
        rt_fl<16, G::tile_width> kg_reg, vg_reg;
    
        row_vec<rt_fl<16, 64>> row_reg; 

        rt_fl<16, 64> s_block_t,  p_block_t; 
        rt_fl<16, 64> ds_block_t, dp_block_t; 
        rt_bf<16, 64> ds_block_t_mma, p_block_t_mma;

        if (warpgroupid == 0) {
            warpgroup::increase_registers<256>();
            wait(kv_b, 0);
            for (int qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {
                compute_bwd_loop<is_causal, G::tile_h_qo, G::tile_h, G::tile_width, D>(
                    wg_tmem,
                    vec_b, q_b, o_b,
                    s_block_t, dp_block_t, p_block_t, ds_block_t, p_block_t_mma, ds_block_t_mma,
                    kg_reg, vg_reg,
                    q_smem, k_smem, v_smem, og_smem, ds_smem, l_smem, d_smem,
                    qo_idx, q_start, tic, toc
                );

                rt_fl<16, G::tile_width> qg_reg; 
                // warpgroup::mm_AtB(qg_reg, ds_smem[0], k_smem[0]);
                // warpgroup::mma_AtB(qg_reg, ds_smem[1], k_smem[1]);
                if(warpgroup::laneid() == 0) {
                    mm_AtB(qg_tt, ds_smem[0], k_smem[0], mma_sem[0]);
                    mma_AtB(qg_tt, ds_smem[1], k_smem[1], mma_sem[0]);
                }
    
                wait(qg_ready, toc);
                if (qo_idx > 0) tma::store_async_wait();

                // warpgroup::mma_async_wait();
                wait(mma_sem[0], 0);
                if(warpgroup::laneid() == 0) arrive(mma_sem[0], 2);
                warpgroup::load_async(qg_reg, qg_tt);
                warpgroup::store(qg_smem, qg_reg);
                group<4>::sync(warpgroup::groupid()+4);
    
                if (warpgroup::laneid() == 0) arrive(compute_done[tic]);
            }
            warpgroup::load_async(kg_reg, kg_tt);
            warpgroup::load_async(vg_reg, vg_tt);
            // one(kg_reg);
            // one(vg_reg);
            kv_store<kg_tile, vg_tile>(kg_smem, kg_reg, vg_smem, vg_reg, g, qg_ready, kv_head_idx, toc);
        }
        else {
            warpgroup::increase_registers<224>();
            wait(kv_b, 0);
            for (int qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {
                compute_bwd_loop<is_causal, G::tile_h_qo, G::tile_h, G::tile_width, D>(
                    wg_tmem,
                    vec_b, q_b, o_b,
                    s_block_t, dp_block_t, p_block_t, ds_block_t, p_block_t_mma, ds_block_t_mma,
                    kg_reg, vg_reg,
                    q_smem, k_smem, v_smem, og_smem, ds_smem, l_smem, d_smem,
                    qo_idx, q_start, tic, toc
                );
            }
            warpgroup::load_async(kg_reg, kg_tt);
            warpgroup::load_async(vg_reg, vg_tt);
            // one(kg_reg);
            // one(vg_reg);
            kv_store<kg_tile, vg_tile>(kg_smem, kg_reg, vg_smem, vg_reg, g, qg_ready, kv_head_idx, toc);
        }
    }
}

PYBIND11_MODULE(_C, m) {
    m.doc() = "";
    py::bind_kernel<bf16_mha_fwd::kernel>(m, "bf16_mha_fwd",
        &bf16_mha_fwd::globals::Q,
        &bf16_mha_fwd::globals::K,
        &bf16_mha_fwd::globals::V,
        &bf16_mha_fwd::globals::L,
        &bf16_mha_fwd::globals::O
    );
    py::bind_kernel<bf16_mha_bwd_prep::kernel>(m, "bf16_mha_bwd_prep",
        &bf16_mha_bwd_prep::globals::O_grad,
        &bf16_mha_bwd_prep::globals::O,
        &bf16_mha_bwd_prep::globals::D
    );
    py::bind_kernel<bf16_mha_bwd::kernel>(m, "bf16_mha_bwd",
        &bf16_mha_bwd::globals::Q,
        &bf16_mha_bwd::globals::K,
        &bf16_mha_bwd::globals::V,
        &bf16_mha_bwd::globals::O_grad,
        &bf16_mha_bwd::globals::Q_grad,
        &bf16_mha_bwd::globals::K_grad,
        &bf16_mha_bwd::globals::V_grad,
        &bf16_mha_bwd::globals::L,
        &bf16_mha_bwd::globals::D
    );
    py::bind_kernel<fwd_attend_ker<128, false>>(m, "fwd_attend_ker_128_noncausal", &fwd_globals<128>::q, &fwd_globals<128>::k, &fwd_globals<128>::v, &fwd_globals<128>::l, &fwd_globals<128>::o);
    py::bind_kernel<bwd_attend_prep_ker<128>>(m, "bwd_attend_prep_ker_128", &bwd_prep_globals<128>::og, &bwd_prep_globals<128>::o, &bwd_prep_globals<128>::d);
    py::bind_kernel<bwd_attend_ker<128, false>>(m, "bwd_attend_ker_128_noncausal", &bwd_globals<128>::q, &bwd_globals<128>::k, &bwd_globals<128>::v, &bwd_globals<128>::og, &bwd_globals<128>::qg, &bwd_globals<128>::kg, &bwd_globals<128>::vg, &bwd_globals<128>::l, &bwd_globals<128>::d, &bwd_globals<128>::N, &bwd_globals<128>::hr);
}
