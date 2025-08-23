#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

constexpr int BLOCKIDX = 0;

namespace bf16_mha_fwd {

struct config {
    static constexpr int CLUSTER_SIZE = 2;

    static constexpr int NUM_BLOCKS = 148;
    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int NUM_WARPGROUPS = 4; // 8 softmax + 4 scaling & epilogue + 1 TMA + 1 MMA + 2 unused
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS; 
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};

struct globals {
    static constexpr int BLOCK_SIZE = 128;

    static constexpr int QK_DIM = 128;
    static constexpr int VO_DIM = 128;

    static constexpr int PIPELINE_STAGES = 3;
    static constexpr int NUM_PIPELINES = 2;

    using Q_tile = st_bf<BLOCK_SIZE, QK_DIM>;              // 32768 B
    using K_tile = st_bf<BLOCK_SIZE / 2, QK_DIM>;          // 16384 B
    using V_tile = st_bf<BLOCK_SIZE, VO_DIM / 2>;          // 16384 B
    using M_vec  = col_vec<st_fl<BLOCK_SIZE, BLOCK_SIZE>>; // 512   B
    using L_vec  = col_vec<st_fl<BLOCK_SIZE, VO_DIM>>;     // 512   B
    using O_tile = st_bf<BLOCK_SIZE, VO_DIM>;              // 32768 B

    using Q_gl = gl<bf16,  -1, -1, -1, -1, Q_tile>; // B, H, N, D
    using K_gl = gl<bf16,  -1, -1, -1, -1, K_tile>;
    using V_gl = gl<bf16,  -1, -1, -1, -1, V_tile>;
    using L_gl = gl<float, -1, -1, -1, -1, L_vec>;
    using O_gl = gl<bf16,  -1, -1, -1, -1, tma::descriptor<O_tile, 2, false>>; // disable swizzle

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

__device__ inline globals::task_info get_task_info(const globals &G, int task_idx) {
    constexpr int Q_block_size = config::CLUSTER_SIZE * globals::NUM_PIPELINES * globals::BLOCK_SIZE;

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
    task_info.KV_block_end = G.K.rows() / globals::BLOCK_SIZE;

    if (task_info.batch_idx >= B)
        return { -1, -1, -1 };
    else
        return task_info;
}

__global__ __cluster_dims__(config::CLUSTER_SIZE) __launch_bounds__(config::NUM_THREADS, 1)
static void kernel(const __grid_constant__ globals G) {
    // Declare shared memory
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);

    // Allocate shared memory
    static_assert(
        sizeof(globals::Q_tile) * globals::NUM_PIPELINES +   // 65536 B
        sizeof(globals::K_tile) * globals::PIPELINE_STAGES + // 49152 B
        sizeof(globals::V_tile) * globals::PIPELINE_STAGES + // 49152 B
        sizeof(globals::L_vec) * globals::NUM_PIPELINES +    // 1024  B
        sizeof(globals::M_vec) * globals::NUM_PIPELINES <=   // 1024  B
        config::DYNAMIC_SHARED_MEMORY
    );
    globals::Q_tile (&Q_smem)[globals::NUM_PIPELINES]   = sm_allocator.allocate<globals::Q_tile, globals::NUM_PIPELINES>();
    globals::K_tile (&K_smem)[globals::PIPELINE_STAGES] = sm_allocator.allocate<globals::K_tile, globals::PIPELINE_STAGES>();
    globals::V_tile (&V_smem)[globals::PIPELINE_STAGES] = sm_allocator.allocate<globals::V_tile, globals::PIPELINE_STAGES>();
    globals::L_vec  (&L_smem)[globals::NUM_PIPELINES]   = sm_allocator.allocate<globals::L_vec, globals::NUM_PIPELINES>();
    globals::M_vec  (&M_smem)[globals::NUM_PIPELINES]   = sm_allocator.allocate<globals::M_vec, globals::NUM_PIPELINES>();
    globals::O_tile (&O_smem)[globals::NUM_PIPELINES]   = *reinterpret_cast<globals::O_tile(*)[globals::NUM_PIPELINES]>(&Q_smem[0]);

    // Allocate tensor memory
    tensor_allocator<1, config::CLUSTER_SIZE> tm_allocator {};

    // Set up mbarriers
    __shared__ semaphore Q_arrived[globals::NUM_PIPELINES];    // TMA load completion for Q tiles
    __shared__ semaphore Q_finished[globals::NUM_PIPELINES];   // Q processing completion (end of current tile)
    __shared__ semaphore K_arrived[globals::PIPELINE_STAGES];  // TMA load completion for K tiles
    __shared__ semaphore K_finished[globals::PIPELINE_STAGES]; // K processing completion (end of 2 QK^T matmuls)
    __shared__ semaphore V_arrived[globals::PIPELINE_STAGES];  // TMA load completion for V tiles
    __shared__ semaphore V_finished[globals::PIPELINE_STAGES]; // V processing completion (end of 2 PV matmuls)
    __shared__ semaphore S_arrived[globals::NUM_PIPELINES];    // QK^T matmul completion
    __shared__ semaphore S_finished;                           // S TMEM emptied (finished loading into registers)
    __shared__ semaphore P_arrived[globals::NUM_PIPELINES];    // P matrix ready for PV matmul
    __shared__ semaphore P_finished[globals::NUM_PIPELINES];   // PV matmul completion
    __shared__ semaphore O_arrived[globals::NUM_PIPELINES];    // PV matmul completion
    __shared__ semaphore O_ready[globals::NUM_PIPELINES];      // O ready for next PV accumulation 
    __shared__ semaphore M_arrived[globals::NUM_PIPELINES];    // Row max SMEM ready (stored in SMEM)
    __shared__ semaphore M_finished[globals::NUM_PIPELINES];   // Row max SMEM emptied (loaded to registers)
    __shared__ semaphore L_arrived[globals::NUM_PIPELINES];    // Row sum SMEM ready (stored in SMEM)
    __shared__ semaphore L_finished[globals::NUM_PIPELINES];   // Row sum SMEM emptied (loaded to registers)
    if (threadIdx.x == 0) {
        init_semaphore(S_finished, 0, config::CLUSTER_SIZE);          // Must wait for one of the CTAs to arrive
        #pragma unroll
        for (int i = 0; i < globals::PIPELINE_STAGES; ++i) {
            init_semaphore(K_arrived[i], 0, config::CLUSTER_SIZE);    // Must wait for 2 CTAs' TMA loads before 2-CTA QK^T
            init_semaphore(K_finished[i], 0, globals::NUM_PIPELINES); // Must wait for 2 pipelines to finish QK^T
            init_semaphore(V_arrived[i], 0, config::CLUSTER_SIZE);    // Must wait for 2 CTAs' TMA loads before 2-CTA PV
            init_semaphore(V_finished[i], 0, globals::NUM_PIPELINES); // Must wait for 2 pipelines to finish PV
        }
        #pragma unroll
        for (int i = 0; i < globals::NUM_PIPELINES; ++i) {
            init_semaphore(Q_arrived[i], 0, config::CLUSTER_SIZE);  // Must wait for 2 CTAs' TMA loads before 2-CTA QK^T
            init_semaphore(Q_finished[i], 0, 1);                    // Must wait for a single 2-CTA QK^T
            init_semaphore(S_arrived[i], 0, 1);                     // Must wait for a single 2-CTA QK^T
            init_semaphore(P_arrived[i], 0, config::CLUSTER_SIZE);  // Must wait for 2 CTAs' TMEM loads before 2-CTA PV
            init_semaphore(P_finished[i], 0, 1);                    // Must wait for a single 2-CTA PV
            init_semaphore(O_arrived[i], 0, 1);                     // Must wait for a single 2-CTA PV
            init_semaphore(O_ready[i], 0, config::CLUSTER_SIZE);    // Must wait for 2 CTAs to complete O processing
            init_semaphore(M_arrived[i], 0, 1);                     // Must wait for current pipeline only
            init_semaphore(M_finished[i], 0, 1);                    // Must wait for current pipeline only
            init_semaphore(L_arrived[i], 0, 1);                     // Must wait for current pipeline only
            init_semaphore(L_finished[i], 0, 1);                    // Must wait for current pipeline only
        }
    }
    everyone::tma::cluster::sync();

    // Pipeline configuration
    const int cluster_id = clusterIdx().x;
    const int cta_id = cluster_ctarank();
    const int warp_id = ::warpid();
    const int lane_id = warp::laneid();
    uint32_t QK_stage = 0;
    uint32_t PV_stage = 0;
    uint32_t phasebits = 0xFFFF'0000;

    // Constants
    constexpr float SQRT_D_INV = 0.08838834764f; // 1 / sqrt(128)
    constexpr float NEG_SQRT_D = -11.313708499f; // -sqrt(128)
    constexpr float LOG2E = 1.44269504089f;

    if (warp_id < 8) { // 2 softmax groups
        // warpgroup::increase_registers<176>();
        constexpr int S_ARRIVED_PB_POS = 0;
        constexpr int P_FINISHED_PB_POS = 1;
        constexpr int M_FINISHED_PB_POS = 2;
        constexpr int L_FINISHED_PB_POS = 3;

        const int pipeline_id = warpgroup::groupid();
        using softmax_group = group<8>;

        uint32_t S_tm = tm_allocator.addr + 0;
        uint32_t P_tm = tm_allocator.addr + globals::BLOCK_SIZE + (globals::BLOCK_SIZE / 2) * pipeline_id;

        for (int task_idx = cluster_id; true; task_idx += gridDim.x / config::CLUSTER_SIZE) {
            globals::task_info task_info = get_task_info(G, task_idx);
            if (task_info.batch_idx == -1) break;

            if ((threadIdx.x == 0 || threadIdx.x == 128) && blockIdx.x == BLOCKIDX) printf("Task ID %d-%d\n", pipeline_id, task_idx);
            
            float row_max = 0xFF800000; // -inf
            float row_sum = 0.f;
            
            for (int i = task_info.KV_block_start; i < task_info.KV_block_end; ++i) {
                if ((threadIdx.x == 0 || threadIdx.x == 128) && blockIdx.x == BLOCKIDX) printf("KV index %d-%d\n", pipeline_id, i);
                
                // if (warp_id >= 4) softmax_group::sync(1);
                if ((threadIdx.x == 0 || threadIdx.x == 128) && blockIdx.x == BLOCKIDX) printf("Waiting for S = QK finished%d-%d\n", pipeline_id, i);
                tma::cluster::wait(S_arrived[pipeline_id], get_phasebit<0>(phasebits, S_ARRIVED_PB_POS));
                if ((threadIdx.x == 0 || threadIdx.x == 128) && blockIdx.x == BLOCKIDX) printf("Done waiting for S = QK finished%d-%d\n", pipeline_id, i);
                update_phasebit<0>(phasebits, S_ARRIVED_PB_POS);
                warpgroup::arrive(K_finished[QK_stage]);
                QK_stage = (QK_stage + 1) % globals::PIPELINE_STAGES;

                if (i == task_info.KV_block_end - 1)
                    warpgroup::arrive(Q_finished[pipeline_id]);

                // TMEM --> registers
                float2 SP_reg[globals::BLOCK_SIZE / 2];
                #pragma unroll
                for (int ii = 0; ii < globals::BLOCK_SIZE / 32; ii++) {
                    asm volatile("{tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];}"
                        : "=f"(SP_reg[ii * 16 + 0].x), "=f"(SP_reg[ii * 16 + 0].y), "=f"(SP_reg[ii * 16 + 1].x), "=f"(SP_reg[ii * 16 + 1].y), "=f"(SP_reg[ii * 16 + 2].x), "=f"(SP_reg[ii * 16 + 2].y), "=f"(SP_reg[ii * 16 + 3].x), "=f"(SP_reg[ii * 16 + 3].y), 
                          "=f"(SP_reg[ii * 16 + 4].x), "=f"(SP_reg[ii * 16 + 4].y), "=f"(SP_reg[ii * 16 + 5].x), "=f"(SP_reg[ii * 16 + 5].y), "=f"(SP_reg[ii * 16 + 6].x), "=f"(SP_reg[ii * 16 + 6].y), "=f"(SP_reg[ii * 16 + 7].x), "=f"(SP_reg[ii * 16 + 7].y),
                          "=f"(SP_reg[ii * 16 + 8].x), "=f"(SP_reg[ii * 16 + 8].y), "=f"(SP_reg[ii * 16 + 9].x), "=f"(SP_reg[ii * 16 + 9].y), "=f"(SP_reg[ii * 16 + 10].x), "=f"(SP_reg[ii * 16 + 10].y), "=f"(SP_reg[ii * 16 + 11].x), "=f"(SP_reg[ii * 16 + 11].y),
                          "=f"(SP_reg[ii * 16 + 12].x), "=f"(SP_reg[ii * 16 + 12].y), "=f"(SP_reg[ii * 16 + 13].x), "=f"(SP_reg[ii * 16 + 13].y), "=f"(SP_reg[ii * 16 + 14].x), "=f"(SP_reg[ii * 16 + 14].y), "=f"(SP_reg[ii * 16 + 15].x), "=f"(SP_reg[ii * 16 + 15].y)
                        : "r"(S_tm + ii * 32));
                }
                float last_row_max_scaled = row_max * (LOG2E * SQRT_D_INV);
                asm volatile("{tcgen05.wait::ld.sync.aligned;}");
                warpgroup::tma::cluster::arrive(S_finished, 0);
                // if (warp_id < 4) softmax_group::sync(1);
                // softmax_group::sync(15);

                // Row-wise max
                #pragma unroll
                for (int ii = 0; ii < globals::BLOCK_SIZE / 2; ii++) {
                    asm volatile("{max.f32 %0, %1, %2, %3;}"
                        : "=f"(row_max)
                        : "f"(row_max), "f"(SP_reg[ii].x), "f"(SP_reg[ii].y));
                }

                // Prepare scales
                float S_scale = row_max * (-LOG2E * SQRT_D_INV);
                float O_scale = last_row_max_scaled + S_scale;

                // Send O scale to correction group
                if (i > task_info.KV_block_start) {
                    // wait(M_finished[pipeline_id], get_phasebit<1>(phasebits, M_FINISHED_PB_POS));
                    // update_phasebit<1>(phasebits, M_FINISHED_PB_POS);
                    // M_smem[pipeline_id][warpgroup::laneid()] = O_scale;
                    // warpgroup::arrive(M_arrived[pipeline_id]);
                }

                // Prepare S scales
                float2 S_scale_2 = {S_scale, S_scale};
                constexpr float2 log_scale_2 = {LOG2E * SQRT_D_INV, LOG2E * SQRT_D_INV};

                #pragma unroll
                for (int ii = 0; ii < globals::BLOCK_SIZE / 2; ii++) {
                    SP_reg[ii] = __ffma2_rn(SP_reg[ii], log_scale_2, S_scale_2);
                    SP_reg[ii].x = exp2f(SP_reg[ii].x);
                    SP_reg[ii].y = exp2f(SP_reg[ii].y);
                }

                // Registers --> TMEM                
                if ((threadIdx.x == 0 || threadIdx.x == 128) && blockIdx.x == BLOCKIDX) printf("Waiting for PV finished%d-%d\n", pipeline_id, i);
                wait(P_finished[pipeline_id], get_phasebit<1>(phasebits, P_FINISHED_PB_POS));
                if ((threadIdx.x == 0 || threadIdx.x == 128) && blockIdx.x == BLOCKIDX) printf("Done waiting for PV finished%d-%d\n", pipeline_id, i);
                update_phasebit<1>(phasebits, P_FINISHED_PB_POS);
                #pragma unroll
                for (int ii = 0; ii < globals::BLOCK_SIZE / 32; ii++) {
                    bf16_2 SP_bf_reg[16];
                    #pragma unroll
                    for (int jj = 0; jj < 16; jj++) {
                        SP_bf_reg[jj] = __float22bfloat162_rn(SP_reg[ii * 16 + jj]);
                    }
                    asm volatile("{tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15};}"
                        :: "r"(*reinterpret_cast<uint32_t *>(&SP_bf_reg[0])), "r"(*reinterpret_cast<uint32_t *>(&SP_bf_reg[1])), "r"(*reinterpret_cast<uint32_t *>(&SP_bf_reg[2])), "r"(*reinterpret_cast<uint32_t *>(&SP_bf_reg[3])),
                           "r"(*reinterpret_cast<uint32_t *>(&SP_bf_reg[4])), "r"(*reinterpret_cast<uint32_t *>(&SP_bf_reg[5])), "r"(*reinterpret_cast<uint32_t *>(&SP_bf_reg[6])), "r"(*reinterpret_cast<uint32_t *>(&SP_bf_reg[7])),
                           "r"(*reinterpret_cast<uint32_t *>(&SP_bf_reg[8])), "r"(*reinterpret_cast<uint32_t *>(&SP_bf_reg[9])), "r"(*reinterpret_cast<uint32_t *>(&SP_bf_reg[10])), "r"(*reinterpret_cast<uint32_t *>(&SP_bf_reg[11])),
                           "r"(*reinterpret_cast<uint32_t *>(&SP_bf_reg[12])), "r"(*reinterpret_cast<uint32_t *>(&SP_bf_reg[13])), "r"(*reinterpret_cast<uint32_t *>(&SP_bf_reg[14])), "r"(*reinterpret_cast<uint32_t *>(&SP_bf_reg[15])),
                           "r"(P_tm + ii * 16));
                }

                // Signal PV launcher
                asm volatile("{tcgen05.wait::st.sync.aligned;}");
                warpgroup::sync(pipeline_id + 1);
                warpgroup::tma::cluster::arrive(P_arrived[pipeline_id], 0);

                // Get row-wise sum
                float2 local_sum = {0.0f, 0.0f};
                #pragma unroll
                for (int ii = 0; ii < globals::BLOCK_SIZE / 2; ii++) {
                    local_sum = __fadd2_rn(local_sum, SP_reg[ii]);
                }
                row_sum *= exp2f(O_scale); // scale previous sum
                row_sum += local_sum.x + local_sum.y;
            }

            group<8>::sync(7);

            // Save for epilogue
            if ((threadIdx.x == 0 || threadIdx.x == 128) && blockIdx.x == BLOCKIDX) printf("Saving for epilogue%d\n", pipeline_id);
            if (task_info.KV_block_start < task_info.KV_block_end) {
                // Send O scale to correction group
                // wait(M_finished[pipeline_id], get_phasebit<1>(phasebits, M_FINISHED_PB_POS));
                // update_phasebit<1>(phasebits, M_FINISHED_PB_POS);
                // M_smem[pipeline_id][warpgroup::laneid()] = row_max;
                // warpgroup::arrive(M_arrived[pipeline_id]);
                // wait(L_finished[pipeline_id], get_phasebit<1>(phasebits, L_FINISHED_PB_POS));
                // update_phasebit<1>(phasebits, L_FINISHED_PB_POS);
                // L_smem[pipeline_id][warpgroup::laneid()] = row_sum;
                // warpgroup::arrive(L_arrived[pipeline_id]);
            }
            group<8>::sync(7);
            if ((threadIdx.x == 0 || threadIdx.x == 128) && blockIdx.x == BLOCKIDX) printf("Saving for epilogue done%d\n", pipeline_id);
        }
    } else if (warp_id < 12) { // Scale & epilogue group
        // warpgroup::decrease_registers<48>();
        constexpr int O_ARRIVED_PB_POS = 0;
        constexpr int M_ARRIVED_PB_POS = O_ARRIVED_PB_POS + globals::NUM_PIPELINES;
        constexpr int L_ARRIVED_PB_POS = M_ARRIVED_PB_POS + globals::NUM_PIPELINES;;

        uint32_t O_tm[globals::NUM_PIPELINES] = {
            tm_allocator.addr + globals::BLOCK_SIZE * 2,
            tm_allocator.addr + globals::BLOCK_SIZE * 3
        };

        for (int task_idx = cluster_id; true; task_idx += gridDim.x / config::CLUSTER_SIZE) {
            globals::task_info task_info = get_task_info(G, task_idx);
            if (task_info.batch_idx == -1) break;

            for (int i = task_info.KV_block_start + 1; i < task_info.KV_block_end; ++i) { // skip the first iteration
                #pragma unroll
                for (int pipeline_id = 0; pipeline_id < globals::NUM_PIPELINES; ++pipeline_id) {
                    warpgroup::tma::cluster::wait(O_arrived[pipeline_id], get_phasebit<0>(phasebits, O_ARRIVED_PB_POS + pipeline_id));
                    update_phasebit<0>(phasebits, O_ARRIVED_PB_POS + pipeline_id);
                    warpgroup::arrive(V_finished[PV_stage]);
                    warpgroup::arrive(P_finished[pipeline_id]); // TODO: maybe it's better for softmax group to wait directly on O_arrived

                    float2 O_scale_2;

                    #pragma unroll
                    for (int ii = 0; ii < 4; ++ii) {
                        // TMEM --> registers
                        float2 O_reg[globals::BLOCK_SIZE / 4 / 2];
                        asm volatile("{tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];}"
                            : "=f"(O_reg[ii * 16 + 0].x), "=f"(O_reg[ii * 16 + 0].y), "=f"(O_reg[ii * 16 + 1].x), "=f"(O_reg[ii * 16 + 1].y), "=f"(O_reg[ii * 16 + 2].x), "=f"(O_reg[ii * 16 + 2].y), "=f"(O_reg[ii * 16 + 3].x), "=f"(O_reg[ii * 16 + 3].y), 
                              "=f"(O_reg[ii * 16 + 4].x), "=f"(O_reg[ii * 16 + 4].y), "=f"(O_reg[ii * 16 + 5].x), "=f"(O_reg[ii * 16 + 5].y), "=f"(O_reg[ii * 16 + 6].x), "=f"(O_reg[ii * 16 + 6].y), "=f"(O_reg[ii * 16 + 7].x), "=f"(O_reg[ii * 16 + 7].y),
                              "=f"(O_reg[ii * 16 + 8].x), "=f"(O_reg[ii * 16 + 8].y), "=f"(O_reg[ii * 16 + 9].x), "=f"(O_reg[ii * 16 + 9].y), "=f"(O_reg[ii * 16 + 10].x), "=f"(O_reg[ii * 16 + 10].y), "=f"(O_reg[ii * 16 + 11].x), "=f"(O_reg[ii * 16 + 11].y),
                              "=f"(O_reg[ii * 16 + 12].x), "=f"(O_reg[ii * 16 + 12].y), "=f"(O_reg[ii * 16 + 13].x), "=f"(O_reg[ii * 16 + 13].y), "=f"(O_reg[ii * 16 + 14].x), "=f"(O_reg[ii * 16 + 14].y), "=f"(O_reg[ii * 16 + 15].x), "=f"(O_reg[ii * 16 + 15].y)
                            : "r"(O_tm[pipeline_id] + ii * 32));

                        // Initially, load the scale
                        if (ii == 0) {
                            // wait(M_arrived[pipeline_id], get_phasebit<0>(phasebits, M_ARRIVED_PB_POS + pipeline_id));
                            // update_phasebit<0>(phasebits, M_ARRIVED_PB_POS + pipeline_id);
                            // O_scale_2.x = exp2f(M_smem[pipeline_id][warpgroup::laneid()]);
                            // O_scale_2.y = O_scale_2.x;
                        }

                        // Rescale O
                        #pragma unroll
                        for (int jj = 0; jj < globals::BLOCK_SIZE / 4 / 2; jj++) {
                            O_reg[jj] = __fmul2_rn(O_reg[jj], O_scale_2);
                        }

                        // Registers --> TMEM
                        asm volatile("{tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31};}"
                            :: "f"(O_reg[ii * 16 + 0].x), "f"(O_reg[ii * 16 + 0].y), "f"(O_reg[ii * 16 + 1].x), "f"(O_reg[ii * 16 + 1].y), "f"(O_reg[ii * 16 + 2].x), "f"(O_reg[ii * 16 + 2].y), "f"(O_reg[ii * 16 + 3].x), "f"(O_reg[ii * 16 + 3].y), 
                               "f"(O_reg[ii * 16 + 4].x), "f"(O_reg[ii * 16 + 4].y), "f"(O_reg[ii * 16 + 5].x), "f"(O_reg[ii * 16 + 5].y), "f"(O_reg[ii * 16 + 6].x), "f"(O_reg[ii * 16 + 6].y), "f"(O_reg[ii * 16 + 7].x), "f"(O_reg[ii * 16 + 7].y),
                               "f"(O_reg[ii * 16 + 8].x), "f"(O_reg[ii * 16 + 8].y), "f"(O_reg[ii * 16 + 9].x), "f"(O_reg[ii * 16 + 9].y), "f"(O_reg[ii * 16 + 10].x), "f"(O_reg[ii * 16 + 10].y), "f"(O_reg[ii * 16 + 11].x), "f"(O_reg[ii * 16 + 11].y),
                               "f"(O_reg[ii * 16 + 12].x), "f"(O_reg[ii * 16 + 12].y), "f"(O_reg[ii * 16 + 13].x), "f"(O_reg[ii * 16 + 13].y), "f"(O_reg[ii * 16 + 14].x), "f"(O_reg[ii * 16 + 14].y), "f"(O_reg[ii * 16 + 15].x), "f"(O_reg[ii * 16 + 15].y)
                               "r"(O_tm[pipeline_id] + ii * 32));
                    }

                    asm volatile("{tcgen05.wait::st.sync.aligned;}");
                    warpgroup::sync(3);
                    warpgroup::tma::cluster::arrive(O_ready[pipeline_id], 0);
                    warpgroup::arrive(M_finished[pipeline_id]);
                }

                PV_stage = (PV_stage + 1) % globals::PIPELINE_STAGES;
            }

            // Epilogue
            group<4>::sync(11);
            if (warpgroup::laneid() == 0 && blockIdx.x == BLOCKIDX) printf("Starting epilogue\n");
            if (task_info.KV_block_start < task_info.KV_block_end) {
                #pragma unroll
                for (int pipeline_id = 0; pipeline_id < globals::NUM_PIPELINES; ++pipeline_id) {
                    warpgroup::tma::cluster::wait(O_arrived[pipeline_id], get_phasebit<0>(phasebits, O_ARRIVED_PB_POS + pipeline_id));
                    update_phasebit<0>(phasebits, O_ARRIVED_PB_POS + pipeline_id);
                    warpgroup::arrive(V_finished[PV_stage]);
                    warpgroup::arrive(P_finished[pipeline_id]);

                    float row_max;
                    float row_sum;

                    #pragma unroll
                    for (int ii = 0; ii < 4; ++ii) {
                        float2 O_reg[globals::BLOCK_SIZE / 4 / 2];
                        asm volatile("{tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];}"
                            : "=f"(O_reg[ii * 16 + 0].x), "=f"(O_reg[ii * 16 + 0].y), "=f"(O_reg[ii * 16 + 1].x), "=f"(O_reg[ii * 16 + 1].y), "=f"(O_reg[ii * 16 + 2].x), "=f"(O_reg[ii * 16 + 2].y), "=f"(O_reg[ii * 16 + 3].x), "=f"(O_reg[ii * 16 + 3].y), 
                            "=f"(O_reg[ii * 16 + 4].x), "=f"(O_reg[ii * 16 + 4].y), "=f"(O_reg[ii * 16 + 5].x), "=f"(O_reg[ii * 16 + 5].y), "=f"(O_reg[ii * 16 + 6].x), "=f"(O_reg[ii * 16 + 6].y), "=f"(O_reg[ii * 16 + 7].x), "=f"(O_reg[ii * 16 + 7].y),
                            "=f"(O_reg[ii * 16 + 8].x), "=f"(O_reg[ii * 16 + 8].y), "=f"(O_reg[ii * 16 + 9].x), "=f"(O_reg[ii * 16 + 9].y), "=f"(O_reg[ii * 16 + 10].x), "=f"(O_reg[ii * 16 + 10].y), "=f"(O_reg[ii * 16 + 11].x), "=f"(O_reg[ii * 16 + 11].y),
                            "=f"(O_reg[ii * 16 + 12].x), "=f"(O_reg[ii * 16 + 12].y), "=f"(O_reg[ii * 16 + 13].x), "=f"(O_reg[ii * 16 + 13].y), "=f"(O_reg[ii * 16 + 14].x), "=f"(O_reg[ii * 16 + 14].y), "=f"(O_reg[ii * 16 + 15].x), "=f"(O_reg[ii * 16 + 15].y)
                            : "r"(O_tm[pipeline_id] + ii * 32));

                        if (ii == 0) {
                            // wait(M_arrived[pipeline_id], get_phasebit<0>(phasebits, M_ARRIVED_PB_POS + pipeline_id));
                            // update_phasebit<0>(phasebits, M_ARRIVED_PB_POS + pipeline_id);
                            // row_max = M_smem[pipeline_id][warpgroup::laneid()] * SQRT_D_INV;
                            // wait(L_arrived[pipeline_id], get_phasebit<0>(phasebits, L_ARRIVED_PB_POS + pipeline_id));
                            // update_phasebit<0>(phasebits, L_ARRIVED_PB_POS + pipeline_id);
                            // row_sum = L_smem[pipeline_id][warpgroup::laneid()];
                        }

                        #pragma unroll
                        for (int jj = 0; jj < globals::BLOCK_SIZE / 4 / 2; jj++) {
                            O_reg[jj].x = __fdividef(O_reg[jj].x, row_sum);
                        }

                        // TODO: store async
                        // TODO: add option in TK to disable TMA swizzle
                    }
                    // store async read wait on thread 0
                    warpgroup::tma::cluster::arrive(O_ready[pipeline_id], 0); // Next tile PV can proceed
                    // warpgroup::arrive(M_finished[pipeline_id]);

                    row_sum = __logf(row_sum); // TODO: use log2f
                    row_sum += row_max;
                    row_sum *= NEG_SQRT_D;

                    // L_smem[pipeline_id][warpgroup::laneid()] = row_sum;
                    // warpgroup::sync(5);
                    // if (warpgroup::laneid() == 0) { // still skeptical of TK group vector store
                    //     tma::store_async(G.L, L_smem[pipeline_id],
                    //                     {task_info.batch_idx, task_info.head_idx, 0, 
                    //                     2 * globals::NUM_PIPELINES * task_info.Q_block_idx + 
                    //                     globals::NUM_PIPELINES * cta_id + pipeline_id});
                    //     tma::store_async_read_wait();
                    // }
                    // warpgroup::arrive(L_finished[pipeline_id]);
                    warpgroup::sync(5);
                }

                PV_stage = (PV_stage + 1) % globals::PIPELINE_STAGES;
            }
            group<4>::sync(11);
            if (warpgroup::laneid() == 0 && blockIdx.x == BLOCKIDX) printf("Done epilogue\n");
        }
    } else if (warp_id == 12) { // Loader group
        // warp::decrease_registers<48>();
        constexpr int Q_FINISHED_PB_POS = 0;
        constexpr int K_FINISHED_PB_POS = Q_FINISHED_PB_POS + globals::NUM_PIPELINES;
        constexpr int V_FINISHED_PB_POS = K_FINISHED_PB_POS + globals::PIPELINE_STAGES;

        if (lane_id == 0) {
            for (int task_idx = cluster_id; true; task_idx += gridDim.x / config::CLUSTER_SIZE) {
                globals::task_info task_info = get_task_info(G, task_idx);
                if (task_info.batch_idx == -1) break;

                // Load Q
                #pragma unroll
                for (int pipeline_id = 0; pipeline_id < 2; pipeline_id++) {
                    wait(Q_finished[pipeline_id], get_phasebit<1>(phasebits, Q_FINISHED_PB_POS + pipeline_id));
                    update_phasebit<1>(phasebits, Q_FINISHED_PB_POS + pipeline_id);
                    tma::cluster::expect_bytes(Q_arrived[pipeline_id], sizeof(globals::Q_tile), 0);
                    tma::cluster::load_async(Q_smem[pipeline_id], G.Q,
                                             {task_info.batch_idx, task_info.head_idx, 
                                             config::CLUSTER_SIZE * globals::NUM_PIPELINES * task_info.Q_block_idx + 
                                             globals::NUM_PIPELINES * cta_id + pipeline_id, 0},
                                             Q_arrived[pipeline_id], (uint16_t)(1 << cta_id), 0);
                }

                // Stream K & V
                for (int i = task_info.KV_block_start; i < task_info.KV_block_end + 1; ++i) { // add 1 more iteration to load 2 Ks first
                    if (i < task_info.KV_block_end) {
                        wait(K_finished[QK_stage], get_phasebit<1>(phasebits, K_FINISHED_PB_POS + QK_stage));
                        update_phasebit<1>(phasebits, K_FINISHED_PB_POS + QK_stage);
                        tma::cluster::expect(K_arrived[QK_stage], 0, K_smem[QK_stage]);
                        tma::cluster::load_async(K_smem[QK_stage], G.K, 
                                                 {task_info.batch_idx, task_info.head_idx, 2 * i + cta_id, 0},
                                                 K_arrived[QK_stage], (uint16_t)(1 << cta_id), 0);
                        QK_stage = (QK_stage + 1) % globals::PIPELINE_STAGES;
                    }

                    if (i > task_info.KV_block_start) {
                        wait(V_finished[PV_stage], get_phasebit<1>(phasebits, V_FINISHED_PB_POS + PV_stage));
                        update_phasebit<1>(phasebits, V_FINISHED_PB_POS + PV_stage);
                        tma::cluster::expect(V_arrived[PV_stage], 0, V_smem[PV_stage]);
                        tma::cluster::load_async(V_smem[PV_stage], G.V, 
                                                 {task_info.batch_idx, task_info.head_idx, i, cta_id},
                                                 V_arrived[PV_stage], (uint16_t)(1 << cta_id), 0);
                        PV_stage = (PV_stage + 1) % globals::PIPELINE_STAGES;
                    }
                }
            }

            if (blockIdx.x == BLOCKIDX) printf("Loader group done\n");
        }
    } else if (warp_id == 13) { // MMA launcher group
        // warp::decrease_registers<88>();
        constexpr int Q_ARRIVED_PB_POS = 0;
        constexpr int K_ARRIVED_PB_POS = Q_ARRIVED_PB_POS + globals::NUM_PIPELINES;
        constexpr int V_ARRIVED_PB_POS = K_ARRIVED_PB_POS + globals::PIPELINE_STAGES;
        constexpr int S_FINISHED_PB_POS = V_ARRIVED_PB_POS + globals::PIPELINE_STAGES;
        constexpr int P_ARRIVED_PB_POS = S_FINISHED_PB_POS + 1;
        constexpr int O_READY_PB_POS = P_ARRIVED_PB_POS + globals::NUM_PIPELINES;

        if (lane_id == 0 && cta_id == 0) {
            using tm_fl_t = tt<float, globals::BLOCK_SIZE, globals::BLOCK_SIZE>;
            using tm_bf_t = tt<bf16, globals::BLOCK_SIZE, globals::BLOCK_SIZE>;
            tm_fl_t S_tm = tm_allocator.allocate<tm_fl_t>(0);
            tm_bf_t P_tm[globals::NUM_PIPELINES] = {
                tm_allocator.allocate<tm_bf_t>(globals::BLOCK_SIZE + (globals::BLOCK_SIZE / 2) * 0),
                tm_allocator.allocate<tm_bf_t>(globals::BLOCK_SIZE + (globals::BLOCK_SIZE / 2) * 1)
            };
            tm_fl_t O_tm[globals::NUM_PIPELINES] = {
                tm_allocator.allocate<tm_fl_t>(globals::BLOCK_SIZE * 2),
                tm_allocator.allocate<tm_fl_t>(globals::BLOCK_SIZE * 3)
            };
    
            for (int task_idx = cluster_id; true; task_idx += gridDim.x / config::CLUSTER_SIZE) {
                globals::task_info task_info = get_task_info(G, task_idx);
                if (task_info.batch_idx == -1) break;

                // Wait for Q arrival
                if (task_info.KV_block_start < task_info.KV_block_end) {
                    #pragma unroll
                    for (int pipeline_id = 0; pipeline_id < globals::NUM_PIPELINES; ++pipeline_id) {
                        tma::cluster::wait(Q_arrived[pipeline_id], get_phasebit<0>(phasebits, Q_ARRIVED_PB_POS + pipeline_id));
                        update_phasebit<0>(phasebits, Q_ARRIVED_PB_POS + pipeline_id);
                    }
                }

                for (int i = task_info.KV_block_start; i < task_info.KV_block_end + 1; ++i) {
                    #pragma unroll
                    for (int pipeline_id = 0; pipeline_id < globals::NUM_PIPELINES; ++pipeline_id) {
                        // Launch S = QK^T
                        if (i < task_info.KV_block_end) {
                            if (pipeline_id == 0) {
                                tma::cluster::wait(K_arrived[QK_stage], get_phasebit<0>(phasebits, K_ARRIVED_PB_POS + QK_stage));
                                update_phasebit<0>(phasebits, K_ARRIVED_PB_POS + QK_stage);
                            }
                            tma::cluster::wait(S_finished, get_phasebit<1>(phasebits, S_FINISHED_PB_POS));
                            update_phasebit<1>(phasebits, S_FINISHED_PB_POS); // must be updated for every iteration
                            mm2_ABt(S_tm, Q_smem[pipeline_id], K_smem[QK_stage], S_arrived[pipeline_id]);
                            if (pipeline_id == globals::NUM_PIPELINES - 1) {
                                QK_stage = (QK_stage + 1) % globals::PIPELINE_STAGES;
                            }
                        }

                        // Launch O = PV
                        if (i > task_info.KV_block_start) {
                            if (pipeline_id == 0) {
                                tma::cluster::wait(V_arrived[PV_stage], get_phasebit<0>(phasebits, V_ARRIVED_PB_POS + PV_stage));
                                update_phasebit<0>(phasebits, V_ARRIVED_PB_POS + PV_stage);
                            }
                            tma::cluster::wait(P_arrived[pipeline_id], get_phasebit<0>(phasebits, P_ARRIVED_PB_POS + pipeline_id));
                            update_phasebit<0>(phasebits, P_ARRIVED_PB_POS + pipeline_id);
                            tma::cluster::wait(O_ready[pipeline_id], get_phasebit<1>(phasebits, O_READY_PB_POS + pipeline_id));
                            update_phasebit<1>(phasebits, O_READY_PB_POS + pipeline_id);
                            if (i == task_info.KV_block_start + 1)
                                mm2_AB(O_tm[pipeline_id], P_tm[pipeline_id], V_smem[PV_stage], O_arrived[pipeline_id]);
                            else
                                mma2_AB(O_tm[pipeline_id], P_tm[pipeline_id], V_smem[PV_stage], O_arrived[pipeline_id]);
                            if (pipeline_id == globals::NUM_PIPELINES - 1) {
                                PV_stage = (PV_stage + 1) % globals::PIPELINE_STAGES;
                            }
                        }
                    }
                }
            }

            if (blockIdx.x == BLOCKIDX) printf("MMA group done\n");
        }
    } else if (warp_id == 14 || warp_id == 15) {
        warp::decrease_registers<24>(); // unused warps
    }

    // For TM deallocation
    everyone::tma::cluster::sync();

    if (blockIdx.x < 8 && threadIdx.x == 0) printf("Everyone Done %d!\n", blockIdx.x);
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
}
