/*
Learnings:
    - Must always ablate and find the best SUPER_M (gives extra 4~5% perf boost)

Just loader + 2 consumers
--------------------  M=4096 N=4096 K=4096  --------------------
Block size: 256x256x64
Num tasks: 512
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 49.2208 us
Achieved performance: 2792.29 TFLOPs

Just loader (no consumer)
--------------------  M=4096 N=4096 K=4096  --------------------
Block size: 256x256x64
Num tasks: 512
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 41.7574 us
Achieved performance: 3291.36 TFLOPs

Just loader (no consumer warpgroup)
--------------------  M=4096 N=4096 K=4096  --------------------
Block size: 256x256x64
Num tasks: 512
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 40.9021 us
Achieved performance: 3360.19 TFLOPs

Reduced to loading 128x256x64 each stage
--------------------  M=4096 N=4096 K=4096  --------------------
Block size: 256x256x64
Num tasks: 512
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 59.4224 us
Achieved performance: 2312.91 TFLOPs

Increase SUPER_M from 8 -> 12
--------------------  M=4096 N=4096 K=4096  --------------------
Block size: 256x256x64
Num tasks: 512
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 60.4637 us
Achieved performance: 2273.08 TFLOPs

After some ablation, best SUPER_M was 4
--------------------  M=4096 N=4096 K=4096 SUPER_M=4  --------------------
Block size: 256x256x64
Num tasks: 512
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 56.7722 us
Achieved performance: 2420.89 TFLOPs

Reducing swizzle bytes to 64, instead of 128
--------------------  M=4096 N=4096 K=4096 SUPER_M=4  --------------------
Block size: 256x256x64
Num tasks: 512
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 73.7731 us
Achieved performance: 1862.99 TFLOPs

Reducing swizzle bytes further, to 32
Learning: use the largest swizzle bytes possible
--------------------  M=4096 N=4096 K=4096 SUPER_M=4  --------------------
Block size: 256x256x64
Num tasks: 512
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 143.187 us
Achieved performance: 959.859 TFLOPs

Using swizzle bytes = 64 + at each stage, a cta loads 1/2 of the A tile & multicasts to one other CTA + using 132 SMs + 4-Cluster
--------------------  M=4096 N=4096 K=4096 SUPER_M=4  --------------------
Block size: 256x256x64
Num tasks: 512
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 55.3664 us
Achieved performance: 2482.35 TFLOPs

Using swizzle bytes = 128 + at each stage, one of the two CTAs load A tile & multicasts to one other CTA + using 132 SMs + 4-Cluster

*/

#include "kittens.cuh"

using namespace kittens;

constexpr int NUM_CONSUMERS = 0;
constexpr int NUM_PRODUCERS = 1;
constexpr int NUM_WORKERS = (NUM_CONSUMERS + NUM_PRODUCERS) * 4;
constexpr int NUM_THREADS = NUM_WORKERS * WARP_THREADS;

static constexpr int Mb = 128;
static constexpr int Nb = 256;
static constexpr int Kb = 64;

constexpr int PIPE_DEPTH = 4;

constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1024;

constexpr int CLUSTER_SIZE = 4;

struct matmul_globals {
    using a_tile = st_bf<Mb, Kb/2, true, 64>;
    using b_tile = st_bf<Nb/2, Kb>;

    using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
    using d_gl = gl<bf16, 1, 1, -1, -1>;

    a_gl a;
    b_gl b;
    d_gl d;

    __host__ __inline__ dim3 grid() {
        return dim3(132);
    }
};

template<int SUPER_M> __device__ static inline int2 get_task_idx(const matmul_globals &g, int task_iter) {
    constexpr int CLUSTER_M = 2*Mb, CLUSTER_N = 2*Nb;
    int cluster_x = clusterIdx().x;
    int task_id = task_iter * (gridDim.x/CLUSTER_SIZE) + cluster_x;
    int Rblocks = g.d.rows() / CLUSTER_M, Cblocks = g.d.cols() / CLUSTER_N;
    int super_rows = (Rblocks/SUPER_M)*SUPER_M,
        final_rows = Rblocks - super_rows,
        super_repeat = SUPER_M*Cblocks;
    if (task_id < super_rows * Cblocks) {
        return { (SUPER_M*(task_id/super_repeat) + task_id%SUPER_M), (task_id%super_repeat)/SUPER_M };
    }
    else if (task_id < Rblocks*Cblocks) {
        int remainder_id = task_id - super_rows*Cblocks;
        return { (super_rows + remainder_id%final_rows), remainder_id/final_rows };
    }
    else {
        return { -1, -1 };
    }
}

template<int SUPER_M>
__global__ __cluster_dims__(CLUSTER_SIZE, 1, 1) __launch_bounds__(NUM_THREADS, 1)
void matmul(const __grid_constant__ matmul_globals g) {
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    const int cta_rank = cluster_ctarank();
    const int iters_per_task = g.a.cols() / Kb;

    using a_tile = matmul_globals::a_tile;
    using b_tile = matmul_globals::b_tile;
    
    static_assert(sizeof(a_tile) * PIPE_DEPTH * 2 +
                  sizeof(b_tile) * PIPE_DEPTH <= DYNAMIC_SHARED_MEMORY);
    a_tile (&a_smem)[PIPE_DEPTH][2] = al.allocate<a_tile, PIPE_DEPTH, 2>();
    b_tile (&b_smem)[PIPE_DEPTH]    = al.allocate<b_tile, PIPE_DEPTH>();

    __shared__ semaphore inputs_arrived[PIPE_DEPTH];
    __shared__ semaphore inputs_finished[PIPE_DEPTH];
    uint32_t bitfield = 0xFFFF0000;

    if (threadIdx.x == 0) { 
        #pragma unroll
        for(int i = 0; i < PIPE_DEPTH; i++) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, CLUSTER_SIZE/2);
        }
    }
    everyone::tma::cluster::sync();

    if(warp::laneid() == 0 && warpgroup::warpid() == 3) {
        const int a_idx = (cta_rank&0b10)>>1;
        const uint16_t a_mask = 0b101<<(cta_rank&1);
        const uint16_t b_mask = 1<<cta_rank;
        const int semaphore_cta = cta_rank&(~1);
        int input_ring = 0; // tracking which input block is being loaded
        for(int task_iter = 0; true; task_iter++) {
            int2 rowcol = get_task_idx<SUPER_M>(g, task_iter);
            if(rowcol.x == -1) break;
            for (int idx = 0; idx < iters_per_task; idx++) {
                tma::cluster::wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                update_phasebit<1>(bitfield, input_ring);
                tma::cluster::load_async(a_smem[input_ring][a_idx], g.a, {rowcol.x*2+(cta_rank&1), idx*2+a_idx}, inputs_arrived[input_ring], a_mask, semaphore_cta);
                tma::cluster::load_async(b_smem[input_ring],        g.b, {rowcol.y*4+cta_rank,             idx}, inputs_arrived[input_ring], b_mask, semaphore_cta);
                input_ring=ring_advance<PIPE_DEPTH>(input_ring);
            }
        }
    }
    else if(cta_rank%2 == 0 && warp::laneid() == 0 && warpgroup::warpid() == 0) { // launch the MMA's
        constexpr uint16_t cta_mask = (1<<CLUSTER_SIZE) - 1;
        int input_ring = 0; // tracking which input block is being loaded
        for(int task_iter = 0; true; task_iter++) {
            int2 rowcol = get_task_idx<SUPER_M>(g, task_iter);
            if(rowcol.x == -1) break;
            for(int idx = 0; idx < iters_per_task; idx++) {
                tma::cluster::expect_bytes(inputs_arrived[input_ring], 4*sizeof(a_tile)+2*sizeof(b_tile));
                tma::cluster::wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
                update_phasebit<0>(bitfield, input_ring);
                detail::tcgen05::commit<2>(inputs_finished[input_ring], cta_mask);
                input_ring=ring_advance<PIPE_DEPTH>(input_ring);
            }
        }
    }
    everyone::tma::cluster::sync();
}


#include <iostream>
#include <random>

constexpr int NCU = false;

template<int SUPER_M>
void inner_run(bf16 *d_A, bf16 *d_B, bf16 *d_C, size_t M, size_t N, size_t K) {
    using globals  = matmul_globals;
    typename globals::a_gl Ag{d_A, nullptr, nullptr, M, K};
    typename globals::b_gl Bg{d_B, nullptr, nullptr, N, K};
    typename globals::d_gl Dg{d_C, nullptr, nullptr, M, N};
    globals G{Ag, Bg, Dg};
    matmul<SUPER_M><<<G.grid(), NUM_THREADS, DYNAMIC_SHARED_MEMORY>>>(G);
}

template<int SUPER_M=8>
int run_benchmark(size_t M, size_t N, size_t K) {
    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << " SUPER_M=" << SUPER_M << "  --------------------\n";
    std::cout << "Block size: " << Mb*2 << "x" << Nb << "x" << Kb << "\n";
    std::cout << "Num tasks: " << (M/Mb*N/Nb) << "\n";

    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    std::cout << "Allocated host memory" << std::endl;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    // Initialize matrices with random values
    for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen);
    for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen);
    std::cout << "Initialized matrices" << std::endl;

    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B, *d_C;
    CUDACHECK(cudaMalloc(&d_A, M*K*sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&d_B, K*N*sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&d_C, M*N*sizeof(__nv_bfloat16)));
    std::cout << "Allocated device memory" << std::endl;

    // Convert to __nv_bfloat16 and copy to device. Otherwise GPU does not truly perform a matmul
    __nv_bfloat16 *h_A_bf16 = new __nv_bfloat16[M * K];
    __nv_bfloat16 *h_B_bf16 = new __nv_bfloat16[K * N];
    for (int i = 0; i < M * K; ++i) h_A_bf16[i] = __float2bfloat16(h_A[i]);
    for (int i = 0; i < K * N; ++i) h_B_bf16[i] = __float2bfloat16(h_B[i]);
    CUDACHECK(cudaMemcpy(d_A, h_A_bf16, M*K*2, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_B, h_B_bf16, K*N*2, cudaMemcpyHostToDevice));
    std::cout << "Copied matrices to device" << std::endl;

    // Set kernel attributes
    cudaFuncSetAttribute(matmul<SUPER_M>, cudaFuncAttributeMaxDynamicSharedMemorySize, DYNAMIC_SHARED_MEMORY);

    // Warmup
    for(int i = 0; i < (NCU ? 0 : 500); i++)
        inner_run<SUPER_M>(d_A, d_B, d_C, M, N, K);

    // Benchmark
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaEventRecord(start));
    constexpr int ITERS = (NCU ? 1 : 100);
    for(int i = 0; i < ITERS; i++)
        inner_run<SUPER_M>(d_A, d_B, d_C, M, N, K);
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    // Calculate duration and TFLOPs
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double useconds = milliseconds * 1000.0 / ITERS;
    double flops = double(2.0) * M * N * K; // 2 FLOPs per multiply-add
    double tflops = (flops / useconds) / 1e6;
    std::cout << "Avg Kernel execution time: " << useconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_A_bf16;
    delete[] h_B_bf16;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

int main() {
    int N;
    if (NCU) {
        N = 4096;
        run_benchmark(N, N, N);
    } else {
        // N = 1024;
        // run_benchmark(N, N, N);
        // N = 2048;
        // run_benchmark(N, N, N);
        N = 4096;
        run_benchmark<4>(N, N, N);
        // N = 8192;
        // run_benchmark(N, N, N);
        // N = 16384;
        // run_benchmark(N, N, N);
    }
    return 0;
}
