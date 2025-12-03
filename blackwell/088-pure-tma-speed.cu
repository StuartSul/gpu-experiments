/*
Learnings:
    - Must always ablate and find the best SUPER_M (gives extra 4~5% perf boost)
    - Higher swizzle size = higher throughput
    - TMA multicast benefits only if splitting (by K) & loading together

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
--------------------  M=4096 N=4096 K=4096 SUPER_M=4  --------------------
Block size: 256x256x64
Num tasks: 512
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 57.4179 us
Achieved performance: 2393.66 TFLOPs

Using swizzle bytes = 128 + only one of the two CTAs load A tile & multicasts to one other CTA + using 132 SMs + 4-Cluster
Learning: no benefit in alternating the load. Major benefit in loading together.
--------------------  M=4096 N=4096 K=4096 SUPER_M=4  --------------------
Block size: 256x256x64
Num tasks: 512
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 57.064 us
Achieved performance: 2408.51 TFLOPs

Back to 256x256x64 per block, but with cluster size = 4. Ablated to find the best SUPER_M
--------------------  M=4096 N=4096 K=4096 SUPER_M=2  --------------------
Block size: 256x256x64
Num tasks: 512
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 37.0432 us
Achieved performance: 3710.23 TFLOPs

^ but using TMA multicast
--------------------  M=4096 N=4096 K=4096 SUPER_M=2  --------------------
Block size: 256x256x64
Num tasks: 512
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 40.2656 us
Achieved performance: 3413.31 TFLOPs

Changed to pure TMA load kernel. Single kernel, 2-cluster, no multicasting, 256x64 per iter, 132-block
Every CTA-pair loads the same thing (intentionally redundant)
--------------------  M=4096 N=4096  --------------------
Block stasksize: 128x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 8.25216 us
Achieved performance: 4066.14 GB/s
--------------------  M=16384 N=16384  --------------------
Block stasksize: 128x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 78.7738 us
Achieved performance: 6815.35 GB/s
--------------------  M=32768 N=32768  --------------------
Block stasksize: 128x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 306.4 us
Achieved performance: 7008.77 GB/s

Now, ^ + multicast to remove redundancy
Learning: barely any difference in utilizing multicast. L2 is fast enough. 
--------------------  M=4096 N=4096  --------------------
Block stasksize: 128x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 8.23232 us
Achieved performance: 4075.94 GB/s
--------------------  M=16384 N=16384  --------------------
Block stasksize: 128x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 78.7181 us
Achieved performance: 6820.17 GB/s
--------------------  M=32768 N=32768  --------------------
Block stasksize: 128x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 305.519 us
Achieved performance: 7028.96 GB/s
*/

#include "kittens.cuh"

using namespace kittens;

constexpr int NUM_THREADS = WARPGROUP_WARPS * WARP_THREADS;

static constexpr int Mb = 128;
static constexpr int Nb = 64;

constexpr int PIPE_DEPTH = 6;

constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1024;

constexpr int CLUSTER_SIZE = 2;

struct matmul_globals {
    using a_tile = st_bf<Mb, Nb>;
    using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
    a_gl a;

    __host__ __inline__ dim3 grid() {
        return dim3(132);
    }
};

__global__ __cluster_dims__(CLUSTER_SIZE, 1, 1) __launch_bounds__(NUM_THREADS, 1)
void matmul(const __grid_constant__ matmul_globals g) {
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    const int cta_rank = cluster_ctarank();
    const int Rblocks = g.a.rows() / (2*Mb);
    const int Cblocks = g.a.cols() / Nb;
    const int num_tasks = Rblocks * Cblocks;

    using a_tile = matmul_globals::a_tile;
    a_tile (&a_smem)[PIPE_DEPTH][2] = al.allocate<a_tile, PIPE_DEPTH, 2>();

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
        const int a_idx = cta_rank;
        const uint16_t a_mask = 0b11;
        // const uint16_t a_mask = 1<<cta_rank;
        const int semaphore_cta = cta_rank&(~1);
        int input_ring = 0;
        for (int idx = blockIdx.x/2; idx < num_tasks; idx+=gridDim.x/2) {
            const int r = idx / Cblocks;
            const int c = idx % Cblocks;
            tma::cluster::wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
            update_phasebit<1>(bitfield, input_ring);
            tma::cluster::load_async(a_smem[input_ring][a_idx], g.a, {r*2+a_idx, c}, inputs_arrived[input_ring], a_mask, semaphore_cta);
            input_ring=ring_advance<PIPE_DEPTH>(input_ring);
        }
    }
    else if(cta_rank%2 == 0 && warp::laneid() == 0 && warpgroup::warpid() == 0) {
        constexpr uint16_t cta_mask = (1<<CLUSTER_SIZE) - 1;
        int input_ring = 0; // tracking which input block is being loaded
        for (int idx = blockIdx.x/2; idx < num_tasks; idx+=gridDim.x/2) {
            tma::cluster::expect_bytes(inputs_arrived[input_ring], 4*sizeof(a_tile));
            tma::cluster::wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
            update_phasebit<0>(bitfield, input_ring);
            detail::tcgen05::commit<2>(inputs_finished[input_ring], cta_mask);
            input_ring=ring_advance<PIPE_DEPTH>(input_ring);
        }
    }
    everyone::tma::cluster::sync();
}


#include <iostream>
#include <random>

void inner_run(bf16 *d_A, size_t M, size_t N) {
    using globals  = matmul_globals;
    typename globals::a_gl Ag{d_A, nullptr, nullptr, M, N};
    globals G{Ag};
    matmul<<<G.grid(), NUM_THREADS, DYNAMIC_SHARED_MEMORY>>>(G);
}

int run_benchmark(size_t M, size_t N) {
    std::cout << "--------------------  M=" << M << " N=" << N << "  --------------------\n";
    std::cout << "Block stasksize: " << Mb << "x" << Nb << "\n";

    // Allocate host memory
    float *h_A = new float[M * N];
    std::cout << "Allocated host memory" << std::endl;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    // Initialize matrices with random values
    for (int i = 0; i < M * N; ++i) h_A[i] = dis(gen);
    std::cout << "Initialized matrices" << std::endl;

    // Allocate device memory
    __nv_bfloat16 *d_A;
    CUDACHECK(cudaMalloc(&d_A, M*N*sizeof(__nv_bfloat16)));
    std::cout << "Allocated device memory" << std::endl;

    // Convert to __nv_bfloat16 and copy to device. Otherwise GPU does not truly perform a matmul
    __nv_bfloat16 *h_A_bf16 = new __nv_bfloat16[M * N];
    for (int i = 0; i < M * N; ++i) h_A_bf16[i] = __float2bfloat16(h_A[i]);
    CUDACHECK(cudaMemcpy(d_A, h_A_bf16, M*N*2, cudaMemcpyHostToDevice));
    std::cout << "Copied matrices to device" << std::endl;

    // Set kernel attributes
    cudaFuncSetAttribute(matmul, cudaFuncAttributeMaxDynamicSharedMemorySize, DYNAMIC_SHARED_MEMORY);

    // Warmup
    for(int i = 0; i < 500; i++)
        inner_run(d_A, M, N);

    // Benchmark
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaEventRecord(start));
    constexpr int ITERS = 100;
    for(int i = 0; i < ITERS; i++)
        inner_run(d_A, M, N);
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    // Calculate duration and gbps
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double useconds = milliseconds * 1000.0 / ITERS;
    double gbps = (double(2.0) * M * N / useconds) / 1e3;
    std::cout << "Avg Kernel execution time: " << useconds << " us\n";
    std::cout << "Achieved performance: " << gbps << " GB/s\n";

    // Clean up
    delete[] h_A;
    delete[] h_A_bf16;
    cudaFree(d_A);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

int main() {
    int N;
    // N = 1024;
    // run_benchmark(N, N);
    // N = 2048;
    // run_benchmark(N, N);
    N = 4096;
    run_benchmark(N, N);
    // N = 8192;
    // run_benchmark(N, N);
    N = 16384;
    run_benchmark(N, N);
    N = 32768;
    run_benchmark(N, N);
    return 0;
}
