/*
    B200 Results:
        ==================================
        CHIPLET=0, N=4096, REPEAT=128
        Size: 32 MB
        Time: 241.99 us
        Throughput: 17748.53 GB/s
        ==================================
        CHIPLET=1, N=4096, REPEAT=128
        Size: 32 MB
        Time: 258.41 us
        Throughput: 16620.81 GB/s
    
    WHAT??
*/

#include <random>
#include <vector>

#include "kittens.cuh"

using namespace kittens;

template <int _CHIPLET_ID, int _N, int _REPEAT, int _TILE_N=128>
struct globals {
    static constexpr int CHIPLET_ID = _CHIPLET_ID;
    static constexpr int N = _N;
    static constexpr int REPEAT = _REPEAT;
    static constexpr int TILE_N = _TILE_N;

    using tile = st_bf<TILE_N, TILE_N>;
    using a_gl = gl<bf16, 1, 1, N, N, tile>;
    using b_gl = gl<bf16, 1, 1, N, N, tile>;

    a_gl a;
    b_gl b;

    __host__ dim3 grid() { return dim3(148); }
    __host__ dim3 block() { return dim3(1); }
    __host__ __device__ constexpr int dynamic_shared_memory() const { 
        return MAX_SHARED_MEMORY - 1024; 
    }
};

template <typename globals_t>
__global__ void kernel(const __grid_constant__ globals_t g) {
    constexpr int PIPE_DEPTH = g.dynamic_shared_memory() / sizeof(globals_t::tile);
    constexpr int CHIPLET_SMS = 74;

    int smid;
    asm volatile("{mov.u32 %0, %smid;}" : "=r"(smid));

    if ((globals_t::CHIPLET_ID == 0 && smid >= CHIPLET_SMS) ||
        (globals_t::CHIPLET_ID == 1 && smid < CHIPLET_SMS))
        return;

    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    auto (&tile)[PIPE_DEPTH] = al.allocate<typename globals_t::tile, PIPE_DEPTH>();

    __shared__ semaphore inputs_arrived[PIPE_DEPTH];
    #pragma unroll
    for (int i = 0; i < PIPE_DEPTH; i++)
        init_semaphore(inputs_arrived[i], 0, 1);

    constexpr int rblks = globals_t::N / globals_t::TILE_N;
    constexpr int cblks = globals_t::N / globals_t::TILE_N;
    constexpr int nblks = rblks * cblks;

    for (int i = 0; true; i++) {
        int task_idx = (smid - globals_t::CHIPLET_ID * CHIPLET_SMS) + (CHIPLET_SMS) * i;
        if (task_idx >= nblks * globals_t::REPEAT) break;
        task_idx %= nblks;

        const int row = task_idx / cblks;
        const int col = task_idx % cblks;
        const int stage = i % PIPE_DEPTH;
        const int phasebit = ((i + PIPE_DEPTH) / PIPE_DEPTH) % 2;

        wait(inputs_arrived[stage], phasebit);
        tma::load_async(tile[stage], g.a, {row, col}, inputs_arrived[stage]);
        tma::expect_bytes(inputs_arrived[stage], sizeof(globals_t::tile));
    }
}

template <typename globals_t>
void run() {
    constexpr int CHIPLET_ID = globals_t::CHIPLET_ID;
    constexpr int N = globals_t::N;
    constexpr int REPEAT = globals_t::REPEAT;
    constexpr int SIZE = N * N * sizeof(bf16);

    printf("==================================\n");
    printf("CHIPLET=%d, N=%d, REPEAT=%d\n", CHIPLET_ID, N, REPEAT);
    printf("Size: %.0f MB\n", SIZE / 1024. / 1024.);

    // Sleep for 50 ms to limit power consumption and thermals
    usleep(50000);

    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    std::vector<bf16> h_A(N * N);
    for (int i = 0; i < N * N; ++i) h_A[i] = __float2bfloat16(dis(gen));

    bf16 *d_A, *d_B;
    CUDACHECK(cudaMalloc(&d_A, SIZE));
    CUDACHECK(cudaMalloc(&d_B, SIZE));
    CUDACHECK(cudaMemcpy(d_A, h_A.data(), SIZE, cudaMemcpyHostToDevice));

    typename globals_t::a_gl a_gl{d_A, nullptr, nullptr, nullptr, nullptr};
    typename globals_t::b_gl b_gl{d_B, nullptr, nullptr, nullptr, nullptr};
    globals_t g{a_gl, b_gl};

    constexpr int num_warmups = 500;
    constexpr int num_iters = 100;
    cudaFuncSetAttribute(kernel<globals_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory());

    for(int i = 0; i < num_warmups; i++)
        kernel<globals_t><<<g.grid(), g.block(), g.dynamic_shared_memory()>>>(g);

    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaEventRecord(start));
    for(int i = 0; i < num_iters; i++)
        kernel<globals_t><<<g.grid(), g.block(), g.dynamic_shared_memory()>>>(g);
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double avg_time = double(milliseconds) / 1000.0 / num_iters;
    double bps = double(REPEAT) * SIZE / avg_time;
    printf("Time: %.2f us\n", avg_time * 1e6);
    printf("Throughput: %.2f GB/s\n", bps / 1e9);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int device_id;
    CUDACHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDACHECK(cudaGetDeviceProperties(&prop, device_id));
    printf("L2 Size=%.2f MB\n", (double)prop.l2CacheSize / 1024 / 1024);

    run<globals<0, 4096, 128, 128>>();
    run<globals<1, 4096, 128, 128>>();
}
