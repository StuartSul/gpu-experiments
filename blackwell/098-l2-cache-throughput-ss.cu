/*
Per-SM throughput measurement
*/

#include <random>
#include <vector>

#include "kittens.cuh"

using namespace kittens;

template <int _SM_ID, int _N, int _REPEAT, int _TILE_N, bool _VERIFY>
struct globals {
    static constexpr int SM_ID = _SM_ID;
    static constexpr int N = _N;
    static constexpr int REPEAT = _REPEAT;
    static constexpr int TILE_N = _TILE_N;
    static constexpr bool VERIFY = _VERIFY;

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

    int sm_id;
    asm volatile("{mov.u32 %0, %smid;}" : "=r"(sm_id));
    if (sm_id != globals_t::SM_ID) return;

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
    constexpr int num_iters = globals_t::VERIFY ? nblks : nblks * globals_t::REPEAT;

    for (int i = 0; i < num_iters; i++) {
        const int task_idx = i % nblks;
        const int row = task_idx / cblks;
        const int col = task_idx % cblks;
        const int stage = i % PIPE_DEPTH;
        const int phasebit = ((i + PIPE_DEPTH) / PIPE_DEPTH) % 2;

        wait(inputs_arrived[stage], phasebit);
        tma::load_async(tile[stage], g.a, {row, col}, inputs_arrived[stage]);
        tma::expect_bytes(inputs_arrived[stage], sizeof(globals_t::tile));

        if constexpr (globals_t::VERIFY) {
            wait(inputs_arrived[stage], (i / PIPE_DEPTH) % 2);
            tma::store_async(g.b, tile[stage], {row, col});
            tma::store_async_wait();
        }
    }
}

template <typename globals_t>
__host__ void run() {
    constexpr int SM_ID = globals_t::SM_ID;
    constexpr int N = globals_t::N;
    constexpr int REPEAT = globals_t::REPEAT;
    constexpr bool VERIFY = globals_t::VERIFY;
    constexpr int SIZE = N * N * sizeof(bf16);

    printf("==================================\n");
    printf("SM_ID=%d, N=%d, REPEAT=%d\n", SM_ID, N, REPEAT);
    printf("Size: %.0f MB\n", SIZE / 1024. / 1024.);

    // Sleep for 100 ms to limit power consumption and thermals
    usleep(100000);

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

    constexpr int num_warmups = VERIFY ? 0 : 500;
    constexpr int num_iters = VERIFY ? 1 : 100;
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

    if (!VERIFY) {
        float milliseconds;
        cudaEventElapsedTime(&milliseconds, start, stop);
        double avg_time = double(milliseconds) / 1000.0 / num_iters;
        double bps = double(REPEAT) * SIZE / avg_time;
        printf("Time: %.2f us\n", avg_time * 1e6);
        printf("Throughput: %.2f GB/s\n", bps / 1e9);
    } else {
        std::vector<bf16> h_B(N * N);
        CUDACHECK(cudaMemcpy(h_B.data(), d_B, SIZE, cudaMemcpyDeviceToHost));
        int error_count = 0;
        for (int i = 0; i < N * N; ++i) {
            if (h_A[i] != h_B[i]) error_count += 1;
        }
        printf("Error count: %d\n", error_count);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

template <int SM_ID, int N, int REPEAT, int TILE_N, bool VERIFY>
__host__ void sm_id_loop() {
    run<globals<SM_ID, N, REPEAT, TILE_N, VERIFY>>();
    if constexpr (SM_ID > 0)
        sm_id_loop<SM_ID - 1, N, REPEAT, TILE_N, VERIFY>();
}

__host__ int main() {
    int device_id;
    CUDACHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDACHECK(cudaGetDeviceProperties(&prop, device_id));
    printf("L2 Size=%.2f MB\n", (double)prop.l2CacheSize / 1024 / 1024);

    sm_id_loop<147, 4096, 16, 128, false>();
}
