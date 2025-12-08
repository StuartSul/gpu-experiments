/*
This didn't really work out :(
*/

#include <random>
#include <vector>

#include "kittens.cuh"

using namespace kittens;

template <int _SM_ID, int _N, int _REPEAT, int _TILE_N>
struct globals {
    static constexpr int SM_ID = _SM_ID;
    static constexpr int N = _N;
    static constexpr int REPEAT = _REPEAT;
    static constexpr int TILE_N = _TILE_N;

    using tile = st_bf<TILE_N, TILE_N>;
    using a_gl = gl<bf16, 1, 1, N, N, tile>;

    a_gl a;

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
    constexpr int num_iters = nblks * globals_t::REPEAT;

    for (int i = 0; i < num_iters; i++) {
        const int task_idx = i % nblks;
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
__host__ void inner_run(size_t base_addr) {
    constexpr int SM_ID = globals_t::SM_ID;
    constexpr int N = globals_t::N;
    constexpr int REPEAT = globals_t::REPEAT;
    constexpr int SIZE = N * N * sizeof(bf16);

    printf("SM_ID=%03d, N=%d, REPEAT=%d, Size: %.3f MB, ", SM_ID, N, REPEAT, SIZE / 1024. / 1024.);

    // Sleep for 100 ms to limit power consumption and thermals
    usleep(100000);

    bf16 *d_A = reinterpret_cast<bf16 *>(base_addr);
    typename globals_t::a_gl a_gl{d_A, nullptr, nullptr, nullptr, nullptr};
    globals_t g{a_gl};

    cudaFuncSetAttribute(kernel<globals_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory());

    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaEventRecord(start));
    kernel<globals_t><<<g.grid(), g.block(), g.dynamic_shared_memory()>>>(g);
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double avg_time = double(milliseconds) / 1000.0;
    double bps = double(REPEAT) * SIZE / avg_time;
    printf("Time: %.2f us, Throughput: %.2f GB/s\n", avg_time * 1e6, bps / 1e9);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

template <int SM_ID_0, int SM_ID_1, int N, int REPEAT, int TILE_N>
__host__ void outer_run() {
    size_t base_addr;
    CUDACHECK(cudaMalloc(reinterpret_cast<void **>(&base_addr), N * N * sizeof(bf16)));
    printf("Allocated address: %p\n", reinterpret_cast<void *>(base_addr));

    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    std::vector<bf16> h_A(N * N);
    for (int i = 0; i < N * N; ++i) h_A[i] = __float2bfloat16(dis(gen));
    CUDACHECK(cudaMemcpy(reinterpret_cast<void *>(base_addr), h_A.data(), N * N, cudaMemcpyHostToDevice));

    inner_run<globals<SM_ID_0, N, REPEAT, TILE_N>>(base_addr);
    inner_run<globals<SM_ID_1, N, REPEAT, TILE_N>>(base_addr);

    CUDACHECK(cudaFree(reinterpret_cast<void *>(base_addr)));
}

template <int SM_ID_0, int SM_ID_1, int N, int REPEAT, int TILE_N>
__host__ void inner_loop() {
    outer_run<SM_ID_0, SM_ID_1, N, REPEAT, TILE_N>();
    if constexpr (SM_ID_1 > 0)
        inner_loop<SM_ID_0, SM_ID_1 - 1, N, REPEAT, TILE_N>();
}

template <int SM_ID, int N, int REPEAT, int TILE_N>
__host__ void outer_loop() {
    inner_loop<SM_ID, SM_ID - 1, N, REPEAT, TILE_N>();
    if constexpr (SM_ID > 1)
        outer_loop<SM_ID - 1, N, REPEAT, TILE_N>();
}

__host__ int main() {
    outer_loop<147, 2048, 1, 128>();
}
