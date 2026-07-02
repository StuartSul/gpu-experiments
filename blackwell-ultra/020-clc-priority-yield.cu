/*

Observation:
- CLC just fails everything until the higher priority unscheduled blocks are gone
- It doesn't fail "just the right number of blocks". It fails everything while there is anything higher priority unscheduled
- Once everything from higher priority is scheduled, CLC succeeds (whether those higher priority blocks are done doesn't matter)

How to run (sort -n to sort by time):
    make
    ./020-clc-priority-yield.out | sort -n
*/

#include <chrono>
#include <thread>

#include "kittens.cuh"

using namespace kittens;

constexpr unsigned int LP_WORK_MS  = 3000;
constexpr unsigned int HP_WORK_MS  = 10000;
constexpr unsigned int HP_DELAY_MS = 100;
constexpr int          HP_BLOCKS   = 24;

__device__ static inline unsigned long long globaltimer_ns() {
    unsigned long long t;
    asm volatile("{mov.u64 %0, %%globaltimer;}" : "=l"(t));
    return t;
}

__device__ static inline double now_ms() {
    return (double)globaltimer_ns() / 1e6;
}

__device__ static inline void sleep_ms(unsigned int dur) {
    const unsigned long long end = globaltimer_ns() + (unsigned long long)dur * 1000000ull;
    while (globaltimer_ns() < end) __nanosleep(50000);
}

__global__ __cluster_dims__(1, 1, 1) void lp_clc_kernel() {
    __shared__ clc::handle handle;
    __shared__ semaphore sem;
    init_semaphore(sem, 0, 1);
    asm volatile("{fence.mbarrier_init.release.cluster;}" ::: "memory");

    int ctaid = blockIdx.x;
    bool stolen = false;
    int phase = 0;
    while (true) {
        printf("%12.1f ms | SM %3d | LP ctaid %3d starts (%s)\n", now_ms(), smid(), ctaid, stolen ? "STOLEN via CLC" : "fresh launch");
        sleep_ms(LP_WORK_MS);                 // "compute the tile"
        tma::expect_bytes(sem, sizeof(handle));
        clc::schedule(handle, sem);           // clusterlaunchcontrol.try_cancel
        wait(sem, phase);
        phase ^= 1;
        clc::result r = clc::query(handle);   // clusterlaunchcontrol.query_cancel
        if (!r.success) break;                // "end-of-schedule": block ends, SM freed
        ctaid = r.x;
        stolen = true;
    }
    printf("%12.1f ms | SM %3d | LP try_cancel FAILED -> block exits, SM freed\n", now_ms(), smid());
}

__global__ void hp_fixed_kernel() {
    printf("%12.1f ms | SM %3d | HP block %2d STARTS\n", now_ms(), smid(), blockIdx.x);
    sleep_ms(HP_WORK_MS);
    printf("%12.1f ms | SM %3d | HP block %2d ENDS\n", now_ms(), smid(), blockIdx.x);
}

int main() {
    cudaDeviceProp prop;
    CUDACHECK(cudaGetDeviceProperties(&prop, 0));
    int least_priority = 0;
    int greatest_priority = 0;
    CUDACHECK(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
    const int lp_blocks = 4 * prop.multiProcessorCount;
    printf("GPU: %s, %d SMs | LP grid=%d on prio %d | HP grid=%d on prio %d\n",
           prop.name, prop.multiProcessorCount, lp_blocks, least_priority, HP_BLOCKS, greatest_priority);

    cudaStream_t stream_low;
    cudaStream_t stream_high;
    CUDACHECK(cudaStreamCreateWithPriority(&stream_low, cudaStreamNonBlocking, least_priority));
    CUDACHECK(cudaStreamCreateWithPriority(&stream_high, cudaStreamNonBlocking, greatest_priority));

    CUDACHECK(cudaFuncSetAttribute(lp_clc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY - 1024));
    CUDACHECK(cudaFuncSetAttribute(hp_fixed_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY - 1024));

    lp_clc_kernel<<<lp_blocks, 1, MAX_SHARED_MEMORY - 1024, stream_low>>>();
    std::this_thread::sleep_for(std::chrono::milliseconds(HP_DELAY_MS));
    hp_fixed_kernel<<<HP_BLOCKS, 1, MAX_SHARED_MEMORY - 1024, stream_high>>>();
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaGetLastError());
    return 0;
}
