/*
    Practice using cluster launch control (CLC) and work stealing.
    Example kernel performs simple vector-scalar multiplication
*/

#include "kittens.cuh"

using namespace kittens;

__global__ void init(float *data, const int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        unsigned long long clock_val = clock64();
        unsigned int seed = (unsigned int)(clock_val ^ (idx << 16) ^ (i << 8));
        seed = seed * 1103515245 + 12345;
        data[i] = static_cast<float>(seed & 0x7FFFFFFF) / static_cast<float>(0x7FFFFFFF);
    }
}

__global__ void kernel (float* data, const float alpha, const int N) {
    __shared__ uint4 clc_handle;
    __shared__ semaphore clc_arrived;
    uint32_t phasebit = 0xFFFF0000;

    if (threadIdx.x == 0)
        init_semaphore(clc_arrived, 0, 1);
    __syncthreads();

    uint32_t curr_block_idx = blockIdx.x;

    while (true) {
        if (threadIdx.x == 0) {
            // Prevent reordering w.r.t. CLC result write
            asm volatile("{fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster;}" ::: "memory");
            
            // Request cancellation of a cluster not launched yet
            // Note: requesting subsequent clusterlaunchcontrol.try_cancel after a failure is undefined
            asm volatile("{clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];}"
                :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&clc_handle))), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&clc_arrived)))
                : "memory"
            );
            tma::expect_bytes(clc_arrived, sizeof(clc_handle)); // TODO change to cluster scope
        }

        // Do the work
        int i = curr_block_idx * blockDim.x + threadIdx.x;
        if (i < N)
            data[i] *= alpha;

        wait(clc_arrived, get_phasebit<0>(phasebit, 0));
        update_phasebit<0>(phasebit, 0);

        uint32_t success;
        int3 next_cta_id;
        asm volatile(
            "{\n"
            ".reg .pred SUCCESS;\n"
            ".reg .b128 CLC_HANDLE;\n"
            ".reg .b32 IGNORE;\n"
            "ld.shared.b128 CLC_HANDLE, [%4];\n"
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 SUCCESS, CLC_HANDLE;\n"
            "selp.u32 %0, 1, 0, SUCCESS;\n"
            "@!SUCCESS bra.uni DONE;\n"
            "clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%1, %2, %3, IGNORE}, CLC_HANDLE;\n"
            "fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;\n" // Release read of result to the async proxy:
            "DONE:\n"
            "}"
            : "=r"(success), "=r"(next_cta_id.x), "=r"(next_cta_id.y), "=r"(next_cta_id.z)
            : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&clc_handle)))
            : "memory"
        );
        if (!success)
            break;

        // Prepare for next iteration
        curr_block_idx = next_cta_id.x;

        // Prevent next cancellation request from overwriting
        __syncthreads(); 
    }
}

int main() {
    constexpr int N = 1024 * 1024 * 1024;
    constexpr float alpha = 3.14f;

    float *data_device;
    CUDACHECK(cudaMalloc(&data_device, N * sizeof(float)));
    init<<<1024, 1024>>>(data_device, N);
    CUDACHECK(cudaDeviceSynchronize());
    
    float *data_host = (float*)malloc(N * sizeof(float));
    CUDACHECK(cudaMemcpy(data_host, data_device, N * sizeof(float), cudaMemcpyDeviceToHost));

    kernel<<<N / 256, 256>>>(data_device, alpha, N);
    CUDACHECK(cudaDeviceSynchronize());

    float *results_host = (float*)malloc(N * sizeof(float));
    CUDACHECK(cudaMemcpy(results_host, data_device, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    int error_count = 0;
    for (int i = 0; i < N; ++i) {
        if (fabsf(results_host[i] - data_host[i] * alpha) > 1e-12f)
            error_count++;
    }
    printf("Error count: %d\n", error_count);

    free(data_host);
    CUDACHECK(cudaFree(data_device));

    return 0;
}
