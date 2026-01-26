/*
    Observations:
    - With small grid size, cudaLaunchKernelEx is faster than <<<>>>
    - With small grid size, adding PGL attribute on top of ^ is faster
    - With small grid size, adding trigger/sync on top of ^ is faster
    - With large grid size, cudaLaunchKernelEx is as fast as <<<>>>
    - With large grid size, adding PGL attribute on top of ^ is faster
    - With large grid size, adding trigger/sync on top of ^ is slower than <<<>>> (by 1.7x)
*/

#include "kittens.cuh"

constexpr int ITERATIONS = 100;
constexpr int GRID_SIZE = 128;  // Should try 1, 1024, 4096, 16384, 65536

__global__ void kernel_empty() {}

__global__ void kernel_trigger() {
    cudaTriggerProgrammaticLaunchCompletion();
}

__global__ void kernel_sync() {
    cudaGridDependencySynchronize();
}

__global__ void kernel_both() {
    cudaTriggerProgrammaticLaunchCompletion();
    cudaGridDependencySynchronize();
}

int main() {
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    // PDL attribute config
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config_pdl = {0};
    config_pdl.gridDim = dim3(GRID_SIZE);
    config_pdl.blockDim = dim3(1);
    config_pdl.stream = 0;
    config_pdl.attrs = attribute;
    config_pdl.numAttrs = 1;

    // No attribute config
    cudaLaunchConfig_t config_no_attr = {0};
    config_no_attr.gridDim = dim3(GRID_SIZE);
    config_no_attr.blockDim = dim3(1);
    config_no_attr.stream = 0;
    config_no_attr.attrs = nullptr;
    config_no_attr.numAttrs = 0;

    // Warmup
    for (int i = 0; i < 10; i++) {
        kernel_empty<<<GRID_SIZE, 1>>>();
        CUDACHECK(cudaLaunchKernelEx(&config_no_attr, kernel_empty));
        CUDACHECK(cudaLaunchKernelEx(&config_pdl, kernel_empty));
        CUDACHECK(cudaLaunchKernelEx(&config_pdl, kernel_trigger));
        CUDACHECK(cudaLaunchKernelEx(&config_pdl, kernel_sync));
        CUDACHECK(cudaLaunchKernelEx(&config_pdl, kernel_both));
    }
    CUDACHECK(cudaDeviceSynchronize());

    printf("Grid size: %d blocks\n\n", GRID_SIZE);

    // (1) Baseline: empty kernel with <<<>>>
    CUDACHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERATIONS; i++) {
        kernel_empty<<<GRID_SIZE, 1>>>();
    }
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));
    float baseline_ms;
    CUDACHECK(cudaEventElapsedTime(&baseline_ms, start, stop));

    // (2) cudaLaunchKernelEx (no PDL attr)
    CUDACHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERATIONS; i++) {
        CUDACHECK(cudaLaunchKernelEx(&config_no_attr, kernel_empty));
    }
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));
    float launch_ex_ms;
    CUDACHECK(cudaEventElapsedTime(&launch_ex_ms, start, stop));

    // (3) cudaLaunchKernelEx with PDL attr (empty kernel)
    CUDACHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERATIONS; i++) {
        CUDACHECK(cudaLaunchKernelEx(&config_pdl, kernel_empty));
    }
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));
    float pdl_attr_ms;
    CUDACHECK(cudaEventElapsedTime(&pdl_attr_ms, start, stop));

    // (4) PDL attr + cudaTriggerProgrammaticLaunchCompletion
    CUDACHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERATIONS; i++) {
        CUDACHECK(cudaLaunchKernelEx(&config_pdl, kernel_trigger));
    }
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));
    float trigger_ms;
    CUDACHECK(cudaEventElapsedTime(&trigger_ms, start, stop));

    // (5) PDL attr + cudaGridDependencySynchronize
    CUDACHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERATIONS; i++) {
        CUDACHECK(cudaLaunchKernelEx(&config_pdl, kernel_sync));
    }
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));
    float sync_ms;
    CUDACHECK(cudaEventElapsedTime(&sync_ms, start, stop));

    // (6) PDL attr + both trigger and sync
    CUDACHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERATIONS; i++) {
        CUDACHECK(cudaLaunchKernelEx(&config_pdl, kernel_both));
    }
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));
    float both_ms;
    CUDACHECK(cudaEventElapsedTime(&both_ms, start, stop));

    printf("(1) Baseline <<<>>>:    %6.2f us\n", baseline_ms / ITERATIONS * 1000);
    printf("(2) cudaLaunchKernelEx: %6.2f us\n", launch_ex_ms / ITERATIONS * 1000);
    printf("(3) + PDL attr:         %6.2f us\n", pdl_attr_ms / ITERATIONS * 1000);
    printf("(4) + Trigger:          %6.2f us\n", trigger_ms / ITERATIONS * 1000);
    printf("(5) + Sync:             %6.2f us\n", sync_ms / ITERATIONS * 1000);
    printf("(6) + Trigger + Sync:   %6.2f us\n", both_ms / ITERATIONS * 1000);

    return 0;
}
