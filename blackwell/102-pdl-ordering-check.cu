#include "kittens.cuh"

__global__ void primary_kernel(bool verbose) {
    if (verbose) printf("[Primary] Before cudaTriggerProgrammaticLaunchCompletion()\n");

    // Let the secondary kernel start
    cudaTriggerProgrammaticLaunchCompletion();

    for (int i = 0; i < 10; i++) __nanosleep(10000);

    if (verbose) printf("[Primary] After cudaTriggerProgrammaticLaunchCompletion()\n");

    for (int i = 0; i < 10; i++) __nanosleep(10000);
}

__global__ void secondary_kernel(bool verbose) {
    if (verbose) printf("[Secondary] Before cudaGridDependencySynchronize()\n");

    // Wait until primary kernel fully completes
    cudaGridDependencySynchronize();

    if (verbose) printf("[Secondary] After cudaGridDependencySynchronize()\n");
}

int main() {
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;

    // Set up launch config for secondary kernel
    cudaLaunchConfig_t config = {0};
    config.gridDim = dim3(1);
    config.blockDim = dim3(1);
    config.dynamicSmemBytes = 0;
    config.stream = 0;  // default stream
    config.attrs = attribute;
    config.numAttrs = 1;

    // Warmup runs
    for (int i = 0; i < 10; i++) {
        primary_kernel<<<1, 1>>>(false);
        CUDACHECK(cudaLaunchKernelEx(&config, secondary_kernel, false));
    }
    CUDACHECK(cudaDeviceSynchronize());

    // Main run
    primary_kernel<<<1, 1>>>(true);
    CUDACHECK(cudaLaunchKernelEx(&config, secondary_kernel, true));
    CUDACHECK(cudaDeviceSynchronize());

    return 0;
}
