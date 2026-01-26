/*
    This test uses 1 thread per threadblock, but each threadblock uses full SMEM.
    So only 1 threadblock can get scheduled per SM.
    
    Observations:
    - PDL uses all SMs to their fullest; if some SMs are available, it schedules a subset 
      of second kernel threadblocks on them when possible.
    - But PDL does not let you overlap scheduling within a fully occupied threadblock; for 
      instance when you set `NUM_BLOCKS = 148` below, the second kernel won't begin until
      all of the first kernel threadblocks finish, making PDLs just an expensive overhead.
      But when you set `NUM_BLOCKS = 148 - 1`, the first threadblock of the second kernel
      gets scheduled in advance and can print out its message.
*/

#include "kittens.cuh"

// constexpr int NUM_BLOCKS = 148 - 1;
constexpr int NUM_BLOCKS = 148;

__global__ void primary_kernel(bool verbose) {
    if (verbose && blockIdx.x == 0) printf("[Primary] Before cudaTriggerProgrammaticLaunchCompletion()\n");

    // Let the secondary kernel start
    cudaTriggerProgrammaticLaunchCompletion();

    for (int i = 0; i < 10; i++) __nanosleep(10000);

    if (verbose && blockIdx.x == 0) printf("[Primary] After cudaTriggerProgrammaticLaunchCompletion()\n");

    for (int i = 0; i < 10; i++) __nanosleep(10000);
}

__global__ void secondary_kernel(bool verbose) {
    if (verbose && blockIdx.x == 0) printf("[Secondary] Before cudaGridDependencySynchronize()\n");

    cudaGridDependencySynchronize();

    if (verbose && blockIdx.x == 0) printf("[Secondary] After cudaGridDependencySynchronize()\n");
}

int main() {
    CUDACHECK(cudaFuncSetAttribute(primary_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kittens::MAX_SHARED_MEMORY));
    CUDACHECK(cudaFuncSetAttribute(secondary_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kittens::MAX_SHARED_MEMORY));

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;

    // Set up launch config for secondary kernel
    cudaLaunchConfig_t config = {0};
    config.gridDim = dim3(NUM_BLOCKS);
    config.blockDim = dim3(1);
    config.dynamicSmemBytes = kittens::MAX_SHARED_MEMORY;
    config.stream = 0;  // default stream
    config.attrs = attribute;
    config.numAttrs = 1;

    // Warmup runs
    for (int i = 0; i < 10; i++) {
        primary_kernel<<<NUM_BLOCKS, 1, kittens::MAX_SHARED_MEMORY>>>(false);
        CUDACHECK(cudaLaunchKernelEx(&config, secondary_kernel, false));
    }
    CUDACHECK(cudaDeviceSynchronize());

    // Main run
    primary_kernel<<<NUM_BLOCKS, 1, kittens::MAX_SHARED_MEMORY>>>(true);
    CUDACHECK(cudaLaunchKernelEx(&config, secondary_kernel, true));
    CUDACHECK(cudaDeviceSynchronize());

    return 0;
}
