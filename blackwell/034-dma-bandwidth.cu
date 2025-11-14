/*
    Seems like it's kind of hard to get HBM -> HBM bandwidth at the stated HBM bandwidth
    I have to admit this is a weird type of workload to consider
    This code results in 3149 GB/s
*/

#include "gpu-experiments.cuh"

using namespace std;

#define KB(x) ((x) * 1000)
#define MB(x) (KB(x) * 1000)
#define GB(x) (MB(x) * 1000)

static constexpr int SIZE = GB(2);

// Should get 720 GB/s for B200s
int main() {
    // Initialize device
    CUDACHECK(cudaSetDevice(0));
    unsigned char *d_buf[2];
    CUDACHECK(cudaMalloc(&d_buf[0], SIZE));
    CUDACHECK(cudaMalloc(&d_buf[1], SIZE));
    CUDACHECK(cudaMemset(d_buf[0], 0xAE, SIZE));
    CUDACHECK(cudaDeviceSynchronize());

    // Run
    auto begin = chrono::high_resolution_clock::now();
    CUDACHECK(cudaMemcpyAsync(d_buf[1], d_buf[0], SIZE, cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaDeviceSynchronize());
    auto end = chrono::high_resolution_clock::now();
    auto time_us = chrono::duration_cast<chrono::microseconds>(end - begin).count();
    cout << "Time: " << time_us << " us" << endl;
    cout << "Bandwidth: " << (SIZE / (time_us / 1e6)) / (1000. * 1000. * 1000.) << " GB/s" << endl;
    
    // Clean up
    CUDACHECK(cudaFree(d_buf[0]));
    CUDACHECK(cudaFree(d_buf[1]));
    return 0;
}
