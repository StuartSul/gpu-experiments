/*
    Inter-node, intra-rack (NVL72) NVLink bandwidth test via the copy engine.
*/

#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Throwing variant (so failures surface as Python exceptions rather than killing the process).
#define CUDACHECK(cmd) do {                                                   \
    cudaError_t err = (cmd);                                                   \
    if (err != cudaSuccess)                                                    \
        throw std::runtime_error(std::string("CUDA error ") + __FILE__ + ":" + \
            std::to_string(__LINE__) + " '" + cudaGetErrorString(err) + "'");  \
} while (0)

static double benchmark_copy_engine(uintptr_t dst_ptr, uintptr_t src_ptr, size_t nbytes, int warmup_iters, int timed_iters) {
    void *dst = reinterpret_cast<void *>(dst_ptr);
    void *src = reinterpret_cast<void *>(src_ptr);

    cudaStream_t stream;
    cudaEvent_t start, stop;
    CUDACHECK(cudaStreamCreate(&stream));
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    // Warm up
    for (int i = 0; i < warmup_iters; ++i)
        CUDACHECK(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, stream));
    CUDACHECK(cudaStreamSynchronize(stream));

    // Timed
    CUDACHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < timed_iters; ++i)
        CUDACHECK(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, stream));
    CUDACHECK(cudaEventRecord(stop, stream));
    CUDACHECK(cudaStreamSynchronize(stream));

    float milliseconds = 0.0f;
    CUDACHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));
    CUDACHECK(cudaStreamDestroy(stream));

    return static_cast<double>(milliseconds) / timed_iters;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("benchmark_copy_engine", &benchmark_copy_engine,
          "Average ms for cudaMemcpyAsync(dst, src, nbytes) over timed_iters copies",
          py::arg("dst_ptr"), py::arg("src_ptr"), py::arg("nbytes"),
          py::arg("warmup_iters") = 1, py::arg("timed_iters") = 1);
}
