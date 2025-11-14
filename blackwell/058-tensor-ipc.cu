/*
    Goal: Be able to share tensors from different processes before the kernel launch!
    Benchmark: Overhead of tensor IPC

    Results (8 GPUs):
        Time taken to initialize tensor IPC: 73668.13 us
        Time taken to initialize tensor IPC: 52550.96 us
        Time taken to initialize tensor IPC: 82662.36 us
        Time taken to initialize tensor IPC: 10526.94 us
        Time taken to initialize tensor IPC: 39107.79 us
        Time taken to initialize tensor IPC: 95825.40 us
        Time taken to initialize tensor IPC: 72496.76 us
        Time taken to initialize tensor IPC: 77326.99 us
        Time taken to export IPC handle: 61 us
        Time taken to export IPC handle: 67 us
        Time taken to export IPC handle: 53 us
        Time taken to export IPC handle: 84 us
        Time taken to export IPC handle: 76 us
        Time taken to export IPC handle: 67 us
        Time taken to export IPC handle: 130 us
        Time taken to export IPC handle: 58 us
        Time taken to share IPC handle: 106 us
        Time taken to share IPC handle: 99 us
        Time taken to share IPC handle: 108 us
        Time taken to share IPC handle: 97 us
        Time taken to share IPC handle: 75 us
        Time taken to share IPC handle: 130 us
        Time taken to share IPC handle: 127 us
        Time taken to share IPC handle: 126 us
        Time taken to import IPC handle: 889 us
        Time taken to import IPC handle: 611 us
        Time taken to import IPC handle: 811 us
        Time taken to import IPC handle: 1036 us
        Time taken to import IPC handle: 543 us
        Time taken to import IPC handle: 673 us
        Time taken to import IPC handle: 924 us
        Time taken to import IPC handle: 839 us
        Time taken to close import pointer: 728 us
        Time taken to close import pointer: 1352 us
        Time taken to close import pointer: 993 us
        Time taken to close import pointer: 803 us
        Time taken to close import pointer: 816 us
        Time taken to close import pointer: 1029 us
        Time taken to close import pointer: 1031 us
        Time taken to close import pointer: 1036 us
        Time taken to sync: 124 us
        Time taken to sync: 82 us
        Time taken to sync: 103 us
        Time taken to sync: 105 us
        Time taken to sync: 106 us
        Time taken to sync: 105 us
        Time taken to sync: 99 us
        Time taken to sync: 116 us
        Time taken to destroy tensor IPC: 49.59 us
        Time taken to destroy tensor IPC: 74.78 us
        Time taken to destroy tensor IPC: 56.11 us
        Time taken to destroy tensor IPC: 50.80 us
        Time taken to destroy tensor IPC: 67.78 us
        Time taken to destroy tensor IPC: 75.41 us
        Time taken to destroy tensor IPC: 38.47 us
        Time taken to destroy tensor IPC: 39.75 us
*/

#include "gpu-experiments.cuh"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <semaphore.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define SHM_KEY      "/tensor_shm"
#define SEM_COUNTER_KEY     "/tensor_sem_counter"
#define SEM_ENTER_KEY     "/tensor_sem_enter"
#define SEM_EXIT_KEY     "/tensor_sem_exit"
#define SEM_READY_KEY "/tensor_sem_ready"
#define SHM_SIZE     8192

struct TensorIPCShm {
    int count;
    cudaIpcMemHandle_t handle[(SHM_SIZE - sizeof(int)) / sizeof(cudaIpcMemHandle_t)];
    TensorIPCShm() : count(0) {}
};

static TensorIPCShm *shm = nullptr;
static sem_t *sem_counter = nullptr;
static sem_t *sem_enter = nullptr;
static sem_t *sem_exit = nullptr;
static sem_t *sem_ready = nullptr;

static inline void xsem_wait(sem_t* s) { while (sem_wait(s) == -1 && errno == EINTR); }
static inline void xsem_post(sem_t* s) { while (sem_post(s) == -1); }

static void sync(int rank, int world_size) {
    // Phase 1: arrive
    xsem_wait(sem_counter);
    if (++shm->count == world_size) {
        for (int i = 0; i < world_size; i++)
            xsem_post(sem_enter);
    }
    xsem_post(sem_counter);
    xsem_wait(sem_enter);
  
    // Phase 2: depart
    xsem_wait(sem_counter);
    if (--shm->count == 0) {
        for (int i = 0; i < world_size; i++)
            xsem_post(sem_exit);
    }
    xsem_post(sem_counter);
    xsem_wait(sem_exit);
}

static bool is_tensor_ipc_initialized() {
    return shm != nullptr && sem_counter != nullptr && sem_enter != nullptr && sem_exit != nullptr && sem_ready != nullptr;
}

static bool is_tensor_ipc_destroyed() {
    return shm == nullptr && sem_counter == nullptr && sem_enter == nullptr && sem_exit == nullptr && sem_ready == nullptr;
}

static bool is_tensor_ipc_supported(int device_id = -1) {
    // Check if IPC is supported
    if (device_id == -1)
        CUDACHECK(cudaGetDevice(&device_id));
    int ipc_supported;
    CUDACHECK(cudaDeviceGetAttribute(&ipc_supported, cudaDevAttrIpcEventSupport, device_id));
    return ipc_supported;
}

void init_tensor_ipc(int rank, int world_size) {
    if (rank >= world_size)
        throw std::runtime_error("Rank is greater than world size");
    if ((SHM_SIZE - sizeof(int)) / sizeof(cudaIpcMemHandle_t) < world_size)
        throw std::runtime_error("Allocate more shared memory");
    if (world_size > SEM_VALUE_MAX)
        throw std::runtime_error("world_size > SEM_VALUE_MAX");
    if (is_tensor_ipc_initialized())
        throw std::runtime_error("Already initialized");
    if (!is_tensor_ipc_supported())
        throw std::runtime_error("Tensor IPC is not supported on this device");

    // Create or open existing shared memory
    int shm_id = shm_open(SHM_KEY, O_CREAT | O_RDWR, 0600);
    if (shm_id == -1)
        throw std::runtime_error("Failed to create shared memory");

    // Create named semaphores
    sem_counter = sem_open(SEM_COUNTER_KEY, O_CREAT, 0600, 1);
    sem_enter = sem_open(SEM_ENTER_KEY, O_CREAT, 0600, 0);
    sem_exit = sem_open(SEM_EXIT_KEY, O_CREAT, 0600, 0);
    sem_ready = sem_open(SEM_READY_KEY, O_CREAT, 0600, 0);
    if (sem_counter == SEM_FAILED || sem_enter == SEM_FAILED || sem_exit == SEM_FAILED || sem_ready == SEM_FAILED)
        throw std::runtime_error("Failed to create semaphores");

    if (rank == 0) {
        // Allocate a page-aligned block
        if (ftruncate(shm_id, SHM_SIZE) == -1)
            throw std::runtime_error("Failed to allocate shared memory");

        // Map shared memory
        shm = (TensorIPCShm *)mmap(
            nullptr, SHM_SIZE, PROT_READ | PROT_WRITE, 
            MAP_SHARED, shm_id, 0
        );
        if (shm == MAP_FAILED)
            throw std::runtime_error("Failed to map shared memory");
        close(shm_id);
    
        // Initialize shared memory
        shm->count = 0;

        // Wake up other processes
        for (int i = 0; i < world_size - 1; i++)
            xsem_post(sem_ready);
    } else {
        // Wait until initialized
        xsem_wait(sem_ready);

        // Map shared memory
        void* p = mmap(
            nullptr, SHM_SIZE, PROT_READ | PROT_WRITE, 
            MAP_SHARED, shm_id, 0
        );
        if (p == MAP_FAILED)
            throw std::runtime_error("Failed to map shared memory");
        close(shm_id);
        shm = (TensorIPCShm *)p;
    }

    // Ensure all processes reach here
    sync(rank, world_size);

    // Unlink immediately
    if (rank == 0) {
        shm_unlink(SHM_KEY);
        sem_unlink(SEM_COUNTER_KEY);
        sem_unlink(SEM_ENTER_KEY);
        sem_unlink(SEM_EXIT_KEY);
        sem_unlink(SEM_READY_KEY);
    }

    // Clean up
    sem_close(sem_ready);
}

void destroy_tensor_ipc(int rank, int world_size) {
    if (sem_counter) sem_close(sem_counter);
    if (sem_enter) sem_close(sem_enter);
    if (sem_exit) sem_close(sem_exit);
    if (shm) munmap(shm, SHM_SIZE);
    sem_counter = nullptr;
    sem_enter = nullptr;
    sem_exit = nullptr;
    shm = nullptr;
}

__global__ void set_value(float *ptr, float val, int N) {
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        ptr[i] = val;
}

void tensor_ipc(pybind11::object tensor, int rank, int world_size) {
    // Simple checks
    if (!pybind11::hasattr(tensor, "__class__"))
        throw std::runtime_error("Not a Python object.");
    if (tensor.attr("__class__").attr("__name__").cast<std::string>() != "Tensor")
        throw std::runtime_error("Not a torch.Tensor object.");
    if (!tensor.attr("is_contiguous")().cast<bool>())
        throw std::runtime_error("Tensor must be contiguous");
    if (tensor.attr("device").attr("type").cast<std::string>() != "cuda")
        throw std::runtime_error("Tensor must be on CUDA device");
    if (!is_tensor_ipc_initialized())
        throw std::runtime_error("Tensor IPC is not initialized");

    // Retrieve device pointer
    // ** Calling cudaFree on export_ptr before cudaIpcCloseMemHandle by peers results in undefined behavior **
    void *export_ptr = reinterpret_cast<void *>(tensor.attr("data_ptr")().cast<uint64_t>());
    if (!export_ptr)
        throw std::runtime_error("Export data pointer is null");

    // Export IPC handle
    auto t0 = std::chrono::high_resolution_clock::now();
    cudaIpcMemHandle_t export_handle;
    CUDACHECK(cudaIpcGetMemHandle(&export_handle, export_ptr));
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < world_size; i++) {
        if (i == rank)
            std::cout << "Time taken to export IPC handle: " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() << " us" << std::endl;
        sync(rank, world_size);
    }

    // Share IPC handle
    t0 = std::chrono::high_resolution_clock::now();
    shm->handle[rank] = export_handle;
    sync(rank, world_size);
    t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < world_size; i++) {
        if (i == rank)
            std::cout << "Time taken to share IPC handle: " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() << " us" << std::endl;
        sync(rank, world_size);
    }

    // Import IPC handle
    // This implicitly does cudaDeviceEnablePeerAccess
    // ** import_ptr MUST be freed with cudaIpcCloseMemHandle **
    t0 = std::chrono::high_resolution_clock::now();
    void *import_ptr;
    cudaIpcMemHandle_t import_handle = shm->handle[(rank + 1) % world_size];
    CUDACHECK(cudaIpcOpenMemHandle(&import_ptr, import_handle, cudaIpcMemLazyEnablePeerAccess)); // this is the only flag supported
    t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < world_size; i++) {
        if (i == rank)
            std::cout << "Time taken to import IPC handle: " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() << " us" << std::endl;
        sync(rank, world_size);
    }

    // Run kernel
    set_value<<<1, 1>>>((float *)import_ptr, (float)rank, 128 * 128);
    CUDACHECK(cudaDeviceSynchronize());

    // Close import_ptr (MUST be done first)
    t0 = std::chrono::high_resolution_clock::now();
    CUDACHECK(cudaIpcCloseMemHandle(import_ptr));
    t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < world_size; i++) {
        if (i == rank)
            std::cout << "Time taken to close import pointer: " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() << " us" << std::endl;
        sync(rank, world_size);
    }

    t0 = std::chrono::high_resolution_clock::now();
    sync(rank, world_size);
    t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < world_size; i++) {
        if (i == rank)
            std::cout << "Time taken to sync: " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() << " us" << std::endl;
        sync(rank, world_size);
    }

    // Freeing export_ptr will be done by PyTorch
}

PYBIND11_MODULE(_C, m){
    m.def("init_tensor_ipc", &init_tensor_ipc);
    m.def("tensor_ipc", &tensor_ipc);
    m.def("destroy_tensor_ipc", &destroy_tensor_ipc);
}
