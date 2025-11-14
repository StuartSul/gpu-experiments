#include <errno.h>
#include <fcntl.h>
#include <stdexcept>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    #error "KittensBroker is not supported on Windows"
#endif

namespace kittens {
namespace detail {
namespace broker {
    
constexpr const char *SHM_KEY = "kbroker";
constexpr const char *SOCK_KEY = "kbroker";

struct KittensVault {
    static constexpr int INIT_CODE = 0x43617473; // "Cats"
    int barrier;
    int sense;
    int init;
};

__host__ inline static void init_sync(
    int local_rank,
    volatile KittensVault *vault
) {
    if (local_rank == 0) {
        // initialize barrier resources
        vault->barrier = 0;
        vault->sense = 0;
        __sync_synchronize(); // make previous writes visible
        vault->init = KittensVault::INIT_CODE;
    } else {
        while (vault->init != KittensVault::INIT_CODE) usleep(1);
        __sync_synchronize(); // see leader's previous writes
    }
}

__host__ inline static void sync(
    int local_world_size,
    volatile KittensVault *vault
) {
    if (vault->init != KittensVault::INIT_CODE)
        throw std::runtime_error("KittensVault not initialized");

    // Phase 1
    int arrived = __sync_add_and_fetch(&vault->barrier, 1);
    if (arrived == local_world_size) vault->sense = 1;
    while (!vault->sense) usleep(1);

    // Phase 2
    arrived = __sync_add_and_fetch(&vault->barrier, -1);
    if (arrived == 0) vault->sense = 0;
    while (vault->sense) usleep(1);
}

__host__ inline void *create_shm(const char *key, size_t size) {
    int shm_fd;
    shm_fd = shm_open(key, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);

    if (shm_fd < 0) {
        if (errno == EEXIST)
            throw std::runtime_error("Named shared memory already exists");
        throw std::runtime_error("Failed to create shared memory");
    }

    if (ftruncate(shm_fd, size) != 0) {
        shm_unlink(key);
        close(shm_fd);
        throw std::runtime_error("Failed to truncate shared memory");
    }

    void *addr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    close(shm_fd);
    if (addr == MAP_FAILED) {
        shm_unlink(key);
        throw std::runtime_error("Failed to map to shared memory");
    }

    return addr;
}

__host__ inline void *open_shm(const char *key, size_t size) {
    int shm_fd;
    while (true) {
        shm_fd = shm_open(key, O_RDWR | O_CLOEXEC, 0);
        if (shm_fd >= 0)
            break;
        if (errno != ENOENT) 
            throw std::runtime_error("Failed to open shared memory");
        usleep(1);
    }

    struct stat shm_st;
    do {
        if (fstat(shm_fd, &shm_st) != 0) {
            shm_unlink(key);
            close(shm_fd);
            throw std::runtime_error("Failed to open shared memory stats");
        }
        usleep(1);
    } while ((size_t)shm_st.st_size < size);

    void *addr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    close(shm_fd);
    if (addr == MAP_FAILED) {
        shm_unlink(key);
        throw std::runtime_error("Failed to map to shared memory");
    }

    return addr;
}

__host__ inline void unlink_shm(const char *key) {
    shm_unlink(key);
}

__host__ inline void unmap_shm(void *addr, size_t size) {
    munmap(addr, size);
}

__host__ inline int create_socket(const char *key, int local_rank) {
    int sock_fd;
    if ((sock_fd = socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0)) < 0)
        throw std::runtime_error("Socket creation error");

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;

    char unique_key[64];
    int n = snprintf(unique_key, sizeof(unique_key), "%s%d", key, local_rank);
    if (n < 0 || n >= (int)sizeof(unique_key)) {
        close(sock_fd);
        throw std::runtime_error("Socket name too long"); 
    }

    size_t len = strnlen(unique_key, sizeof(addr.sun_path));
    if (len > (sizeof(addr.sun_path) - 1)) {
        close(sock_fd);
        throw std::runtime_error("Socket name too long");
    }
    strcpy(addr.sun_path, unique_key);
    unlink(unique_key);

    if (bind(sock_fd, (struct sockaddr *)&addr, SUN_LEN(&addr)) < 0) {
        close(sock_fd);
        throw std::runtime_error("Failed to bind socket");
    }

    return sock_fd;
}

__host__ inline void send_fd(int sock_fd, int data_fd, const char *dst_key, int dst_local_rank) {
    union {
      struct cmsghdr cm;
      char* control;
    } control_un;

    size_t sizeof_control = CMSG_SPACE(sizeof(int));
    control_un.control = reinterpret_cast<char *>(malloc(sizeof_control));
    if (!control_un.control) {
        close(sock_fd);
        close(data_fd);
        throw std::runtime_error("Failed to allocate a control buffer");
    }
  
    struct msghdr msg {};
    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof_control;
  
    struct cmsghdr *cmptr = CMSG_FIRSTHDR(&msg);
    cmptr->cmsg_len = CMSG_LEN(sizeof(int));
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type = SCM_RIGHTS;
    memmove(CMSG_DATA(cmptr), &data_fd, sizeof(data_fd));

    struct sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    char dst_unique_key[64];
    int n = snprintf(dst_unique_key, sizeof(dst_unique_key), "%s%d", dst_key, dst_local_rank);
    if (n < 0 || n >= (int)sizeof(dst_unique_key)) { 
        free(control_un.control);
        close(sock_fd);
        close(data_fd);
        throw std::runtime_error("dst path too long"); 
    }
    strcpy(addr.sun_path, dst_unique_key);
    msg.msg_name = (void *)&addr;
    msg.msg_namelen = sizeof(struct sockaddr_un);
  
    // Dummy payload
    struct iovec iov[1];
    iov[0].iov_base = (void *)"";
    iov[0].iov_len = 1;
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
  
    while (true) {
        ssize_t sent = sendmsg(sock_fd, &msg, 0);
        if (sent <= 0) {
            if (errno == EINTR) continue;
            close(sock_fd);
            close(data_fd);
            free(control_un.control);
            throw std::runtime_error("Failed to send FD over socket");
        }
        break;
    }

    free(control_un.control);
}

__host__ inline int recv_fd(int sock_fd) {
    union {
      struct cmsghdr cm;
      char* control;
    } control_un;

    size_t sizeof_control = CMSG_SPACE(sizeof(int));
    control_un.control = reinterpret_cast<char *>(malloc(sizeof_control));
    if (!control_un.control) {
        close(sock_fd);
        throw std::runtime_error("Failed to allocate a control buffer");
    }

    struct msghdr msg {};
    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof_control;

    struct iovec iov[1];
    char dummy_payload[1];
    iov[0].iov_base = (void *)dummy_payload;
    iov[0].iov_len = sizeof(dummy_payload);
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;

    while (true) {
        ssize_t received = recvmsg(sock_fd, &msg, 0);
        if (received <= 0) {
            if (errno == EINTR) {
                msg.msg_controllen = sizeof_control;
                continue;
            }
            free(control_un.control);
            close(sock_fd);
            throw std::runtime_error("Failed to receive data over socket");
        }
        break;
    }

    if (msg.msg_flags & MSG_CTRUNC) {
        free(control_un.control);
        close(sock_fd);
        throw std::runtime_error("control data truncated");
    }

    struct cmsghdr *cmptr = CMSG_FIRSTHDR(&msg);
    if (!cmptr ||
        cmptr->cmsg_len < CMSG_LEN(sizeof(int)) ||
        cmptr->cmsg_level != SOL_SOCKET ||
        cmptr->cmsg_type != SCM_RIGHTS) {
        free(control_un.control);
        close(sock_fd);
        throw std::runtime_error("Failed to receive data over socket");
    }

    int data_fd;
    memmove(&data_fd, CMSG_DATA(cmptr), sizeof(data_fd));
    free(control_un.control);

    return data_fd;
}

__host__ inline void unlink_socket(const char *key, int local_rank) {
    char unique_key[64];
    int n = snprintf(unique_key, sizeof(unique_key), "%s%d", key, local_rank);
    if (n < 0 || n >= (int)sizeof(unique_key))
        throw std::runtime_error("Socket name too long");
    unlink(unique_key);
}

__host__ inline void close_socket(int sock_fd) {
    close(sock_fd);
}

} // namespace broker
} // namespace detail
} // namespace kittens

__global__ void kernel(float *ptr, int N) {
    for (int i = 0; i < N; i++) ptr[i] = 3.14;
}

__host__ inline static void enable_all_peer_access(int local_world_size) {
    CUCHECK(cuInit(0));

    int num_available_devices;
    CUCHECK(cuDeviceGetCount(&num_available_devices));
    if (num_available_devices < local_world_size)
        throw std::runtime_error("Not enough GPUs available");

    std::vector<CUdevice> devices(local_world_size);
    std::vector<CUcontext> contexts(local_world_size);

    for (int i = 0; i < local_world_size; i++) {
        CUCHECK(cuDeviceGet(&devices[i], i));
        CUCHECK(cuCtxCreate(&contexts[i], 0, devices[i]));
    }

    for (int i = 0; i < local_world_size; i++) {
        int device_compute_mode;
        CUCHECK(cuDeviceGetAttribute(&device_compute_mode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, devices[i]));
        if (device_compute_mode != CU_COMPUTEMODE_DEFAULT)
            throw std::runtime_error("Device is in an unsupported compute mode");

        int vmm_supported = 0;
        CUCHECK(cuDeviceGetAttribute(&vmm_supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, devices[i]));
        if (!vmm_supported)
        throw std::runtime_error("Device does not support CUDA VMM");
    
        int ipc_handle_supported;
        CUCHECK(cuDeviceGetAttribute(&ipc_handle_supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, devices[i]));
        if (!ipc_handle_supported)
            throw std::runtime_error("Device does not support IPC handles");

        for (int j = 0; j < local_world_size; j++) {
            if (i == j) continue;
            int can_access_peer;
            CUCHECK(cuDeviceCanAccessPeer(&can_access_peer, devices[i], devices[j]));
            if (!can_access_peer)
                throw std::runtime_error("Device cannot access peer device");
            CUCHECK(cuCtxSetCurrent(contexts[i]));
            CUCHECK(cuCtxEnablePeerAccess(contexts[j], 0));
        }
    }

    for (size_t i = 0; i < contexts.size(); ++i)
        CUCHECK(cuCtxDestroy(contexts[i]));
}

void vmm_ipc(int local_rank, int local_world_size) {
    // Setup
    if (local_world_size <= 0)
        throw std::runtime_error("local_world_size must be greater than 0");
    if (local_rank < 0 || local_rank >= local_world_size)
        throw std::runtime_error("local_rank must be between 0 and local_world_size - 1");

    void *p;
    volatile kittens::detail::broker::KittensVault *shm;
    if (local_rank == 0) {
        p = kittens::detail::broker::create_shm(kittens::detail::broker::SHM_KEY, sizeof(kittens::detail::broker::KittensVault));
        shm = reinterpret_cast<volatile kittens::detail::broker::KittensVault *>(p);
        memset(p, 0, sizeof(kittens::detail::broker::KittensVault));
    } else {
        p = kittens::detail::broker::open_shm(kittens::detail::broker::SHM_KEY, sizeof(kittens::detail::broker::KittensVault));
        shm = reinterpret_cast<volatile kittens::detail::broker::KittensVault *>(p);
    }

    kittens::detail::broker::init_sync(local_rank, shm);
    kittens::detail::broker::sync(local_world_size, shm);
    if (local_rank ==0) kittens::detail::broker::unlink_shm(kittens::detail::broker::SHM_KEY);
    kittens::detail::broker::sync(local_world_size, shm);
    int sock = kittens::detail::broker::create_socket(kittens::detail::broker::SOCK_KEY, local_rank);
    kittens::detail::broker::sync(local_world_size, shm);

    // Main
    int N = 1024;
    void *raw_ptr;
    size_t size = N * N * sizeof(float);
    size_t allocated_size;
    kittens::detail::vmm::vm_alloc_map_set_access(&raw_ptr, &allocated_size, size, local_rank, local_world_size);
    using ipc_handle_t = kittens::detail::ipc::handle<kittens::detail::ipc::flavor::VMM>;
    ipc_handle_t ipc_handle; // file descriptor
    kittens::detail::ipc::export_handle(&ipc_handle, raw_ptr);
    kittens::detail::broker::sync(local_world_size, shm);

    enable_all_peer_access(local_world_size);
    kittens::detail::broker::sync(local_world_size, shm);

    if (local_rank == 0) {
        close(ipc_handle.handle_); // no need to share local handle
        std::vector<void *> all_raw_ptrs;
        for (int i = 0; i < local_world_size - 1; i++) {
            int fd = kittens::detail::broker::recv_fd(sock);
            void *other_raw_ptr;
            kittens::detail::ipc::import_handle(&other_raw_ptr, *reinterpret_cast<ipc_handle_t*>(&fd), allocated_size);
            all_raw_ptrs.push_back(other_raw_ptr);
        }
        CUDACHECK(cudaSetDevice(0));
        for (int i = 0; i < local_world_size - 1; i++) {
            kernel<<<1, 1>>>(reinterpret_cast<float *>(all_raw_ptrs[i]), N * N);
            CUDACHECK(cudaDeviceSynchronize());
        }
        // Cleanup
        for (int i = 0; i < local_world_size - 1; i++) {
            kittens::detail::vmm::vm_unmap(all_raw_ptrs[i], allocated_size);
        }
    } else {
        kittens::detail::broker::send_fd(sock, ipc_handle.handle_, kittens::detail::broker::SOCK_KEY, 0);
        close(ipc_handle.handle_);
    }
    kittens::detail::broker::sync(local_world_size, shm);

    // Load data from device to host and print first 10 elements
    float *host_data = new float[N * N];
    CUDACHECK(cudaSetDevice(local_rank));
    CUDACHECK(cudaMemcpy(host_data, raw_ptr, size, cudaMemcpyDeviceToHost)); 
    for (int i = 0; i < local_world_size; i++) {
        if (i == local_rank) {
            std::cout << local_rank << ": ";
            for (int j = 0; j < 10 && j < N * N; j++) {
                std::cout << host_data[j] << " ";
            }
            std::cout << std::endl;
        }
        kittens::detail::broker::sync(local_world_size, shm);
    }

    // Cleanup for main
    kittens::detail::vmm::vm_unmap(raw_ptr, allocated_size);
    delete[] host_data;

    // Cleanup
    kittens::detail::broker::unmap_shm(p, sizeof(kittens::detail::broker::KittensVault));
    kittens::detail::broker::unlink_socket(kittens::detail::broker::SOCK_KEY, local_rank);
    kittens::detail::broker::close_socket(sock);
}

// To test with torchrun
PYBIND11_MODULE(_C, m){
    m.def("vmm_ipc", &vmm_ipc);
}
