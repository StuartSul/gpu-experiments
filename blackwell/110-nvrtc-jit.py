import ctypes
import functools
import hashlib
import os
from pathlib import Path

import numpy as np

import cuda.bindings.driver as cuda_driver
import cuda.bindings.nvrtc as nvrtc

# ---------------------------------------------------------------------------
# CUDA helpers
# ---------------------------------------------------------------------------

def check_cuda(err: cuda_driver.CUresult) -> None:
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error: {err}")


def check_nvrtc(err: nvrtc.nvrtcResult) -> None:
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"NVRTC error: {err}")


@functools.cache
def initialize_cuda_context(device_index: int = 0) -> None:
    (err,) = cuda_driver.cuInit(0)
    check_cuda(err)
    err, dev = cuda_driver.cuDeviceGet(device_index)
    check_cuda(err)
    err, ctx = cuda_driver.cuDevicePrimaryCtxRetain(dev)
    check_cuda(err)
    (err,) = cuda_driver.cuCtxSetCurrent(ctx)
    check_cuda(err)


@functools.cache
def get_sm_arch(device_index: int = 0) -> tuple[int, int]:
    err, dev = cuda_driver.cuDeviceGet(device_index)
    check_cuda(err)
    err, major = cuda_driver.cuDeviceGetAttribute(
        cuda_driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev
    )
    check_cuda(err)
    err, minor = cuda_driver.cuDeviceGetAttribute(
        cuda_driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev
    )
    check_cuda(err)
    return major, minor


@functools.cache
def get_cuda_driver_version() -> int:
    err, version = cuda_driver.cuDriverGetVersion()
    check_cuda(err)
    return version


@functools.cache
def find_cuda_include_dir() -> str:
    def _check(base: Path) -> str | None:
        p = base / "include"
        if (p / "cuda_bf16.h").exists():
            return str(p)
        return None

    # 1. Explicit env vars
    for env_var in ("CUDA_HOME", "CUDA_PATH"):
        env_val = os.environ.get(env_var)
        if env_val and (result := _check(Path(env_val))):
            return result

    # 2. Infer from PATH / LD_LIBRARY_PATH
    for env_var in ("PATH", "LD_LIBRARY_PATH"):
        env_val = os.environ.get(env_var, "")
        for entry in env_val.split(os.pathsep):
            if not entry:
                continue
            candidate = Path(entry).parent  # remove bin/ or include/
            if result := _check(candidate):
                return result

    # 3. Common system paths
    for base in ("/usr/local/cuda", "/usr/cuda"):
        if result := _check(Path(base)):
            return result

    raise RuntimeError("Cannot find CUDA include directory")

# ---------------------------------------------------------------------------
# NVRTC compilation logic
# ---------------------------------------------------------------------------

COMMON_NVRTC_FLAGS = (
    "--std=c++20",
    "--use_fast_math",
    "-Xptxas=--verbose",
    "-Xptxas=--warn-on-spills",
    "-DNDEBUG",
    "-lineinfo",
    # --pch, --create-pch=..., --use-pch=...
)
CUBIN_CACHE_DIR = Path.home() / ".cache" / "megakittens" / "cubin"


def _cubin_cache_key(src: str, major: int, minor: int) -> str:
    sm_suffix = "a" if major >= 9 else ""
    cuda_ver = get_cuda_driver_version()
    payload = f"cuda_{cuda_ver}\nsm_{major}{minor}{sm_suffix}\n{src}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_cubin_from_cache(key: str) -> bytes | None:
    path = CUBIN_CACHE_DIR / f"{key}.cubin"
    if path.exists():
        return path.read_bytes()
    return None


def _save_cubin_to_cache(key: str, cubin: bytes) -> None:
    CUBIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CUBIN_CACHE_DIR / f"{key}.cubin").write_bytes(cubin)


@functools.cache
def compile_source_to_cubin(src: str, major: int, minor: int) -> bytes:
    # 0. Check file-backed cache
    cache_key = _cubin_cache_key(src, major, minor)
    cached = _load_cubin_from_cache(cache_key)
    if cached is not None:
        return cached

    # 1. Create NVRTC program instance
    err, prog = nvrtc.nvrtcCreateProgram(
        src.encode("utf-8"),  # CUDA source code
        b"kernel.cu",         # program name (for diagnostics)
        0,                    # number of inline headers
        None,                 # array of inline header sources
        None,                 # array of inline header include names
    )
    check_nvrtc(err)

    # 2. Prepare compiler flags and compile
    sm_suffix = "a" if major >= 9 else ""
    cuda_include = find_cuda_include_dir()
    opts = tuple(flag.encode("utf-8") for flag in COMMON_NVRTC_FLAGS) + (
        f"--gpu-architecture=sm_{major}{minor}{sm_suffix}".encode("utf-8"),
        f"--include-path={cuda_include}".encode("utf-8"),
    )
    (err_compile,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)

    # 3. Print compiler logs and check compilation error
    err, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
    check_nvrtc(err)
    log = b" " * log_size
    (err,) = nvrtc.nvrtcGetProgramLog(prog, log)
    check_nvrtc(err)
    decoded_log = log.decode(errors="ignore").strip()
    if decoded_log:
        print(decoded_log)
    check_nvrtc(err_compile)

    # 4. Retrieve the compiled CUBIN binary
    err, cubin_size = nvrtc.nvrtcGetCUBINSize(prog)
    check_nvrtc(err)
    if cubin_size == 0:
        raise RuntimeError("NVRTC returned no CUBIN")
    cubin = b" " * cubin_size
    (err,) = nvrtc.nvrtcGetCUBIN(prog, cubin)
    check_nvrtc(err)

    # 5. Destroy NVRTC program instance
    (err,) = nvrtc.nvrtcDestroyProgram(prog)
    check_nvrtc(err)

    # 6. Save to file-backed cache
    _save_cubin_to_cache(cache_key, cubin)

    return cubin


def load_cubin_module(cubin: bytes) -> cuda_driver.CUmodule:
    err, module = cuda_driver.cuModuleLoadData(np.char.array(cubin))
    check_cuda(err)
    return module


def get_kernel_from_cubin_module(
    module: cuda_driver.CUmodule, kernel_name: bytes
) -> cuda_driver.CUfunction:
    err, fn = cuda_driver.cuModuleGetFunction(module, kernel_name)
    check_cuda(err)
    return fn


def unload_cubin_module(module: cuda_driver.CUmodule) -> None:
    (err,) = cuda_driver.cuModuleUnload(module)
    check_cuda(err)

# ---------------------------------------------------------------------------
# Program logic
# ---------------------------------------------------------------------------

BLOCK_SIZE = 32

import torch
def launch(fn, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int, stream):
    grid_x = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_y = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    arg_A = ctypes.c_void_p(A.data_ptr())
    arg_B = ctypes.c_void_p(B.data_ptr())
    arg_C = ctypes.c_void_p(C.data_ptr())
    arg_N = ctypes.c_int32(N)

    packed = (ctypes.c_void_p * 4)()
    packed[0] = ctypes.addressof(arg_A)
    packed[1] = ctypes.addressof(arg_B)
    packed[2] = ctypes.addressof(arg_C)
    packed[3] = ctypes.addressof(arg_N)

    config = cuda_driver.CUlaunchConfig()
    config.gridDimX = grid_x
    config.gridDimY = grid_y
    config.gridDimZ = 1
    config.blockDimX = BLOCK_SIZE
    config.blockDimY = BLOCK_SIZE
    config.blockDimZ = 1
    config.sharedMemBytes = 0
    config.hStream = stream
    config.numAttrs = 0
    config.attrs = []

    (err,) = cuda_driver.cuLaunchKernelEx(config, fn, packed, 0)
    check_cuda(err)

# Extern "C" is required to prevent C++ name mangling
KERNEL_SOURCE = r"""
#include <cuda_bf16.h>

extern "C" __global__ void kernel(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int N) {
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;

   if (row < N && col < N) {
       float sum = 0.0f;
       for (int k = 0; k < N; k++) {
           sum += __bfloat162float(A[row * N + k] * B[k * N + col]);
       }
       C[row * N + col] = __float2bfloat16(sum);
   }
}
"""


def main():
    import time

    device_index = 0
    N = 128

    initialize_cuda_context(device_index)
    major, minor = get_sm_arch(device_index)

    t0 = time.perf_counter()
    cubin = compile_source_to_cubin(KERNEL_SOURCE, major, minor)
    t1 = time.perf_counter()
    print(f"Compile 1: {t1 - t0:.4f}s")

    t2 = time.perf_counter()
    cubin = compile_source_to_cubin(KERNEL_SOURCE, major, minor)
    t3 = time.perf_counter()
    print(f"Compile 2: {t3 - t2:.4f}s")

    module = load_cubin_module(cubin)
    fn = get_kernel_from_cubin_module(module, b"kernel")

    A = torch.randn(N, N, device=f"cuda:{device_index}", dtype=torch.bfloat16)
    B = torch.randn(N, N, device=f"cuda:{device_index}", dtype=torch.bfloat16)
    C = torch.empty(N, N, device=f"cuda:{device_index}", dtype=torch.bfloat16)

    stream = torch.cuda.current_stream(device_index).cuda_stream
    torch.cuda.synchronize(device_index)

    t2 = time.perf_counter()
    launch(fn, A, B, C, N, stream)
    torch.cuda.synchronize(device_index)
    t3 = time.perf_counter()
    print(f"Kernel launch + sync: {t3 - t2:.3f}s")

    # Verify against PyTorch's matmul.
    C_ref = A.float() @ B.float()
    torch.testing.assert_close(C, C_ref.bfloat16(), atol=0.5, rtol=0)
    print("Correctness check passed!")

    # Cleanup.
    unload_cubin_module(module)


if __name__ == "__main__":
    main()
