"""
Naive, non-persistent, non-TMA matmul

Reaches suprisingly good 1116 TFLOP/s with BF16!
"""

import numpy as np
import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()
CUDA_AUTOTUNE_CONFIG = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
]


@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


# triton.autotune will try all the given `configs`, whenever ones of the given `key` changes
@triton.autotune(configs=CUDA_AUTOTUNE_CONFIG, key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel(
    a_ptr, # M x K
    b_ptr, # K x N
    c_ptr, # M x N
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # This helps to guide integer analysis in the backend to optimize load/store offset address calculation
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # Create pointers
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Do the actual computation
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Epilogue
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)

    # Convert to fp16 before write-back
    c = accumulator.to(tl.bfloat16)

    # Store
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b, c, activation=""):
    # This is messed up but it's fine
    M, K = a.shape
    K, N = b.shape
    M, N = c.shape

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation
    )

    return c


if __name__ == "__main__":
    M = 16384
    K = 16384
    N = 16384

    NUM_WARMUPS = 5
    NUM_ITERS = 10

    torch.manual_seed(0)
    a = torch.randn((M, K), device=DEVICE, dtype=torch.bfloat16) / K ** 0.25
    b = torch.randn((K, N), device=DEVICE, dtype=torch.bfloat16) / K ** 0.25
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    
    # Check correctness
    matmul(a, b, c)
    c_ref = torch.matmul(a, b)
    assert torch.allclose(c, c_ref, atol=1e-12, rtol=0), "Triton and Torch differ"

    # Warmup
    for i in range(NUM_WARMUPS):
        matmul(a, b, c)
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    l2_cache = torch.randn(1024 * 1024 * 128 // 2, device=DEVICE, dtype=torch.bfloat16)
    times = []
    for i in range(NUM_ITERS):
        l2_cache.random_(0, 1)
        start_event.record()
        matmul(a, b, c)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event) * 1e-3)

    avg_time = np.mean(times)
    std_time = np.std(times)
    flops = 2.0 * M * N * K
    tflops = flops * 1e-12

    print(f"Average time: {avg_time * 1e6:.2f} ± {std_time * 1e6:.2f} us")
    print(f"Average TFLOPS: {tflops / (avg_time):.2f} TFLOp/s")
