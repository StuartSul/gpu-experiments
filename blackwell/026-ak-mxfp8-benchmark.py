import numpy as np
import torch
torch.random.manual_seed(42)
torch.set_printoptions(sci_mode=False)

# Import anykernels
import anykernels


# Function is naive for clarity, should not be like this in production
def quantize_2d(V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(V.shape) == 2
    assert V.shape[0] % 128 == 0
    assert V.shape[1] % 128 == 0
    assert V.dtype == torch.bfloat16

    # Following not the OCP MX specs, but the NVIDIA recipe: https://arxiv.org/pdf/2506.08027
    # This specifically follows the Appendix (page 13)
    # To be more precise, we have to use Blackwell hardware (thus, in the kernel)
    block_amax = torch.amax(torch.abs(V).view(V.shape[0], V.shape[1] // 32, 32), dim=-1)
    dest_max = 448.0
    decode_scale = block_amax / dest_max
    V_sc = torch.clamp(torch.ceil(torch.log2(decode_scale)), min=-127)
    V_fp8 = (V / (2 ** V_sc.repeat_interleave(32, dim=-1))).to(torch.float8_e4m3fn)

    # Torch does not support float8_e8m0, so we need to manually convert to uint8
    fp8e8m0_bias = 127
    V_sc = (V_sc + fp8e8m0_bias).to(torch.uint8)

    # Scale loads with TMA should be MN-major
    V_sc = V_sc.T.contiguous()

    return V_fp8, V_sc


# Function is naive for clarity, should not be like this in production
def dequantize_2d(V_fp8: torch.Tensor, V_sc: torch.Tensor) -> torch.Tensor:
    assert len(V_fp8.shape) == 2
    assert len(V_sc.shape) == 2
    assert V_fp8.dtype == torch.float8_e4m3fn
    assert V_sc.dtype == torch.uint8
    assert V_fp8.shape[0] == V_sc.shape[1]
    assert V_fp8.shape[1] == V_sc.shape[0] * 32

    # Torch does not support float8_e8m0, so we need to manually convert
    fp8e8m0_bias = 127
    scale = 2 ** (V_sc.to(torch.float32) - fp8e8m0_bias)

    # Scales are MN-major
    scale = scale.T

    return V_fp8.to(torch.float32) * scale.repeat_interleave(32, dim=-1)


def reshape_sc(A_sc: torch.Tensor) -> torch.Tensor:
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
    assert len(A_sc.shape) == 2
    assert A_sc.dtype == torch.uint8
    assert A_sc.shape[1] % 128 == 0
    assert (A_sc.shape[0] * 32) % 128 == 0

    M_BLOCK = 128
    K_BLOCK = 4 # 128 / 32

    k_blocks, m = A_sc.shape

    _A_sc = A_sc                    # (k_blocks, m)
    _A_sc = _A_sc.transpose(0, 1)   # (m, k_blocks)
    _A_sc = _A_sc.reshape(          # (m / 128, 128, k / 4, 4)
        m // M_BLOCK, M_BLOCK,
        k_blocks // K_BLOCK, K_BLOCK
    )
    _A_sc = _A_sc.transpose(1, 2)   # (m / 128, k / 4, 128, 4) --> last 2 dims are all we need per MM
    _A_sc = _A_sc.reshape(          # (m / 128, k / 4, 4, 32, 4)
        m // M_BLOCK,
        k_blocks // K_BLOCK,
        4, M_BLOCK // 4, K_BLOCK
    )
    _A_sc = _A_sc.transpose(-2, -3) # (m / 128, k / 4, 32, 4, 4)
    _A_sc = _A_sc.reshape(          # (m / 128, k / 4, 32, 16)
        m // M_BLOCK,
        k_blocks // K_BLOCK,
        M_BLOCK // 4, K_BLOCK * 4
    )
    _A_sc = _A_sc.reshape(          # (m / 128, k / 4, 512)
        m // M_BLOCK,               # this step is TK-specific (to load with SV)
        k_blocks // K_BLOCK,
        M_BLOCK * K_BLOCK
    )

    return _A_sc.contiguous()


# Matrix dimensions
M = 204800
N = 2048

# Generate random BF16 matrix from fp8 and scale matrix
A_fp8_ref = ((torch.rand(M, N, dtype=torch.float32, device="cuda") * 2 - 1) * 448).to(torch.float8_e4m3fn)
_A_sc_ref = torch.randint(127 - 20, 127 + 20, (N // 32, M), dtype=torch.uint8, device="cuda")
A_sc_ref = reshape_sc(_A_sc_ref)
A = dequantize_2d(A_fp8_ref, _A_sc_ref).to(torch.bfloat16)

# Benchmark
NUM_WARMUPS = 10
NUM_ITERS = 50

start_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_ITERS)]
end_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_ITERS)]

for i in range(NUM_WARMUPS):
    anykernels.mxfp8_quantize(A)

l2_cache_size = 1024 * 1024 * 128 # ~128MB for Blackwell
l2_cache = torch.randn(l2_cache_size // 2, dtype=torch.bfloat16)
cache_clear = lambda: l2_cache.random_(0, 1)

for i in range(NUM_ITERS):
    cache_clear()
    start_events[i].record()
    anykernels.mxfp8_quantize(A)
    end_events[i].record()
torch.cuda.synchronize()

times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
avg_time = np.mean(times) * 1e-3
std_time = np.std(times) * 1e-3
gb = M * N * (2 + 1 + 1 / 32) * 1e-9
gbps = gb / avg_time
tflop = M * N * 5 * 1e-12

print(f"Average time: {avg_time * 1e6:.2f} ± {std_time * 1e6:.2f} us")
print(f"Average throughput: {gbps:.2f} GB/s")
print(f"Average TFLOPS: {tflop / avg_time:.2f} TFLOP/s")
