import torch
torch.random.manual_seed(42)

# Import our Python bindings
from _C import mxfp8_matmul


# Function is naive for clarity, should not be like this in production
def quantize_2d(V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(V.shape) == 2
    assert V.shape[0] % 128 == 0
    assert V.shape[1] % 128 == 0
    assert V.dtype == torch.float32

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


def reshape_A_sc(A_sc: torch.Tensor) -> torch.Tensor:
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

    return _A_sc.contiguous()


# Check quantization loss
print("Checking quantization loss...")
V = torch.randn(1024, 1024, dtype=torch.float32)
V_fp8, V_sc = quantize_2d(V)
V_deq = dequantize_2d(V_fp8, V_sc)
abs_diff = torch.abs(V - V_deq)
print('Max adiff:', abs_diff.max())
print('Mean adiff (should be around 1e-2):', abs_diff.mean())

# Matrix dimensions (should not change)
M = 128
K = 128
N = 128

# Generate random matrices
A = torch.randn(M, K, dtype=torch.float32, device="cuda:0") / K ** 0.25
B = torch.randn(N, K, dtype=torch.float32, device="cuda:0") / K ** 0.25
C = torch.zeros(M, N, dtype=torch.float32, device="cuda:0")

# Quantize matrices
A_fp8, _A_sc = quantize_2d(A)
B_fp8, B_sc = quantize_2d(B)

# Reshape scales according to tcgen05 requirements
A_sc = reshape_A_sc(_A_sc)

# Run kernel
mxfp8_matmul(A_fp8, A_sc, B_fp8, B_sc, C)
torch.cuda.synchronize()

# Check correctness
C_ref = torch.matmul(A_fp8.to(torch.float32), B_fp8.T.to(torch.float32))
assert C_ref.dtype == C.dtype
abs_diff = torch.abs(C_ref - C)
print(f"Max adiff: {abs_diff.max()}")
print(f"Mean adiff: {abs_diff.mean()}")
