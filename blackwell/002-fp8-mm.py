import torch
torch.random.manual_seed(42)

# Import our Python bindings
from _C import fp8_matmul

# Matrix dimensions (should not change)
M = 128
K = 128
N = 128

A = (torch.randn(M, K, dtype=torch.bfloat16, device="cuda:0") / K ** 0.25).to(torch.float8_e4m3fn)
B = (torch.randn(N, K, dtype=torch.bfloat16, device="cuda:0") / K ** 0.25).to(torch.float8_e4m3fn)
C = torch.zeros(M, N, dtype=torch.float32, device="cuda:0")

fp8_matmul(A, B, C)
torch.cuda.synchronize()

# Check correctness
C_ref = torch.matmul(A.to(torch.float32), B.T.to(torch.float32))
assert C_ref.dtype == C.dtype
abs_diff = torch.abs(C_ref - C)
print(f"Max absolute difference: {abs_diff.max()}")
print(f"Mean absolute difference: {abs_diff.mean()}")
