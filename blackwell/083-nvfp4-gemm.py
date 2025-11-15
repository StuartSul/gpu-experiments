import torch
torch.random.manual_seed(42)

from _C import nvfp4_gemm

# Generate tensors
print("Generating tensors...")

M = 256
N = 256
K = 128

A_fp4 = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device="cuda").view(dtype=torch.float4_e2m1fn_x2)
A_local_scale = ((torch.rand(M, K, dtype=torch.float32, device="cuda") * 2 - 1) * 448).to(dtype=torch.float8_e4m3fn)
A_global_scale = torch.tensor([32], dtype=torch.float32, device="cuda") # just an arbitrary value

B_fp4 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda").view(dtype=torch.float4_e2m1fn_x2)
B_local_scale = ((torch.rand(N, K, dtype=torch.float32, device="cuda") * 2 - 1) * 448).to(dtype=torch.float8_e4m3fn)
B_global_scale = torch.tensor([32], dtype=torch.float32, device="cuda") # just an arbitrary value

C = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")

# Launch the kernel
print("Launching kernel...")
nvfp4_gemm(A_fp4, A_local_scale, A_global_scale, B_fp4, B_local_scale, B_global_scale, C)

print(A_fp4)
