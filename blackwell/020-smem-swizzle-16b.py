import numpy as np
import torch
torch.random.manual_seed(42)
torch.set_printoptions(sci_mode=False)

# Import our Python bindings
from _C import kernel


# Matrix dimensions
M = 128
N = 128

# Globals
print("Initializing globals...")
inputs = []
for i in range(M):
    inputs.append([])
    for j in range(N):
        inputs[i].append((i << 8) + j)
inputs = torch.tensor(inputs, dtype=torch.uint16, device="cuda").view(torch.bfloat16)
loaded = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
stored = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

# Run kernel
print("Running kernel...")
kernel(inputs, loaded, stored)
torch.cuda.synchronize()

# Convert back for comparison
loaded = loaded.view(torch.uint16).to(torch.int32)
stored = stored.view(torch.uint16).to(torch.int32)

loaded = (loaded >> 8) * 10_000_000 + 9_990_000 + (loaded & 0xFF)
stored = (stored >> 8) * 10_000_000 + 9_990_000 + (stored & 0xFF)

print(loaded)
print(stored)

abs_diff = (loaded - stored).abs()
print(f"Max adiff: {abs_diff.max().item()}")
print(f"Mean adiff: {abs_diff.to(torch.float32).mean().item()}")

row_sum = loaded.sum(dim=1)
print(f"Row sum: {row_sum}")

col_sum = loaded.sum(dim=0)
print(f"Col sum: {col_sum}")

np.savetxt('loaded_matrix.txt', loaded.cpu().numpy(), fmt='%010d', delimiter=' ')
np.savetxt('stored_matrix.txt', stored.cpu().numpy(), fmt='%010d', delimiter=' ')
