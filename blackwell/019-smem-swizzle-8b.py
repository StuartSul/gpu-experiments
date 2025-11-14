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
inputs = torch.arange(N, dtype=torch.uint8, device="cuda").unsqueeze(0).repeat_interleave(M, dim=0)
loaded = torch.zeros(M, N, dtype=torch.uint8, device="cuda")
stored = torch.zeros(M, N, dtype=torch.uint8, device="cuda")

# Run kernel
print("Running kernel...")
kernel(inputs, loaded, stored)
torch.cuda.synchronize()

# Convert back for comparison
loaded = loaded.to(torch.int32)
stored = stored.to(torch.int32)

print(loaded)
print(stored)

abs_diff = (loaded - stored).abs()
print(f"Max adiff: {abs_diff.max().item()}")
print(f"Mean adiff: {abs_diff.to(torch.float32).mean().item()}")

row_sum = loaded.sum(dim=1)
print(f"Row sum: {row_sum}")

col_sum = loaded.sum(dim=0)
print(f"Col sum: {col_sum}")

for i in range(128 // 16):
    abs_diff = (loaded[:16] - loaded[i * 16:(i + 1) * 16]).abs()
    print(f"Max adiff: {abs_diff.max().item()}")

if False:
    np.savetxt('loaded_matrix.txt', loaded.cpu().numpy(), fmt='%3d', delimiter=' ')
