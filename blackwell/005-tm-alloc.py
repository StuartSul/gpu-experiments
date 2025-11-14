# Simply invokes the kernel

import torch

from _C import kernel

t = torch.zeros(128, 128, dtype=torch.int32, device="cuda")

kernel(t)
torch.cuda.synchronize() # tensor memory unallocated error will be raised
