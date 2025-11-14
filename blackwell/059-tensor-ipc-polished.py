"""
torchrun --nproc_per_node 8 59-tensor-ipc-polished.py
"""

from datetime import timedelta
import os

import torch
torch.set_printoptions(sci_mode=False)

from _C import TensorIPC, tensor_ipc_example_func


# Initialize distributed environment
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
world_size = int(os.environ.get("WORLD_SIZE", 1))
torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(
    backend="nccl",
    device_id=local_rank,
    rank=rank,
    world_size=world_size,
    timeout=timedelta(seconds=30),
)

# Initialize tensor IPC
tensor_ipc = TensorIPC(local_rank, local_world_size)

# Generate inputs
torch.random.manual_seed(42 + rank)
A = torch.randn(128, 128, dtype=torch.float32, device="cuda")

# Run kernel
tensor_ipc_example_func(A, tensor_ipc)
torch.cuda.synchronize()

# Print results
for i in range(world_size):
    if i == rank:
        print(f"Rank {i}:")
        print(A)
        print()
    torch.distributed.barrier()

# Cleanup
if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()
