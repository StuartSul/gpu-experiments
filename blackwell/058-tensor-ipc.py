"""
torchrun --nproc_per_node 2 58-tensor-ipc.py
"""

from datetime import timedelta
import os
import time

import torch
torch.set_printoptions(sci_mode=False)

from _C import init_tensor_ipc, tensor_ipc, destroy_tensor_ipc


def clean_print(rank, world_size, *args, **kwargs):
    for i in range(world_size):
        if i == rank:
            print(*args, **kwargs, flush=True)
        torch.distributed.barrier()


# Initialize distributed environment
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
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
t0 = time.perf_counter()
init_tensor_ipc(rank, world_size)
t1 = time.perf_counter()
clean_print(rank, world_size, f"Time taken to initialize tensor IPC: {(t1 - t0) * 1e6:.2f} us")

# Generate inputs
clean_print(rank, world_size, "Generating inputs...")
torch.random.manual_seed(42 + rank)
A = torch.randn(128, 128, dtype=torch.float32, device="cuda")

# Run kernel
clean_print(rank, world_size, "Running kernel...")
tensor_ipc(A, rank, world_size)
torch.cuda.synchronize()

# Print results
print_results = False
if print_results:
    for i in range(world_size):
        if i == rank:
            print(f"Rank {i}:")
            print(A)
            print()
        torch.distributed.barrier()

# Cleanup
t0 = time.perf_counter()
destroy_tensor_ipc(rank, world_size)
t1 = time.perf_counter()
clean_print(rank, world_size, f"Time taken to destroy tensor IPC: {(t1 - t0) * 1e6:.2f} us")
if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()
