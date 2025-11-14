"""
torchrun --nproc_per_node=8 61-vmm-ipc.py
"""

from datetime import timedelta
import os
from time import sleep

import torch
import torch.distributed
torch.set_printoptions(sci_mode=False)

from _C import TKParallelTensor, sync


# Initialize distributed environment
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
world_size = int(os.environ.get("WORLD_SIZE", 1))
assert world_size == local_world_size, "multi-node runs are not supported"
device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)
torch.distributed.init_process_group(
    backend="nccl",
    device_id=local_rank,
    rank=rank,
    world_size=world_size,
    timeout=timedelta(seconds=30),
)

# Allocate the barrier tensor. Must be initialized to zero
barriers = TKParallelTensor((1, 1), dtype=torch.int, local_rank=local_rank, local_world_size=local_world_size, multicast=True)
barriers.data_.zero_()

# Run our function
if True:
    print("Profiling...")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_modules=True,
        with_stack=True
    ) as profiler:
        for i in range(10):
            sync(barriers)

    # Export to Chrome trace format
    if rank == 0:
        profiler.export_chrome_trace(f"trace.json")
        print(f"\nProfiler trace exported to trace.json")

else:
    sleep(local_rank)
    print(f"Rank {local_rank} started")
    sync(barriers)
    torch.cuda.synchronize()
    print(f"Rank {local_rank} done")

# Clean up
torch.distributed.destroy_process_group()
