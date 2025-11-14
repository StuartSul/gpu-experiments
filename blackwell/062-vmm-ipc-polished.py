from datetime import timedelta
import os

import torch
import torch.distributed
torch.set_printoptions(sci_mode=False)

from _C import vmm_ipc


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

# Run our function
vmm_ipc(local_rank, local_world_size)

# Clean up
torch.distributed.destroy_process_group()
