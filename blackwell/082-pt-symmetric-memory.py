"""
OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 --rdzv-backend c10d --rdzv-endpoint localhost:0 082-pt-symmetric-memory.py

Apparently hdl object needs to be deleted, otherwise silent SIGTERM.
"""

import gc
import os

import torch
import torch.distributed._symmetric_memory as symm_mem

local_rank = int(os.environ["LOCAL_RANK"])

device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)
torch.distributed.init_process_group("nccl")
torch.manual_seed(42 + local_rank)

if torch.distributed.get_rank() == 0:
    print(f"Benchmarking symmetric memory-based all-reduce...")

msg_sizes = [2**exp for exp in range(12, 21)]

for msg_sz_bytes in msg_sizes:
    t = symm_mem.empty(
        msg_sz_bytes // torch.bfloat16.itemsize,
        dtype=torch.bfloat16,
        device=device,
    )
    hdl = symm_mem.rendezvous(t, torch.distributed.group.WORLD.group_name)
    # _ = (hdl.buffer_ptrs, hdl.multicast_ptr, hdl.signal_pad_ptrs)

    torch.ops.symm_mem.multimem_all_reduce_(
        t,
        "sum",
        torch.distributed.group.WORLD.group_name,
    )

    del hdl
    gc.collect()

torch.distributed.destroy_process_group()
