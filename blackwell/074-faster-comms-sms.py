"""
#!/bin/bash
for run in {1..10}; do
    echo "Starting run $run at $(date)" | tee -a run_log.txt
    for n in {0..147}; do
        OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 74-faster-comms-sms.py $n 2>&1 | tee -a out${run}.txt
    done
    echo "Completed run $run at $(date)" | tee -a run_log.txt
done

nohup bash -c 'for run in {1..10}; do echo "Starting run $run at $(date)" >> run_log.txt; for n in {0..147}; do OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 74-faster-comms-sms.py $n 2>&1 | tee -a out${run}.txt; done; echo "Completed run $run at $(date)" >> run_log.txt; done' > nohup.out 2>&1 &
"""

import os
from typing import Callable

import torch

from _C import TKParallelTensor, p2p_from_dev0_to_dev1


N = 16384
NUM_WARMUP_ITERS = 5
NUM_ITERS = 10

CHECK_CORRECTNESS = False

import sys
SM_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def benchmark(
    func: Callable,
    num_warmup_iters: int = 10,
    num_iters: int = 50,
) -> float:
    for _ in range(num_warmup_iters):
        func()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        func()
    end_event.record()
    torch.cuda.synchronize()

    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / num_iters

    return avg_ms


local_rank = int(os.environ.get("LOCAL_RANK", 0))
local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
assert world_size == local_world_size, "multi-node runs are not supported"
torch.distributed.init_process_group(
    backend="nccl",
    device_id=local_rank,
    rank=rank,
    world_size=world_size
)

device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)
torch.random.manual_seed(local_rank)

tensor_tk = TKParallelTensor(
    (N, N), 
    dtype=torch.bfloat16, 
    local_rank=local_rank, 
    local_world_size=local_world_size, 
    multicast=False
)

if local_rank == 0:
    torch.randn((N, N), out=tensor_tk.data_)
else:
    tensor_tk.data_.zero_()
torch.distributed.barrier()

tk_func = lambda: p2p_from_dev0_to_dev1(tensor_tk, SM_ID)

if CHECK_CORRECTNESS:
    if rank == 0:
        tk_func()
    torch.distributed.barrier()
    if local_rank == 0:
        tensor_from_rank1 = torch.zeros_like(tensor_tk.data_)
        torch.distributed.recv(tensor_from_rank1, src=1)
        print("Max diff:", (tensor_tk.data_ - tensor_from_rank1).abs().max().item())
        print("Mean diff:", (tensor_tk.data_ - tensor_from_rank1).abs().mean().item())
    elif local_rank == 1:
        torch.distributed.send(tensor_tk.data_, dst=0)
    torch.distributed.barrier()

if rank == 0:
    tk_avg_ms = benchmark(tk_func, NUM_WARMUP_ITERS, NUM_ITERS)
    tk_gbps = (N * N * 2 * 1e-9) / (tk_avg_ms * 1e-3)
    print(f"SM={SM_ID} | {tk_avg_ms:.3f} ms | {tk_gbps:.2f} GB/s")
torch.distributed.barrier()

torch.distributed.destroy_process_group()
