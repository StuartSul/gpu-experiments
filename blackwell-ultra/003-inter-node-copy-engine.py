"""
How to run:
    Set the env vars below and run once per node. The transfer is RANK 0 -> RANK 1.
    Defaults assume a single-node 2-GPU sanity run.

    MASTER_ADDR   IP of global rank 0's node     (default 127.0.0.1)
    MASTER_PORT   rendezvous port                (default 29500)
    RANK          this process's global rank     (default 0)
    WORLD_SIZE    number of participating ranks  (default 2)
    DEVICE        local CUDA device index        (default = RANK)

Build and run:
    # In the Makefile select `SRC := 003-inter-node-copy-engine.cu` and the PyTorch
    # binding configuration (config 4), then on EACH node:
    make

    # On node 0 (DEVICE can be anything):
    MASTER_ADDR=<rank-0 IP> RANK=0 WORLD_SIZE=2 DEVICE=0 make run
    
    # On node 1 (DEVICE can be anything):
    MASTER_ADDR=<rank-0 IP> RANK=1 WORLD_SIZE=2 DEVICE=1 make run

Requires the CUDA symmetric-memory fabric path (nvidia-imex) to be configured across
the rack so a peer node's buffer is NVLink-addressable.
"""

import gc
import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from _C import benchmark_copy_engine

GiB = 1024 ** 3
DATA_BYTES = 10 * GiB
DTYPE = torch.float32
SRC_VALUE = 3.14
WARMUP_ITERS = 1
TIMED_ITERS = 1
SRC_RANK = 0
DST_RANK = 1


def main():
    # Manual rendezvous values (from environment variables)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "2"))
    local_device = int(os.environ.get("DEVICE", str(rank)))
    # MASTER_ADDR / MASTER_PORT are consumed by init_process_group; default them if unset.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    torch.cuda.set_device(local_device)
    device = torch.device(f"cuda:{local_device}")
    print(f"[rank {rank}] connecting to {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']} ...", flush=True)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=local_device)
    print(f"[rank {rank}] process group ready", flush=True)

    is_src = rank == SRC_RANK
    is_dst = rank == DST_RANK
    numel = DATA_BYTES // DTYPE.itemsize
    nbytes = numel * DTYPE.itemsize

    if rank == 0:
        print("\nInter-node NVLink Bandwidth Test (Copy Engine)")
        print("==============================================")
        print(f"Data size: {DATA_BYTES / GiB:.2f} GiB ({DTYPE})")
        print(f"Direction: global rank {SRC_RANK} -> global rank {DST_RANK}")
        print(f"World size: {world_size}\n")

    # Symmetric-memory buffer on every rank (NVLink-addressable across the fabric domain)
    buf = symm_mem.empty(numel, dtype=DTYPE, device=device)
    buf.fill_(SRC_VALUE if is_src else 0.0)
    torch.cuda.synchronize()
    print(f"[rank {rank}] symmetric memory allocated", flush=True)

    hdl = symm_mem.rendezvous(buf, dist.group.WORLD.group_name)
    print(f"[rank {rank}] rendezvous done", flush=True)
    dist.barrier()

    # Source rank pushes its buffer into the destination rank's buffer over NVLink.
    if is_src:
        avg_ms = benchmark_copy_engine(
            hdl.buffer_ptrs[DST_RANK], hdl.buffer_ptrs[rank], nbytes, WARMUP_ITERS, TIMED_ITERS
        )
        bandwidth = (DATA_BYTES / GiB) / (avg_ms / 1000.0)
        print(f"[rank {rank}] Transfer time: {avg_ms:.3f} ms")
        print(f"[rank {rank}] Bandwidth: {bandwidth:.2f} GB/s")

    dist.barrier()  # ensure the copy is complete before the destination verifies

    # Destination rank checks that it received the source value.
    if is_dst:
        torch.cuda.synchronize()
        expected = torch.full((), SRC_VALUE, dtype=DTYPE, device=device)
        mismatches = int((~torch.isclose(buf, expected, atol=1e-3)).sum().item())
        status = "PASSED" if mismatches == 0 else f"FAILED ({mismatches} mismatches)"
        print(f"[rank {rank}] Correctness check {status}")
        print(f"[rank {rank}] First 10 values: {buf[:10].tolist()}")

    dist.barrier()
    del hdl  # see 082: the handle must be released or the process group teardown hangs
    gc.collect()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
