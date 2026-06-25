"""
How to run:
    make
    mkdir -p ~/anytitan/tmp/nvl72-tests
    cp 011-nvl72-bandwidth.py _C.cpython-312-aarch64-linux-gnu.so ~/anytitan/tmp/nvl72-tests/

    cd ~/anytitan
    xbatch mixture-of-kittens/scripts/xbatch-generic-run.sh \
        --groups 1 --group-size 18 --gpus-per-node 4 --topology-chunk-size 18 --single-block \
        -e 'RUN_CMD=python3 /scratch/anytitan/tmp/nvl72-tests/011-nvl72-bandwidth.py'

Local sanity check:
    torchrun --nproc_per_node=4 011-nvl72-bandwidth.py
"""

import gc
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

# Allow `from _C import ...` regardless of cwd for xbatch run
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _C import benchmark

GiB = 1024 ** 3
DATA_BYTES = 10 * GiB
DTYPE = torch.float32
SRC_VALUE = 3.14
WARMUP_ITERS = 2
TIMED_ITERS = 5
SRC_RANK = 0


def main():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("DEVICE", "0")))
    gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", str(min(world_size, 4))))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=device)

    numel = DATA_BYTES // DTYPE.itemsize
    nbytes = numel * DTYPE.itemsize
    is_src = rank == SRC_RANK
    n_nodes = (world_size + gpus_per_node - 1) // gpus_per_node

    if rank == 0:
        print("\nNVL72 NVLink Bandwidth Sweep (copy-engine / TMA / SM ld-st)")
        print("===========================================================")
        print(f"Data size:   {nbytes / GiB:.2f} GiB ({DTYPE})")
        print(f"World size:  {world_size}  (nodes: {n_nodes}, gpus/node: {gpus_per_node})")
        print(f"Sweep:       rank {SRC_RANK} -> rank j  for j in [0, {world_size - 1}]")
        print(f"Iters:       warmup {WARMUP_ITERS}, timed {TIMED_ITERS}\n", flush=True)

    # Symmetric-memory buffer on every rank (NVLink-addressable across the fabric domain).
    buf = symm_mem.empty(numel, dtype=DTYPE, device=device)
    buf.fill_(SRC_VALUE if is_src else 0.0)
    torch.cuda.synchronize()
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD.group_name)
    if rank == 0:
        print("[rank 0] symmetric memory rendezvous complete", flush=True)
    dist.barrier()

    # 0 -> 0 needs a distinct local destination (cudaMemcpyAsync requires non-overlapping
    # src/dst); for it we measure intra-GPU HBM->HBM copy-engine bandwidth as a baseline.
    local_dst = torch.zeros(numel, dtype=DTYPE, device=device) if is_src else None

    modes = [(0, "copy-engine"), (1, "tma"), (2, "ld/st")]
    for mode_id, mode_name in modes:
        results = []  # (dst_rank, locality, avg_ms, gbps)
        for dst in range(world_size):
            dist.barrier()  # only rank 0 transfers, so the link under test is otherwise idle
            if is_src:
                dst_ptr = local_dst.data_ptr() if dst == SRC_RANK else hdl.buffer_ptrs[dst]
                avg_ms = benchmark(dst_ptr, hdl.buffer_ptrs[SRC_RANK], nbytes, WARMUP_ITERS, TIMED_ITERS, mode_id)
                gbps = (nbytes / GiB) / (avg_ms / 1000.0)
                same_node = (dst // gpus_per_node) == (SRC_RANK // gpus_per_node)
                locality = "self" if dst == SRC_RANK else ("intra-node" if same_node else "inter-node")
                results.append((dst, locality, avg_ms, gbps))
            dist.barrier()

        if rank == 0:
            print(f"\n=== mode: {mode_name} ===")
            print("dst   node  locality     time(ms)     BW(GB/s)")
            print("----  ----  -----------  -----------  ---------")
            for dst, locality, avg_ms, gbps in results:
                print(f"{dst:>4}  {dst // gpus_per_node:>4}  {locality:<11}  {avg_ms:>11.3f}  {gbps:>9.2f}")
            print("", flush=True)

    dist.barrier()
    del hdl  # the handle must be released or process-group teardown can hang
    gc.collect()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
