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

Results:

=== mode: copy-engine ===
dst   node  locality     time(ms)     BW(GB/s)
----  ----  -----------  -----------  ---------
   0     0  self               3.105    3220.83
   1     0  intra-node        13.767     726.37
   2     0  intra-node        13.769     726.26
   3     0  intra-node        13.748     727.39
   4     1  inter-node        13.769     726.29
   5     1  inter-node        13.755     727.01
   6     1  inter-node        13.767     726.36
   7     1  inter-node        13.765     726.50
   8     2  inter-node        13.767     726.40
   9     2  inter-node        13.756     726.96
  10     2  inter-node        13.764     726.52
  11     2  inter-node        13.768     726.30
  12     3  inter-node        13.769     726.29
  13     3  inter-node        13.771     726.16
  14     3  inter-node        13.755     727.03
  15     3  inter-node        13.764     726.51
  16     4  inter-node        13.765     726.46
  17     4  inter-node        13.765     726.46
  18     4  inter-node        13.756     726.95
  19     4  inter-node        13.770     726.24
  20     5  inter-node        13.768     726.32
  21     5  inter-node        13.748     727.37
  22     5  inter-node        13.772     726.10
  23     5  inter-node        13.755     726.99
  24     6  inter-node        13.767     726.38
  25     6  inter-node        13.761     726.71
  26     6  inter-node        13.764     726.53
  27     6  inter-node        13.755     726.99
  28     7  inter-node        13.767     726.40
  29     7  inter-node        13.768     726.30
  30     7  inter-node        13.748     727.38
  31     7  inter-node        13.772     726.12
  32     8  inter-node        13.757     726.92
  33     8  inter-node        13.765     726.46
  34     8  inter-node        13.766     726.45
  35     8  inter-node        13.766     726.41
  36     9  inter-node        13.756     726.98
  37     9  inter-node        13.765     726.48
  38     9  inter-node        13.761     726.69
  39     9  inter-node        13.747     727.42
  40    10  inter-node        13.773     726.06
  41    10  inter-node        13.757     726.89
  42    10  inter-node        13.765     726.50
  43    10  inter-node        13.761     726.67
  44    11  inter-node        13.766     726.42
  45    11  inter-node        13.757     726.90
  46    11  inter-node        13.767     726.36
  47    11  inter-node        13.769     726.28
  48    12  inter-node        13.746     727.46
  49    12  inter-node        13.773     726.08
  50    12  inter-node        13.758     726.87
  51    12  inter-node        13.765     726.48
  52    13  inter-node        13.762     726.64
  53    13  inter-node        13.765     726.46
  54    13  inter-node        13.756     726.97
  55    13  inter-node        13.769     726.28
  56    14  inter-node        13.767     726.35
  57    14  inter-node        13.749     727.34
  58    14  inter-node        13.771     726.15
  59    14  inter-node        13.758     726.85
  60    15  inter-node        13.765     726.46
  61    15  inter-node        13.764     726.54
  62    15  inter-node        13.765     726.47
  63    15  inter-node        13.755     727.01

=== mode: tma ===
dst   node  locality     time(ms)     BW(GB/s)
----  ----  -----------  -----------  ---------
   0     0  self               2.997    3336.36
   1     0  intra-node        14.939     669.39
   2     0  intra-node        14.938     669.42
   3     0  intra-node        14.939     669.39
   4     1  inter-node        14.938     669.42
   5     1  inter-node        14.938     669.42
   6     1  inter-node        14.938     669.42
   7     1  inter-node        14.938     669.43
   8     2  inter-node        14.938     669.43
   9     2  inter-node        14.938     669.41
  10     2  inter-node        14.938     669.44
  11     2  inter-node        14.939     669.39
  12     3  inter-node        14.939     669.39
  13     3  inter-node        14.939     669.41
  14     3  inter-node        14.938     669.43
  15     3  inter-node        14.938     669.43
  16     4  inter-node        14.939     669.39
  17     4  inter-node        14.938     669.42
  18     4  inter-node        14.939     669.39
  19     4  inter-node        14.938     669.43
  20     5  inter-node        14.939     669.40
  21     5  inter-node        14.939     669.39
  22     5  inter-node        14.939     669.37
  23     5  inter-node        14.938     669.43
  24     6  inter-node        14.938     669.44
  25     6  inter-node        14.940     669.34
  26     6  inter-node        14.940     669.34
  27     6  inter-node        14.940     669.34
  28     7  inter-node        14.939     669.37
  29     7  inter-node        14.940     669.33
  30     7  inter-node        14.940     669.35
  31     7  inter-node        14.940     669.34
  32     8  inter-node        14.940     669.36
  33     8  inter-node        14.940     669.34
  34     8  inter-node        14.940     669.33
  35     8  inter-node        14.940     669.35
  36     9  inter-node        14.940     669.35
  37     9  inter-node        14.941     669.29
  38     9  inter-node        14.940     669.35
  39     9  inter-node        14.940     669.36
  40    10  inter-node        14.940     669.36
  41    10  inter-node        14.940     669.32
  42    10  inter-node        14.940     669.33
  43    10  inter-node        14.940     669.33
  44    11  inter-node        14.940     669.36
  45    11  inter-node        14.940     669.36
  46    11  inter-node        14.941     669.32
  47    11  inter-node        14.939     669.37
  48    12  inter-node        14.939     669.38
  49    12  inter-node        14.939     669.39
  50    12  inter-node        14.939     669.39
  51    12  inter-node        14.938     669.43
  52    13  inter-node        14.939     669.40
  53    13  inter-node        14.939     669.39
  54    13  inter-node        14.939     669.39
  55    13  inter-node        14.938     669.45
  56    14  inter-node        14.938     669.43
  57    14  inter-node        14.939     669.39
  58    14  inter-node        14.939     669.40
  59    14  inter-node        14.938     669.45
  60    15  inter-node        14.938     669.41
  61    15  inter-node        14.939     669.37
  62    15  inter-node        14.939     669.40
  63    15  inter-node        14.938     669.42

=== mode: ld/st ===
dst   node  locality     time(ms)     BW(GB/s)
----  ----  -----------  -----------  ---------
   0     0  self               5.083    1967.17
   1     0  intra-node        18.206     549.26
   2     0  intra-node        18.212     549.07
   3     0  intra-node        18.259     547.68
   4     1  inter-node        18.163     550.56
   5     1  inter-node        18.290     546.74
   6     1  inter-node        18.174     550.24
   7     1  inter-node        18.223     548.75
   8     2  inter-node        18.255     547.79
   9     2  inter-node        18.218     548.92
  10     2  inter-node        18.234     548.42
  11     2  inter-node        18.218     548.91
  12     3  inter-node        18.158     550.73
  13     3  inter-node        18.212     549.08
  14     3  inter-node        18.195     549.61
  15     3  inter-node        18.222     548.80
  16     4  inter-node        18.227     548.64
  17     4  inter-node        18.229     548.57
  18     4  inter-node        18.220     548.83
  19     4  inter-node        18.301     546.42
  20     5  inter-node        18.301     546.42
  21     5  inter-node        18.199     549.47
  22     5  inter-node        18.287     546.85
  23     5  inter-node        18.179     550.08
  24     6  inter-node        18.299     546.48
  25     6  inter-node        18.250     547.93
  26     6  inter-node        18.336     545.39
  27     6  inter-node        18.197     549.53
  28     7  inter-node        18.154     550.83
  29     7  inter-node        18.289     546.77
  30     7  inter-node        18.248     547.99
  31     7  inter-node        18.234     548.42
  32     8  inter-node        18.235     548.40
  33     8  inter-node        18.156     550.79
  34     8  inter-node        18.248     547.99
  35     8  inter-node        18.256     547.78
  36     9  inter-node        18.210     549.15
  37     9  inter-node        18.238     548.30
  38     9  inter-node        18.245     548.09
  39     9  inter-node        18.238     548.30
  40    10  inter-node        18.262     547.59
  41    10  inter-node        18.245     548.10
  42    10  inter-node        18.242     548.19
  43    10  inter-node        18.201     549.43
  44    11  inter-node        18.139     551.29
  45    11  inter-node        18.225     548.70
  46    11  inter-node        18.156     550.78
  47    11  inter-node        18.256     547.76
  48    12  inter-node        18.231     548.52
  49    12  inter-node        18.269     547.38
  50    12  inter-node        18.235     548.40
  51    12  inter-node        18.197     549.54
  52    13  inter-node        18.244     548.12
  53    13  inter-node        18.207     549.23
  54    13  inter-node        18.276     547.15
  55    13  inter-node        18.192     549.71
  56    14  inter-node        18.246     548.07
  57    14  inter-node        18.222     548.78
  58    14  inter-node        18.237     548.33
  59    14  inter-node        18.290     546.75
  60    15  inter-node        18.155     550.80
  61    15  inter-node        18.212     549.09
  62    15  inter-node        18.150     550.97
  63    15  inter-node        18.223     548.76

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
