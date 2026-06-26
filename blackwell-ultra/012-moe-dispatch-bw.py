"""
To run:
    make
    torchrun --nproc_per_node=4 012-moe-dispatch-bw.py
"""

import gc
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _C import dispatch, schedule


NUM_LOCAL_TOKENS = 7168
HIDDEN_DIM = 7168
NUM_LOCAL_EXPERTS = 4
TOPK = 4
BLOCK_SIZE = 128

WARMUP_ITERS = 3
TIMED_ITERS = 20
GiB = 1024 ** 3


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=device)

    NUM_EXPERTS = NUM_LOCAL_EXPERTS * world_size
    # Global recv capacity: tokens arriving at this rank average ~NUM_LOCAL_TOKENS*TOPK (1/world_size of
    # all routes); 2x headroom. Multiple of BLOCK_SIZE for the dispatch row tiling.
    recv_rows = NUM_LOCAL_TOKENS * TOPK * 2

    # Per-rank routing (distinct per rank for a realistic, imbalanced dispatch).
    gen = torch.Generator(device=device).manual_seed(1234 + rank)
    router_logits = torch.randn(NUM_LOCAL_TOKENS, NUM_EXPERTS, generator=gen, device=device)

    if rank == 0:
        print("\nMoE Dispatch Bandwidth (PULL-based: destination reads remote tokens over NVLink)")
        print("===========================================================================")
        print(f"tokens/rank: {NUM_LOCAL_TOKENS} x {HIDDEN_DIM} bf16   experts: {NUM_EXPERTS} "
              f"({NUM_LOCAL_EXPERTS}/rank)   topk: {TOPK}")
        print(f"recv cap:    {recv_rows} rows total (local buffer, dense per-expert packing)")
        print(f"iters:       warmup {WARMUP_ITERS}, timed {TIMED_ITERS}\n", flush=True)

    # ----- routing -> all-gathered topk -> round-robin pull schedule (built in CUDA/C++) -----
    _, topk_ids = torch.topk(router_logits, TOPK, dim=1)
    topk_all = torch.empty(world_size, NUM_LOCAL_TOKENS, TOPK, dtype=torch.int32, device=device)
    dist.all_gather_into_tensor(topk_all, topk_ids.to(torch.int32))
    schedule_src_rank = torch.full((recv_rows,), -1, dtype=torch.int32, device=device)
    schedule_src_token_idx = torch.full((recv_rows,), -1, dtype=torch.int32, device=device)

    # ----- time the schedule-build kernel on its own (CUDA events; calls are idempotent) -----
    for _ in range(WARMUP_ITERS):
        schedule(topk_all, schedule_src_rank, schedule_src_token_idx, rank, NUM_LOCAL_EXPERTS)
    torch.cuda.synchronize()
    sched_start = torch.cuda.Event(enable_timing=True)
    sched_end = torch.cuda.Event(enable_timing=True)
    sched_start.record()
    for _ in range(TIMED_ITERS):
        schedule(topk_all, schedule_src_rank, schedule_src_token_idx, rank, NUM_LOCAL_EXPERTS)
    sched_end.record()
    torch.cuda.synchronize()
    sched_ms = sched_start.elapsed_time(sched_end) / TIMED_ITERS

    # ----- symmetric token buffer: this rank's tokens, remotely readable by every other rank -----
    tokens_flat = symm_mem.empty(NUM_LOCAL_TOKENS * HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    tokens = tokens_flat.view(NUM_LOCAL_TOKENS, HIDDEN_DIM)
    tokens.normal_(generator=gen)
    torch.cuda.synchronize()
    hdl = symm_mem.rendezvous(tokens_flat, dist.group.WORLD.group_name)
    tokens_ptrs = [hdl.buffer_ptrs[i] for i in range(world_size)]
    if rank == 0:
        print("[rank 0] symmetric memory rendezvous complete", flush=True)

    # ----- local destination recv buffer (zeroed so padding rows compare cleanly) -----
    recv2d = torch.zeros(recv_rows, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    dist.barrier()

    # ----- benchmark with torch CUDA events (warmup + timed, on the current stream) -----
    for _ in range(WARMUP_ITERS):
        dispatch(tokens, tokens_ptrs, recv2d, schedule_src_rank, schedule_src_token_idx)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    dist.barrier()  # release all ranks together so the dispatch is genuinely concurrent
    start.record()
    for _ in range(TIMED_ITERS):
        dispatch(tokens, tokens_ptrs, recv2d, schedule_src_rank, schedule_src_token_idx)
    end.record()
    torch.cuda.synchronize()
    avg_ms = start.elapsed_time(end) / TIMED_ITERS
    dist.barrier()

    # ----- correctness: recv row r must equal source token (src_rank[r], src_token_idx[r]) -----
    g_tokens = torch.empty(world_size, NUM_LOCAL_TOKENS, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    dist.all_gather_into_tensor(g_tokens, tokens)
    valid = schedule_src_token_idx >= 0
    expected = torch.zeros_like(recv2d)
    expected[valid] = g_tokens[schedule_src_rank[valid].to(torch.long),
                               schedule_src_token_idx[valid].to(torch.long)]
    mism = (recv2d != expected).sum()
    dist.all_reduce(mism, op=dist.ReduceOp.SUM)

    # ----- bandwidth bookkeeping (pull perspective): bytes this rank reads; remote = from other ranks -----
    num_tokens = int(valid.sum().item())
    local_tokens = int((valid & (schedule_src_rank == rank)).sum().item())
    remote_tokens = num_tokens - local_tokens
    bytes_total = num_tokens * HIDDEN_DIM * 2
    bytes_remote = remote_tokens * HIDDEN_DIM * 2
    stats = torch.tensor([num_tokens, local_tokens, remote_tokens,
                          bytes_total, bytes_remote, avg_ms, sched_ms], dtype=torch.float64, device=device)
    gathered_stats = [torch.zeros_like(stats) for _ in range(world_size)]
    dist.all_gather(gathered_stats, stats)

    if rank == 0:
        ok = int(mism.item()) == 0
        print(f"correctness: {'PASSED' if ok else 'FAILED (' + str(int(mism.item())) + ' mismatches)'}\n")
        print("rank  tokens  local  remote   time(ms)   total(GB/s)  remote(GB/s)")
        print("----  ------  -----  ------  ---------  -----------  ------------")
        max_t = 0.0
        sum_remote_bytes = 0.0
        sum_total_bytes = 0.0
        sched_times = []
        for r, s in enumerate(gathered_stats):
            nt, lt, rt, bt, br, ms, sm = s.tolist()
            sc = ms / 1000.0
            print(f"{r:>4}  {int(nt):>6}  {int(lt):>5}  {int(rt):>6}  {ms:>9.3f}  "
                  f"{(bt / GiB) / sc:>11.2f}  {(br / GiB) / sc:>12.2f}")
            max_t = max(max_t, sc)
            sum_remote_bytes += br
            sum_total_bytes += bt
            sched_times.append(sm)
        print("----  ------  -----  ------  ---------  -----------  ------------")
        print(f"aggregate (sum bytes / slowest rank): "
              f"total {(sum_total_bytes / GiB) / max_t:.2f} GB/s, "
              f"remote {(sum_remote_bytes / GiB) / max_t:.2f} GB/s")
        print(f"\nschedule-build kernel (separate, avg over {TIMED_ITERS} iters): "
              + "  ".join(f"r{r} {t:.4f}ms" for r, t in enumerate(sched_times))
              + f"   mean {sum(sched_times) / len(sched_times):.4f} ms", flush=True)

    dist.barrier()
    del hdl
    gc.collect()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
