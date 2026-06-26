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

WARMUP_ITERS = 5
TIMED_ITERS = 10
GiB = 1024 ** 3


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=device)

    NUM_EXPERTS = NUM_LOCAL_EXPERTS * world_size
    CAPACITY = NUM_LOCAL_TOKENS * TOPK * 2 # Global recv capacit, 2x headroom over average

    # Generate inputs and buffers
    gen = torch.Generator(device=device).manual_seed(1234 + rank)
    router_logits = torch.randn(NUM_LOCAL_TOKENS, NUM_EXPERTS, generator=gen, device=device)
    tokens = symm_mem.empty(NUM_LOCAL_TOKENS, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    tokens.normal_(generator=gen)
    hdl = symm_mem.rendezvous(tokens, dist.group.WORLD.group_name)
    tokens_ptrs = [hdl.buffer_ptrs[i] for i in range(world_size)]
    recv2d = torch.zeros(CAPACITY, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        print("\nMoE Dispatch Bandwidth")
        print("===========================================================================")
        print(f"tokens/rank: {NUM_LOCAL_TOKENS} x {HIDDEN_DIM} bf16   experts: {NUM_EXPERTS} "
              f"({NUM_LOCAL_EXPERTS}/rank)   topk: {TOPK}")
        print(f"recv cap:    {CAPACITY} rows total")
        print(f"iters:       warmup {WARMUP_ITERS}, timed {TIMED_ITERS}\n", flush=True)

    # Router all-gather
    _, topk_ids = torch.topk(router_logits, TOPK, dim=1)
    topk_ids = topk_ids.to(torch.int32)
    topk_ids_all = torch.empty(world_size, NUM_LOCAL_TOKENS, TOPK, dtype=torch.int32, device=device)
    dist.all_gather_into_tensor(topk_ids_all, topk_ids)
    schedule_src_rank = torch.empty((CAPACITY,), dtype=torch.int32, device=device)
    schedule_src_token_idx = torch.empty((CAPACITY,), dtype=torch.int32, device=device)

    # Scheduler benchmark
    for _ in range(WARMUP_ITERS):
        schedule(topk_ids_all, schedule_src_rank, schedule_src_token_idx, rank, NUM_LOCAL_EXPERTS)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TIMED_ITERS):
        schedule(topk_ids_all, schedule_src_rank, schedule_src_token_idx, rank, NUM_LOCAL_EXPERTS)
    end.record()
    torch.cuda.synchronize()
    scheduler_ms = start.elapsed_time(end) / TIMED_ITERS

    # Dispatcher benchmark
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
    dispatcher_ms = start.elapsed_time(end) / TIMED_ITERS
    dist.barrier()

    # Correctness check
    tokens_all = torch.empty(world_size, NUM_LOCAL_TOKENS, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    dist.all_gather_into_tensor(tokens_all, tokens)
    valid = schedule_src_token_idx >= 0
    expected = torch.zeros_like(recv2d)
    expected[valid] = tokens_all[schedule_src_rank[valid], schedule_src_token_idx[valid]]
    errors = (recv2d != expected).sum()
    dist.all_reduce(errors, op=dist.ReduceOp.SUM)

    # Performance check
    num_tokens = int(valid.sum().item())
    num_local_tokens = int((valid & (schedule_src_rank == rank)).sum().item())
    num_remote_tokens = num_tokens - num_local_tokens
    bytes_total = num_tokens * HIDDEN_DIM * 2
    bytes_remote = num_remote_tokens * HIDDEN_DIM * 2
    stats = torch.tensor([
        num_tokens, num_local_tokens, num_remote_tokens, bytes_total, bytes_remote, dispatcher_ms, scheduler_ms
    ], dtype=torch.float64, device=device)
    stats_all = [torch.zeros_like(stats) for _ in range(world_size)]
    dist.all_gather(stats_all, stats)

    if rank == 0:
        print(f"correctness: {'PASSED' if int(errors.item()) == 0 else 'FAILED (' + str(int(errors.item())) + ' mismatches)'}\n")
        print("rank  tokens  local  remote   time(ms)  sched(ms)  total(GB/s)  remote(GB/s)")
        print("----  ------  -----  ------  ---------  ---------  -----------  ------------")
        max_seconds = 0.0
        sum_remote_bytes = 0.0
        sum_total_bytes = 0.0
        for rank, stats in enumerate(stats_all):
            nt, lt, rt, bt, br, dms, sms = stats.tolist()
            seconds = dms / 1000.0
            print(f"{rank:>4}  {int(nt):>6}  {int(lt):>5}  {int(rt):>6}  {dms:>9.3f}  {sms:>9.4f}  "
                  f"{(bt / GiB) / seconds:>11.2f}  {(br / GiB) / seconds:>12.2f}")
            max_seconds = max(max_seconds, seconds)
            sum_remote_bytes += br
            sum_total_bytes += bt
        print("----  ------  -----  ------  ---------  ---------  -----------  ------------")
        print(f"aggregate (sum bytes / slowest rank): "
              f"total {(sum_total_bytes / GiB) / max_seconds:.2f} GB/s, "
              f"remote {(sum_remote_bytes / GiB) / max_seconds:.2f} GB/s", flush=True)

    dist.barrier()
    del hdl
    gc.collect()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
