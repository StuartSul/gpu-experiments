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

    # Per-rank inputs (distinct routing per rank for a realistic, imbalanced dispatch).
    gen = torch.Generator(device=device).manual_seed(1234 + rank)
    tokens = torch.randn(NUM_LOCAL_TOKENS, HIDDEN_DIM, generator=gen, device=device, dtype=torch.bfloat16)
    router_logits = torch.randn(NUM_LOCAL_TOKENS, NUM_EXPERTS, generator=gen, device=device)

    SEND_BUFFER_CAPACITY = (NUM_LOCAL_TOKENS * TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
    RECV_BUFFER_CAPACITY = NUM_LOCAL_TOKENS * TOPK * 2 // BLOCK_SIZE
    recv_rows = NUM_LOCAL_EXPERTS * RECV_BUFFER_CAPACITY * BLOCK_SIZE

    if rank == 0:
        print("\nMoE Dispatch Bandwidth (gather-and-dispatch, TMA vector gather + vector store)")
        print("===========================================================================")
        print(f"tokens/rank: {NUM_LOCAL_TOKENS} x {HIDDEN_DIM} bf16   experts: {NUM_EXPERTS} "
              f"({NUM_LOCAL_EXPERTS}/rank)   topk: {TOPK}")
        print(f"send cap:    {SEND_BUFFER_CAPACITY} blocks ({SEND_BUFFER_CAPACITY * BLOCK_SIZE} rows)")
        print(f"recv cap:    {RECV_BUFFER_CAPACITY} blocks/expert ({recv_rows} rows total)")
        print(f"iters:       warmup {WARMUP_ITERS}, timed {TIMED_ITERS}\n", flush=True)

    # ----- schedule: build the dispatch index metadata on-device from the all-gathered routing -----
    # We only all-gather the (tiny) router result; the schedule kernel turns it into the per-slot
    # gather/scatter indices. No token data is touched on the PyTorch side.
    _, topk_ids = torch.topk(router_logits, TOPK, dim=1)             # (num_local_tokens, topk) expert ids
    topk_all = torch.empty(world_size, NUM_LOCAL_TOKENS, TOPK, dtype=torch.int32, device=device)
    dist.all_gather_into_tensor(topk_all, topk_ids.to(torch.int32))

    schedule_len = SEND_BUFFER_CAPACITY * BLOCK_SIZE
    schedule_src_token_idx = torch.empty(schedule_len, dtype=torch.int32, device=device)
    schedule_dst_rank = torch.empty(schedule_len, dtype=torch.int32, device=device)
    schedule_dst_token_idx = torch.empty(schedule_len, dtype=torch.int32, device=device)
    schedule(topk_all, schedule_src_token_idx, schedule_dst_rank, schedule_dst_token_idx,
             rank, NUM_LOCAL_EXPERTS, RECV_BUFFER_CAPACITY)

    # Symmetric recv buffer (NVLink-addressable across the fabric); tokens stay local.
    recv = symm_mem.empty(recv_rows * HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    recv.zero_()
    torch.cuda.synchronize()
    hdl = symm_mem.rendezvous(recv, dist.group.WORLD.group_name)
    if rank == 0:
        print("[rank 0] symmetric memory rendezvous complete", flush=True)
    dist.barrier()

    recv2d = recv.view(recv_rows, HIDDEN_DIM)
    recv_ptrs = [hdl.buffer_ptrs[i] for i in range(world_size)]

    # ----- benchmark with torch CUDA events (warmup + timed, on the current stream) -----
    for _ in range(WARMUP_ITERS):
        dispatch(tokens, recv2d, recv_ptrs, schedule_src_token_idx, schedule_dst_rank, schedule_dst_token_idx)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    dist.barrier()  # release all ranks together so the dispatch is genuinely concurrent
    start.record()
    for _ in range(TIMED_ITERS):
        dispatch(tokens, recv2d, recv_ptrs, schedule_src_token_idx, schedule_dst_rank, schedule_dst_token_idx)
    end.record()
    torch.cuda.synchronize()
    avg_ms = start.elapsed_time(end) / TIMED_ITERS
    dist.barrier()

    # ----- correctness: replay every rank's schedule against the gathered token rows -----
    # send_ref[s] mirrors the kernel's gather (tokens[src], zeros for padding); each rank's schedule
    # then says where each row lands, so we just scatter the rows targeting this rank.
    has_token = schedule_src_token_idx >= 0
    send_ref = torch.zeros(schedule_len, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    send_ref[has_token] = tokens[schedule_src_token_idx[has_token].to(torch.long)]

    g_send = torch.empty(world_size, schedule_len, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    dist.all_gather_into_tensor(g_send, send_ref)
    g_dst_rank = torch.empty(world_size, schedule_len, dtype=torch.int32, device=device)
    dist.all_gather_into_tensor(g_dst_rank, schedule_dst_rank)
    g_dst_token = torch.empty(world_size, schedule_len, dtype=torch.int32, device=device)
    dist.all_gather_into_tensor(g_dst_token, schedule_dst_token_idx)

    expected = torch.zeros_like(recv2d)
    for s in range(world_size):
        sel = g_dst_rank[s] == rank
        expected[g_dst_token[s][sel].to(torch.long)] = g_send[s][sel]

    mism = (recv2d != expected).sum()
    dist.all_reduce(mism, op=dist.ReduceOp.SUM)

    # ----- bandwidth bookkeeping (post-timing). The vector store moves exactly the real tokens (no
    # block padding), so we account for tokens routed, not padded blocks. -----
    num_tokens = int(has_token.sum().item())                         # = NUM_LOCAL_TOKENS * TOPK
    local_tokens = int((has_token & (schedule_dst_rank == rank)).sum().item())
    remote_tokens = num_tokens - local_tokens
    bytes_total = num_tokens * HIDDEN_DIM * 2
    bytes_remote = remote_tokens * HIDDEN_DIM * 2
    sec = avg_ms / 1000.0
    stats = torch.tensor([num_tokens, local_tokens, remote_tokens,
                          bytes_total, bytes_remote, avg_ms], dtype=torch.float64, device=device)
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
        for r, s in enumerate(gathered_stats):
            nt, lt, rt, bt, br, ms = s.tolist()
            sc = ms / 1000.0
            print(f"{r:>4}  {int(nt):>6}  {int(lt):>5}  {int(rt):>6}  {ms:>9.3f}  "
                  f"{(bt / GiB) / sc:>11.2f}  {(br / GiB) / sc:>12.2f}")
            max_t = max(max_t, sc)
            sum_remote_bytes += br
            sum_total_bytes += bt
        print("----  ------  -----  ------  ---------  -----------  ------------")
        print(f"aggregate (sum bytes / slowest rank): "
              f"total {(sum_total_bytes / GiB) / max_t:.2f} GB/s, "
              f"remote {(sum_remote_bytes / GiB) / max_t:.2f} GB/s", flush=True)

    dist.barrier()
    del hdl
    gc.collect()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
