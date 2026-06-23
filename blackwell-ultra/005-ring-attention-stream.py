"""
Stream-overlapped ring attention. Reuses the shared harness + helpers from
004-ring-attention-naive.py; only the ring_attention implementation differs.

Run on 4 GPUs:
    CUDA_VISIBLE_DEVICES=0,1,2,3 ~/anytitan/.venv/bin/torchrun --nproc_per_node=4 005-ring-attention-stream.py
"""

import importlib.util
import pathlib

import torch

_spec = importlib.util.spec_from_file_location("ring_naive", pathlib.Path(__file__).with_name("004-ring-attention-naive.py"))
naive = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(naive)

flash_block = naive.flash_block
combine = naive.combine
ring_exchange = naive.ring_exchange


def ring_attention(q, k, v, rank, world_size):
    """Stream-overlapped ring attention; q, k, v are this rank's shards (B, L, H, D).

    The next K/V shard is prefetched on a dedicated comm stream while the current
    shard's flash attention runs on the compute stream, hiding the ring exchange
    behind compute. Receives use a double buffer so we never clobber the caller's
    k/v nor a shard still being read by the previous step's flash.
    """
    if not hasattr(ring_attention, "_comm_stream"):
        ring_attention._comm_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.current_stream()
    comm_stream = ring_attention._comm_stream

    compute_done = torch.cuda.Event()
    comm_done = torch.cuda.Event()

    # Double-buffered receive scratch
    buf_k = [torch.empty_like(k), torch.empty_like(k)]
    buf_v = [torch.empty_like(v), torch.empty_like(v)]

    acc_o = acc_lse = None
    cur_k, cur_v = k, v

    for step in range(world_size):
        if step > 0:
            compute_stream.wait_event(comm_done)
            cur_k, cur_v = buf_k[(step - 1) % 2], buf_v[(step - 1) % 2]

        if step < world_size - 1:
            if step > 0:
                comm_stream.wait_event(compute_done)
            with torch.cuda.stream(comm_stream):
                ring_exchange([cur_k, cur_v], rank, world_size, recv=[buf_k[step % 2], buf_v[step % 2]])
            comm_done.record(comm_stream)

        o, lse = flash_block(q, cur_k, cur_v)
        if acc_o is None:
            acc_o, acc_lse = o.float(), lse
        else:
            acc_o, acc_lse = combine(acc_o, acc_lse, o, lse)
        compute_done.record(compute_stream)

    return acc_o.to(torch.bfloat16)


if __name__ == "__main__":
    naive.run(ring_attention, "stream")
