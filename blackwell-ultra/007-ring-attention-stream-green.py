"""
Ring attention with green-context SM partitioning for compute vs comm.
Note that both NCCL and FlashAttention ignore the green context. So this is more of a demonstration

Run on 4 GPUs:
    CUDA_VISIBLE_DEVICES=0,1,2,3 ~/anytitan/.venv/bin/torchrun --nproc_per_node=4 007-ring-attention-stream-green.py
"""

import importlib.util
import os
import pathlib

# Allocate SMs for NCCL / FlashAttention
_NCCL_SMS = 16
_FLASH_ATTN_SMS = 148 - _NCCL_SMS
os.environ["NCCL_MAX_CTAS"] = str(_NCCL_SMS)
os.environ["NCCL_MIN_CTAS"] = str(_NCCL_SMS)
import cutlass.utils as _cutlass_utils
_cutlass_utils.HardwareInfo.get_device_multiprocessor_count = lambda self: _FLASH_ATTN_SMS

import torch
from torch.cuda.green_contexts import GreenContext

_spec = importlib.util.spec_from_file_location("ring_naive", pathlib.Path(__file__).with_name("004-ring-attention-naive.py"))
naive = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(naive)

flash_block = naive.flash_block
combine = naive.combine
ring_exchange = naive.ring_exchange


def _green_streams():
    if not hasattr(ring_attention, "_green"):
        dev = torch.cuda.current_device()
        gc_flash = GreenContext.create(_FLASH_ATTN_SMS, dev)
        gc_flash.set_context()
        flash_stream = torch.cuda.Stream()
        gc_flash.pop_context()
        gc_comm = GreenContext.create(_NCCL_SMS, dev)
        gc_comm.set_context()
        comm_stream = torch.cuda.Stream()
        gc_comm.pop_context()
        ring_attention._green = (gc_flash, gc_comm, flash_stream, comm_stream)  # must keep gcs alive
    return ring_attention._green


def ring_attention(q, k, v, rank, world_size):
    """Ring attention with flash on a green-ctx compute stream and NCCL on a green-ctx comm stream."""
    _, _, compute_stream, comm_stream = _green_streams()

    # compute/comm streams must see the caller's inputs + buffer allocations (default stream).
    default_stream = torch.cuda.current_stream()
    compute_stream.wait_stream(default_stream)
    comm_stream.wait_stream(default_stream)

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

        with torch.cuda.stream(compute_stream):
            o, lse = flash_block(q, cur_k, cur_v)
            if acc_o is None:
                acc_o, acc_lse = o.float(), lse
            else:
                acc_o, acc_lse = combine(acc_o, acc_lse, o, lse)
        compute_done.record(compute_stream)

    default_stream.wait_stream(compute_stream)
    return acc_o.to(torch.bfloat16)


if __name__ == "__main__":
    naive.run(ring_attention, "green")
