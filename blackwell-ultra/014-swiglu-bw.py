"""
To run:
    make
    python3 014-swiglu-bw.py
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _C import swiglu


NUM_LOCAL_TOKENS = 7168
INTERMEDIATE_DIM = 2048
TOPK = 4

WARMUP_ITERS = 5
TIMED_ITERS = 10


def main():
    total_tokens = NUM_LOCAL_TOKENS * TOPK

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    gen = torch.Generator(device=device).manual_seed(1234)
    a = torch.randn(total_tokens, INTERMEDIATE_DIM, generator=gen, device=device, dtype=torch.bfloat16)
    b = torch.randn(total_tokens, INTERMEDIATE_DIM, generator=gen, device=device, dtype=torch.bfloat16)
    c = torch.empty(total_tokens, INTERMEDIATE_DIM, device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize()

    for _ in range(WARMUP_ITERS):
        swiglu(a, b, c)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TIMED_ITERS):
        swiglu(a, b, c)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / TIMED_ITERS

    bytes_moved = 3 * total_tokens * INTERMEDIATE_DIM * 2
    print(f"{ms:.4f} ms   {(bytes_moved / 1024 ** 3) / (ms / 1000):.1f} GB/s", flush=True)

    c_ref = (F.silu(a.float()) * b.float()).to(torch.bfloat16)
    diff = (c.float() - c_ref.float()).abs()
    print(f"out  abs mean {c.abs().mean().item():.4f}   abs max {c.abs().max().item():.4f}")
    print(f"ref  abs mean {c_ref.abs().mean().item():.4f}   abs max {c_ref.abs().max().item():.4f}")
    print(f"diff abs mean {diff.mean().item():.6f}   abs max {diff.max().item():.6f}", flush=True)


if __name__ == "__main__":
    main()
