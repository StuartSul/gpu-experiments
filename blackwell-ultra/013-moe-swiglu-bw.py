"""
To run:
    make
    python3 013-moe-swiglu-bw.py
"""

import torch
import torch.nn.functional as F


NUM_LOCAL_TOKENS = 7168
HIDDEN_DIM = 7168
INTERMEDIATE_DIM = 2048
NUM_LOCAL_EXPERTS = 4
TOPK = 4

WARMUP_ITERS = 5
TIMED_ITERS = 10


def swiglu_ref(x, w_gate, w_up, w_down, tokens_per_expert):
    """bf16 grouped SwiGLU reference.

    x:                 (total_tokens, HIDDEN_DIM)  bf16, tokens grouped contiguously by expert
    w_gate / w_up:     (E, HIDDEN_DIM, I)          bf16
    w_down:            (E, I, HIDDEN_DIM)          bf16
    tokens_per_expert: (E,)                         int, per-expert tokens
    returns:           (total_tokens, HIDDEN_DIM)  bf16
    """
    y = torch.empty_like(x)
    offset = 0
    for expert_idx, num_tokens in enumerate(tokens_per_expert.tolist()):
        _x = x[offset:offset + num_tokens]                        # (num_tokens, H)
        gate = _x @ w_gate[expert_idx]                            # (num_tokens, I)
        up = _x @ w_up[expert_idx]                                # (num_tokens, I)
        hid = (F.silu(gate.float()) * up.float()).to(x.dtype)     # (num_tokens, I)
        y[offset:offset + num_tokens] = hid @ w_down[expert_idx]  # (num_tokens, H)
        offset += num_tokens
    return y


def main():
    E, H, I = NUM_LOCAL_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Scheduler output
    tokens_per_expert = torch.full((E,), NUM_LOCAL_TOKENS * TOPK // E, dtype=torch.int32, device=device)
    total_tokens = int(tokens_per_expert.sum())

    # Generate inputs
    gen    = torch.Generator(device=device).manual_seed(1234)
    x      = torch.randn(total_tokens, H, generator=gen, device=device, dtype=torch.bfloat16)
    w_gate = torch.randn(E, H, I, generator=gen, device=device, dtype=torch.bfloat16) * H ** -0.5
    w_up   = torch.randn(E, H, I, generator=gen, device=device, dtype=torch.bfloat16) * H ** -0.5
    w_down = torch.randn(E, I, H, generator=gen, device=device, dtype=torch.bfloat16) * I ** -0.5
    torch.cuda.synchronize()

    # Benchmark reference
    for _ in range(WARMUP_ITERS):
        y = swiglu_ref(x, w_gate, w_up, w_down, tokens_per_expert)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TIMED_ITERS):
        y = swiglu_ref(x, w_gate, w_up, w_down, tokens_per_expert)
    end.record()
    torch.cuda.synchronize()
    ref_ms = start.elapsed_time(end) / TIMED_ITERS

    flops = 6 * total_tokens * H * I

    print("\nMoE SwiGLU (single-GPU expert FFN, post-dispatch / pre-combine)")
    print("===========================================================================")
    print(f"experts: {E}   tokens/expert: {total_tokens // E} ({total_tokens} total)   "
          f"H: {H}   I: {I}   topk: {TOPK}   bf16")
    print(f"compute: {flops / 1e12:.3f} TFLOP")
    print(f"iters:   warmup {WARMUP_ITERS}, timed {TIMED_ITERS}\n", flush=True)

    print("impl         ms      TFLOP/s")
    print("---------  -------  ---------")
    print(f"bf16 ref   {ref_ms:>7.3f}  {flops / 1e9 / ref_ms:>9.1f}", flush=True)


if __name__ == "__main__":
    main()
