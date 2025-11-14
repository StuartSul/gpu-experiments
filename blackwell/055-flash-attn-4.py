import time
import sys

import numpy as np
import torch
torch.random.manual_seed(42)
torch.set_printoptions(sci_mode=False)

import cuda.bindings.driver as cuda
import cutlass
from cutlass.cute.runtime import from_dlpack
from flash_attn.cute.interface import FlashAttnFunc
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100


# Global parameters
B = 1
N = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
H = 128
D_qk = 192
D_vo = 128
print(f"N: {N}, H: {H}, D_qk: {D_qk}, D_vo: {D_vo}")

# Config
CHECK_CORRECTNESS = False
BENCHMARK = True
NUM_WARMUPS = 5
NUM_ITERS = 10

# Generate inputs
Q = torch.randn(B, N, H, D_qk, dtype=torch.bfloat16, device="cuda")
K = torch.randn(B, N, H, D_qk, dtype=torch.bfloat16, device="cuda")
V = torch.randn(B, N, H, D_vo, dtype=torch.bfloat16, device="cuda")
softmax_scale = 1 / (D_qk ** 0.5)


def pytorch_attn_fwd(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        S = Q.permute(0, 2, 1, 3) @ K.permute(0, 2, 3, 1) * softmax_scale
        mask = torch.triu(torch.ones(N, N, device=S.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float('-inf'))
        P = S.softmax(dim=-1)
        O = P @ V.permute(0, 2, 1, 3)
        return O.permute(0, 2, 1, 3)


def flash_attn_fwd(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        O, _ = FlashAttnFunc.apply(Q, K, V, softmax_scale, True) # always causal
        return O


compile_funcs = {}
def flash_attn_fwd_raw(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    softmax_scale: float,
    O: torch.Tensor | None = None,
    L: torch.Tensor | None = None
) -> torch.Tensor:
    B, N, H, D_qk = Q.shape
    _, _, _, D_vo = V.shape

    if O is None:
        O = torch.empty(B, N, H, D_vo, dtype=Q.dtype, device=Q.device)
    if L is None:
        L = torch.empty(B, H, N, dtype=torch.float32, device=Q.device)

    Q_cute = from_dlpack(Q.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=Q.ndim - 1)
    K_cute = from_dlpack(K.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=K.ndim - 1)
    V_cute = from_dlpack(V.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=V.ndim - 1)
    O_cute = from_dlpack(O.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=O.ndim - 1)
    L_cute = from_dlpack(L.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=L.ndim - 1)

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Arbitrary key to use the global dict
    if "FA4" not in compile_funcs:
        _fa_fwd = FlashAttentionForwardSm100(
            D_qk,
            D_vo,
            qhead_per_kvhead=1,
            is_causal=True,
            is_local=False,
            pack_gqa=False,
            is_persistent=False # always false if causal
        )
        compile_funcs["FA4"] = cutlass.cute.compile(
            _fa_fwd, Q_cute, K_cute, V_cute, O_cute, L_cute, softmax_scale, current_stream,
            None, None, None, None, None, None, None, None, None
        )

    compile_funcs["FA4"](
        Q_cute, K_cute, V_cute, O_cute, L_cute, softmax_scale, current_stream,
        None, None, None, None, None, None, None, None, None
    )

    return O, L


# Check correctness
if CHECK_CORRECTNESS:
    # Diff checking utility
    def check_diff(name, A, A_ref):
        print(f"\n{name}")
        print(f"Max diff:  {((A - A_ref).abs().max().item()):.10f}")
        print(f"Mean diff: {((A - A_ref).abs().mean().item()):.10f}")
        print(f"Mean:      {A.abs().mean().item():.10f}")
        print(f"Ref mean:  {A_ref.abs().mean().item():.10f}")
        print(f"Max:       {A.abs().max().item():.10f}")
        print(f"Ref max:   {A_ref.abs().max().item():.10f}")

    # Flash MHA
    O_flash = flash_attn_fwd(Q, K, V)

    # Raw Flash MHA
    O_flash_raw, L_flash_raw = flash_attn_fwd_raw(Q, K, V, softmax_scale)

    # Pytorch MHA
    O_ref = pytorch_attn_fwd(Q, K, V)

    # Check results
    check_diff("O (flash)", O_flash, O_ref)
    check_diff("O (flash_raw)", O_flash_raw, O_ref)

# Benchmark
if BENCHMARK:
    l2_cache = torch.empty(1024 * 1024 * 64, dtype=torch.bfloat16, device="cuda")

    for i in range(NUM_WARMUPS):
        flash_attn_fwd(Q, K, V)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    times = []

    for i in range(NUM_ITERS):
        l2_cache.random_()
        start_event.record()
        flash_attn_fwd(Q, K, V)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event) * 1e-3)

    avg_time = np.mean(times)
    std_time = np.std(times)
    flops = 2 * B * H * (N * (N + 1) / 2) * (D_qk + D_vo)
    tflops = flops * 1e-12

    print(f'Time taken: {avg_time * 1e6:.2f} ± {std_time * 1e6:.2f} us')
    print(f'TFLOPS: {tflops / avg_time:.2f} TFLOP/s')

    for i in range(NUM_WARMUPS):
        flash_attn_fwd_raw(Q, K, V, softmax_scale)
    torch.cuda.synchronize()

    times = []

    for i in range(NUM_ITERS):
        l2_cache.random_()
        start_event.record()
        flash_attn_fwd_raw(Q, K, V, softmax_scale)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event) * 1e-3)

    avg_time = np.mean(times)
    std_time = np.std(times)
    flops = 2 * B * H * (N * (N + 1) / 2) * (D_qk + D_vo)
    tflops = flops * 1e-12

    print(f'Time taken (raw): {avg_time * 1e6:.2f} ± {std_time * 1e6:.2f} us')
    print(f'TFLOPS: {tflops / avg_time:.2f} TFLOP/s')

# To avoid unload_cubin_module on None
del compile_funcs
