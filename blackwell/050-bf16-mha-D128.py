import numpy as np
import torch
torch.manual_seed(42)
torch.set_printoptions(sci_mode=False)

# Import our Python bindings
from _C import bf16_mha_fwd, bf16_mha_bwd_prep, bf16_mha_bwd


def check_diff(name, A, A_ref):
    print(f"\n{name}")
    print(f"Max diff:  {((A - A_ref).abs().max().item()):.10f}")
    print(f"Mean diff: {((A - A_ref).abs().mean().item()):.10f}")
    print(f"Mean:      {A.abs().mean().item():.10f}")
    print(f"Ref mean:  {A_ref.abs().mean().item():.10f}")
    print(f"Max:       {A.abs().max().item():.10f}")
    print(f"Ref max:   {A_ref.abs().max().item():.10f}")


# Input dimensions
B = 1
N = 1024
H = 4
D = 128

# Flags
CHECK_CORRECTNESS = True
CHECK_TORCH_REFERENCE = True
BENCHMARK = False
BENCHMARK_FA_SM100 = False

# Input tensors
print('Generating inputs...')
Q =      torch.randn((B, H, N, D), dtype=torch.bfloat16, device="cuda")
K =      torch.randn((B, H, N, D), dtype=torch.bfloat16, device="cuda")
V =      torch.randn((B, H, N, D), dtype=torch.bfloat16, device="cuda")
L =      torch.zeros((B, H, 1, N), dtype=torch.float,    device="cuda")
O =      torch.zeros((B, H, N, D), dtype=torch.bfloat16, device="cuda")
D_vec =  torch.empty((B, H, 1, N), dtype=torch.float,    device="cuda")
Q_grad = torch.zeros_like(Q,       dtype=torch.float,    device="cuda")
K_grad = torch.zeros_like(K,       dtype=torch.float,    device="cuda")
V_grad = torch.zeros_like(V,       dtype=torch.float,    device="cuda")
O_grad = torch.ones_like(O,        dtype=torch.bfloat16, device="cuda")

if CHECK_CORRECTNESS:
    # Run forward kernel
    print("\nRunning forward kernel...")
    bf16_mha_fwd(Q, K, V, L, O)
    torch.cuda.synchronize()

    # Run backward kernel
    print("\nRunning backward kernel...")
    bf16_mha_bwd_prep(O_grad, O, D_vec)
    bf16_mha_bwd(Q, K, V, O_grad, Q_grad, K_grad, V_grad, L, D_vec)
    torch.cuda.synchronize()

    # Run forward reference
    Q.requires_grad = True
    K.requires_grad = True
    V.requires_grad = True
    S = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)
    M = S.max(dim=-1, keepdim=True).values
    P = torch.softmax(S - M, dim=-1)
    O_ref = torch.matmul(P, V)
    L_ref = (torch.log(torch.sum(torch.exp(S - M), dim=-1, keepdim=True)) + M).transpose(-2, -1)

    # Check forward pass
    check_diff("O", O, O_ref)
    check_diff("L", L, L_ref * (-(D ** 0.5))) # forward kernel performs additional scaling

    # Run backward reference
    O_ref.backward(O_grad)
    Q_grad_ref = Q.grad.detach()
    K_grad_ref = K.grad.detach()
    V_grad_ref = V.grad.detach()
    D_vec_ref = (O * O_grad).sum(dim=-1).unsqueeze(2)

    if CHECK_TORCH_REFERENCE:
        # Run backward Pytorch
        V_grad_torch = torch.matmul(P.transpose(-2, -1), O_grad)              # dV = P^T @ dO
        P_grad = torch.matmul(O_grad, V.transpose(-2, -1))                    # dP = dO @ V^T
        S_grad = (P_grad * P - P * (P_grad * P).sum(dim=-1, keepdim=True))    # dS = (diag(P) - P @ P^T) @ dP
        Q_grad_torch = torch.matmul(S_grad, K) / (D ** 0.5)                   # dQ = dS @ K
        K_grad_torch = torch.matmul(S_grad.transpose(-2, -1), Q) / (D ** 0.5) # dK = dS^T @ Q
        check_diff("Q_grad_torch", Q_grad_torch, Q_grad_ref)
        check_diff("K_grad_torch", K_grad_torch, K_grad_ref)
        check_diff("V_grad_torch", V_grad_torch, V_grad_ref)

        # Run backward Pytorch with FA recipe
        BLOCK_SIZE = 64
        NUM_BLOCKS = N // BLOCK_SIZE
        Q_grad_torch_fa = torch.zeros_like(Q, dtype=torch.float, device="cuda")
        K_grad_torch_fa = torch.zeros_like(K, dtype=torch.float, device="cuda")
        V_grad_torch_fa = torch.zeros_like(V, dtype=torch.float, device="cuda")
        for j in range(NUM_BLOCKS):
            KV_start = j * BLOCK_SIZE
            KV_end = (j + 1) * BLOCK_SIZE
            for i in range(NUM_BLOCKS):
                QO_start = i * BLOCK_SIZE
                QO_end = (i + 1) * BLOCK_SIZE
                S_ij = Q[:, :, QO_start:QO_end, :] @ K[:, :, KV_start:KV_end, ].transpose(-1, -2) / (D ** 0.5)
                P_grad_ij = O_grad[:, :, QO_start:QO_end, :] @ V[:, :, KV_start:KV_end, :].transpose(-1, -2)
                P_ij = torch.exp(S_ij.to(torch.float32) - L_ref[:, :, 0, QO_start:QO_end].unsqueeze(-1))
                S_grad_ij = P_ij * (P_grad_ij.to(torch.float32) - D_vec_ref[:, :, 0, QO_start:QO_end].unsqueeze(-1)) / (D ** 0.5)
                V_grad_torch_fa[:, :, KV_start:KV_end, :] += P_ij.to(torch.bfloat16).transpose(-1, -2) @ O_grad[:, :, QO_start:QO_end, :]
                K_grad_torch_fa[:, :, KV_start:KV_end, :] += S_grad_ij.to(torch.bfloat16).transpose(-1, -2) @ Q[:, :, QO_start:QO_end, :]
                Q_grad_torch_fa[:, :, QO_start:QO_end, :] += S_grad_ij.to(torch.bfloat16) @ K[:, :, KV_start:KV_end, :]
        check_diff("Q_grad_torch_fa", Q_grad_torch_fa, Q_grad_ref)
        check_diff("K_grad_torch_fa", K_grad_torch_fa, K_grad_ref)
        check_diff("V_grad_torch_fa", V_grad_torch_fa, V_grad_ref)

    # Check backward pass
    check_diff("D_vec", D_vec, D_vec_ref)
    check_diff("Q_grad", Q_grad, Q_grad_ref)
    check_diff("K_grad", K_grad, K_grad_ref)
    check_diff("V_grad", V_grad, V_grad_ref)

if BENCHMARK:
    NUM_WARMUPS = 5
    NUM_ITERS = 10

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_ITERS)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_ITERS)]

    print("\nBenchmarking forward pass...")

    for i in range(NUM_WARMUPS):
        bf16_mha_fwd(Q, K, V, L, O)
        torch.cuda.synchronize()

    for i in range(NUM_ITERS):
        start_events[i].record()
        bf16_mha_fwd(Q, K, V, L, O)
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_time = np.mean(times) * 1e-3
    std_time = np.std(times) * 1e-3
    flops = 4 * B * H * N * N * D
    tflops = flops * 1e-12

    print(f'Time taken: {avg_time * 1e6:.2f} ± {std_time * 1e6:.2f} us')
    print(f'TFLOPS: {tflops / avg_time:.2f} TFLOP/s')

    print("\nBenchmarking backward prep...")

    for i in range(NUM_WARMUPS):
        bf16_mha_bwd_prep(O_grad, O, D_vec)
        torch.cuda.synchronize()

    for i in range(NUM_ITERS):
        start_events[i].record()
        bf16_mha_bwd_prep(O_grad, O, D_vec)
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_time = np.mean(times) * 1e-3
    std_time = np.std(times) * 1e-3
    gb = B * H * N * (D * 2 + D * 2 + 4) * 1e-9

    print(f'Time taken: {avg_time * 1e6:.2f} ± {std_time * 1e6:.2f} us')
    print(f'Throughput: {gb / avg_time:.2f} GB/s')

    print("\nBenchmarking backward pass...")

    for i in range(NUM_WARMUPS):
        bf16_mha_bwd_prep(O_grad, O, D_vec)
        bf16_mha_bwd(Q, K, V, O_grad, Q_grad, K_grad, V_grad, L, D_vec)
        torch.cuda.synchronize()

    for i in range(NUM_ITERS):
        start_events[i].record()
        bf16_mha_bwd_prep(O_grad, O, D_vec)
        bf16_mha_bwd(Q, K, V, O_grad, Q_grad, K_grad, V_grad, L, D_vec)
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_time = np.mean(times) * 1e-3
    std_time = np.std(times) * 1e-3
    flops = 2.5 * 4 * B * H * N * N * D
    tflops = flops * 1e-12

    print(f'Time taken: {avg_time * 1e6:.2f} ± {std_time * 1e6:.2f} us')
    print(f'TFLOPS: {tflops / avg_time:.2f} TFLOP/s')

if BENCHMARK_FA_SM100:
    def flash_attn_fwd(Q, K, V):
        from flash_attn.cute.interface import _flash_attn_fwd
        _flash_attn_fwd(Q, K, V, causal=False)

    Q_ = Q.transpose(1, 2).contiguous()
    K_ = K.transpose(1, 2).contiguous()
    V_ = V.transpose(1, 2).contiguous()

    NUM_WARMUPS = 5
    NUM_ITERS = 10

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_ITERS)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_ITERS)]

    print("\nBenchmarking FA SM100 forward pass...")

    for i in range(NUM_WARMUPS):
        flash_attn_fwd(Q_, K_, V_)
        torch.cuda.synchronize()

    for i in range(NUM_ITERS):
        start_events[i].record()
        flash_attn_fwd(Q_, K_, V_)
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_time = np.mean(times) * 1e-3
    std_time = np.std(times) * 1e-3
    flops = 4 * B * H * N * N * D
    tflops = flops * 1e-12

    print(f'Time taken: {avg_time * 1e6:.2f} ± {std_time * 1e6:.2f} us')
    print(f'TFLOPS: {tflops / avg_time:.2f} TFLOP/s')
