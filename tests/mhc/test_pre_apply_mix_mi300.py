"""MI300 validation for pre_apply_mix_kernel.py after adaptation."""
import importlib.util
import os

# Direct import of pre_apply_mix_kernel to bypass unrelated package import errors
spec = importlib.util.spec_from_file_location(
    'pre_apply_mix_kernel',
    os.path.join(os.path.dirname(__file__), '..', '..', 'tile_kernels', 'mhc', 'pre_apply_mix_kernel.py'),
)
pre_apply_mix_kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pre_apply_mix_kernel)

import torch

_IS_HIP = pre_apply_mix_kernel._IS_HIP
_mhc_pre_apply_mix_fwd = pre_apply_mix_kernel._mhc_pre_apply_mix_fwd
_mhc_pre_apply_mix_bwd = pre_apply_mix_kernel._mhc_pre_apply_mix_bwd


def test_fwd(N, mhc, h):
    x = torch.randn(N, mhc, h, dtype=torch.bfloat16, device='cuda')
    mix = torch.randn(N, mhc, dtype=torch.float32, device='cuda')
    out = torch.empty(N, h, dtype=torch.bfloat16, device='cuda')

    # Reference: o[n, h] = sum_m(mix[n, m] * x[n, m, h])
    ref = torch.einsum('nm,nmh->nh', mix, x.float()).bfloat16()

    kernel = _mhc_pre_apply_mix_fwd(mhc, h)
    kernel(x, mix, out)
    torch.cuda.synchronize()

    diff = (out.float() - ref.float()).abs().max().item()
    return diff


def test_bwd(N, mhc, h):
    o_grad = torch.randn(N, h, dtype=torch.bfloat16, device='cuda')
    x = torch.randn(N, mhc, h, dtype=torch.bfloat16, device='cuda')
    mix = torch.randn(N, mhc, dtype=torch.float32, device='cuda')
    x_grad = torch.zeros(N, mhc, h, dtype=torch.bfloat16, device='cuda')

    kernel = _mhc_pre_apply_mix_bwd(mhc, h)
    mix_grad = kernel(o_grad, x, mix, x_grad)
    torch.cuda.synchronize()

    # Reference gradients:
    # mix_grad[n, m] = sum_h(o_grad[n, h] * x[n, m, h])
    # x_grad[n, m, h] += mix[n, m] * o_grad[n, h]
    mix_grad_ref = torch.einsum('nh,nmh->nm', o_grad.float(), x.float()).float()
    x_grad_ref = (mix.unsqueeze(-1) * o_grad.float().unsqueeze(1)).bfloat16()

    return max(
        (mix_grad.float() - mix_grad_ref).abs().max().item(),
        (x_grad.float() - x_grad_ref.float()).abs().max().item(),
    )


def bench(fn, args, warmup=10, iters=100):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn(*args)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


if __name__ == '__main__':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Platform: {"ROCm/HIP" if _IS_HIP else "CUDA"}')
    print(f'FWD defaults: {pre_apply_mix_kernel._FWD_DEFAULTS}')
    print(f'BWD defaults: {pre_apply_mix_kernel._BWD_DEFAULTS}')
    print()

    mhc, h = 4, 4096
    all_pass = True

    # Correctness
    print('=== Correctness ===')
    for N in [1, 32, 256, 4096]:
        fwd_diff = test_fwd(N, mhc, h)
        bwd_diff = test_bwd(N, mhc, h)
        fwd_ok = fwd_diff < 0.1
        bwd_ok = bwd_diff < 1.0
        all_pass &= fwd_ok and bwd_ok
        print(f'  N={N:5d}: FWD diff={fwd_diff:.6f} {"PASS" if fwd_ok else "FAIL"}'
              f'  BWD diff={bwd_diff:.6f} {"PASS" if bwd_ok else "FAIL"}')

    # Performance
    print('\n=== Performance (N=4096) ===')
    N = 4096
    fwd_k = _mhc_pre_apply_mix_fwd(mhc, h)
    bwd_k = _mhc_pre_apply_mix_bwd(mhc, h)

    x = torch.randn(N, mhc, h, dtype=torch.bfloat16, device='cuda')
    mix = torch.randn(N, mhc, dtype=torch.float32, device='cuda')
    out = torch.empty(N, h, dtype=torch.bfloat16, device='cuda')
    o_grad = torch.randn(N, h, dtype=torch.bfloat16, device='cuda')
    x_grad = torch.zeros(N, mhc, h, dtype=torch.bfloat16, device='cuda')

    fwd_ms = bench(fwd_k, (x, mix, out))
    bwd_ms = bench(bwd_k, (o_grad, x, mix, x_grad))
    print(f'  FWD: {fwd_ms:.4f} ms')
    print(f'  BWD: {bwd_ms:.4f} ms')

    print(f'\n{"ALL PASSED" if all_pass else "SOME FAILED"}')
