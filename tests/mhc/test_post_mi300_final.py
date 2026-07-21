"""Final MI300 validation for post_kernel.py after adaptation."""
import importlib.util
import os
import sys

# Direct import of post_kernel to bypass unrelated package import errors
spec = importlib.util.spec_from_file_location(
    'post_kernel',
    os.path.join(os.path.dirname(__file__), '..', '..', 'tile_kernels', 'mhc', 'post_kernel.py'),
)
post_kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(post_kernel)

import torch

_IS_HIP = post_kernel._IS_HIP
_mhc_post_fwd = post_kernel._mhc_post_fwd
_mhc_post_bwd = post_kernel._mhc_post_bwd


def test_fwd(N, mhc, h):
    a = torch.randn(N, mhc, mhc, dtype=torch.float32, device='cuda')
    b = torch.randn(N, mhc, h, dtype=torch.bfloat16, device='cuda')
    c = torch.randn(N, mhc, dtype=torch.float32, device='cuda')
    d = torch.randn(N, h, dtype=torch.bfloat16, device='cuda')
    out = torch.empty(N, mhc, h, dtype=torch.bfloat16, device='cuda')

    ref = (c.unsqueeze(-1) * d.float().unsqueeze(1) +
           torch.einsum('nkm,nkh->nmh', a, b.float())).bfloat16()

    kernel = _mhc_post_fwd(mhc, h)
    kernel(a, b, c, d, out)
    torch.cuda.synchronize()

    diff = (out.float() - ref.float()).abs().max().item()
    return diff


def test_bwd(N, mhc, h):
    dx = torch.randn(N, mhc, h, dtype=torch.bfloat16, device='cuda')
    a = torch.randn(N, mhc, mhc, dtype=torch.float32, device='cuda')
    b = torch.randn(N, mhc, h, dtype=torch.bfloat16, device='cuda')
    c = torch.randn(N, mhc, dtype=torch.float32, device='cuda')
    d = torch.randn(N, h, dtype=torch.bfloat16, device='cuda')

    kernel = _mhc_post_bwd(mhc, h)
    da, db, dc, dd = kernel(dx, a, b, c, d)
    torch.cuda.synchronize()

    da_ref = torch.einsum('nih,njh->nij', b.float(), dx.float()).float()
    db_ref = torch.einsum('nij,njh->nih', a, dx.float()).bfloat16()
    dc_ref = torch.einsum('nh,nmh->nm', d.float(), dx.float()).float()
    dd_ref = torch.einsum('nm,nmh->nh', c, dx.float()).bfloat16()

    return max(
        (da - da_ref).abs().max().item(),
        (db - db_ref).abs().max().item(),
        (dc - dc_ref).abs().max().item(),
        (dd - dd_ref).abs().max().item(),
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
    print(f'FWD defaults: {post_kernel._FWD_DEFAULTS}')
    print(f'BWD defaults: {post_kernel._BWD_DEFAULTS}')
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
    fwd_k = _mhc_post_fwd(mhc, h)
    bwd_k = _mhc_post_bwd(mhc, h)

    a = torch.randn(N, mhc, mhc, dtype=torch.float32, device='cuda')
    b = torch.randn(N, mhc, h, dtype=torch.bfloat16, device='cuda')
    c = torch.randn(N, mhc, dtype=torch.float32, device='cuda')
    d = torch.randn(N, h, dtype=torch.bfloat16, device='cuda')
    out = torch.empty(N, mhc, h, dtype=torch.bfloat16, device='cuda')
    dx = torch.randn(N, mhc, h, dtype=torch.bfloat16, device='cuda')

    fwd_ms = bench(fwd_k, (a, b, c, d, out))
    bwd_ms = bench(bwd_k, (dx, a, b, c, d))
    print(f'  FWD: {fwd_ms:.4f} ms')
    print(f'  BWD: {bwd_ms:.4f} ms')

    print(f'\n{"ALL PASSED" if all_pass else "SOME FAILED"}')
