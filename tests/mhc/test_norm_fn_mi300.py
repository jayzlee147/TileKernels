"""MI300 validation for norm_fn_kernel.py after adaptation.

Only tests the 4 kernels that compile on ROCm/HIP:
  - _mhc_fn_normw_merge_fwd
  - _mhc_fn_normw_merge_bwd
  - _mhc_pre_norm_fn_fwd_norm
  - _mhc_pre_norm_fn_bwd_norm

The GEMM-based kernels (_mhc_pre_norm_fn_fwd_mul, _mhc_pre_norm_fn_bwd_mul)
are skipped due to a tilelang ROCm GEMM lowering bug.
"""
import importlib.util
import os

import torch

# Direct import to bypass unrelated package import errors
spec = importlib.util.spec_from_file_location(
    'norm_fn_kernel',
    os.path.join(os.path.dirname(__file__), '..', '..', 'tile_kernels', 'mhc', 'norm_fn_kernel.py'),
)
norm_fn_kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(norm_fn_kernel)

_IS_HIP = norm_fn_kernel._IS_HIP
_mhc_fn_normw_merge_fwd = norm_fn_kernel._mhc_fn_normw_merge_fwd
_mhc_fn_normw_merge_bwd = norm_fn_kernel._mhc_fn_normw_merge_bwd
_mhc_pre_norm_fn_fwd_norm = norm_fn_kernel._mhc_pre_norm_fn_fwd_norm
_mhc_pre_norm_fn_bwd_norm = norm_fn_kernel._mhc_pre_norm_fn_bwd_norm


# ---------------------------------------------------------------------------
# Tests for _mhc_fn_normw_merge_fwd / _mhc_fn_normw_merge_bwd
# ---------------------------------------------------------------------------

def test_normw_merge_fwd(m, n):
    """out_fn[i, j] = fn[i, j] * normw[j]"""
    fn = torch.randn(m, n, dtype=torch.float32, device='cuda')
    normw = torch.randn(n, dtype=torch.float32, device='cuda')
    out_fn = torch.empty(m, n, dtype=torch.float32, device='cuda')

    ref = fn * normw.unsqueeze(0)

    kernel = _mhc_fn_normw_merge_fwd(m, n)
    kernel(fn, normw, out_fn)
    torch.cuda.synchronize()

    diff = (out_fn - ref).abs().max().item()
    return diff


def test_normw_merge_bwd(m, n):
    """
    fn_grad[i, j] += out_fn_grad[i, j] * normw[j]
    normw_grad[j] += sum_i(out_fn_grad[i, j] * fn[i, j])
    """
    fn = torch.randn(m, n, dtype=torch.float32, device='cuda')
    normw = torch.randn(n, dtype=torch.float32, device='cuda')
    out_fn_grad = torch.randn(m, n, dtype=torch.float32, device='cuda')
    fn_grad = torch.zeros(m, n, dtype=torch.float32, device='cuda')
    normw_grad = torch.zeros(n, dtype=torch.float32, device='cuda')

    fn_grad_ref = out_fn_grad * normw.unsqueeze(0)
    normw_grad_ref = (out_fn_grad * fn).sum(dim=0)

    kernel = _mhc_fn_normw_merge_bwd(m, n)
    kernel(fn, normw, out_fn_grad, fn_grad, normw_grad)
    torch.cuda.synchronize()

    return max(
        (fn_grad - fn_grad_ref).abs().max().item(),
        (normw_grad - normw_grad_ref).abs().max().item(),
    )


# ---------------------------------------------------------------------------
# Smoke tests for _mhc_pre_norm_fn_fwd_norm / _mhc_pre_norm_fn_bwd_norm
# ---------------------------------------------------------------------------

def test_fwd_norm_smoke(num_tokens, mhc_mult3, n_rms_group, rms_group_size, n_splits):
    """Smoke test: output has correct shape, no NaN/Inf."""
    rms_eps = 1e-6

    out_mul_splitted = torch.randn(n_splits, num_tokens, n_rms_group, mhc_mult3,
                                   dtype=torch.float32, device='cuda')
    sqrsum_splitted = torch.randn(n_splits, num_tokens, n_rms_group,
                                  dtype=torch.float32, device='cuda').abs() + 0.1
    out_mul = torch.empty(num_tokens, n_rms_group, mhc_mult3, dtype=torch.float32, device='cuda')
    sqrsum = torch.empty(num_tokens, n_rms_group, dtype=torch.float32, device='cuda')
    out = torch.empty(num_tokens, mhc_mult3, dtype=torch.float32, device='cuda')

    kernel = _mhc_pre_norm_fn_fwd_norm(mhc_mult3, n_rms_group, rms_group_size, rms_eps, n_splits)
    kernel(out_mul_splitted, sqrsum_splitted, out_mul, sqrsum, out)
    torch.cuda.synchronize()

    ok = (
        out.shape == (num_tokens, mhc_mult3)
        and not torch.isnan(out).any().item()
        and not torch.isinf(out).any().item()
    )
    return ok


def test_bwd_norm_smoke(num_tokens, mhc_mult3, n_rms_group, rms_group_size):
    """Smoke test: output has correct shape, no NaN/Inf."""
    rms_eps = 1e-6

    out_grad = torch.randn(num_tokens, mhc_mult3, dtype=torch.float32, device='cuda')
    out_mul = torch.randn(num_tokens, n_rms_group, mhc_mult3, dtype=torch.float32, device='cuda')
    sqrsum = (torch.randn(num_tokens, n_rms_group, dtype=torch.float32, device='cuda').abs() + 0.1)
    out_mul_grad = torch.empty(num_tokens, n_rms_group, mhc_mult3, dtype=torch.float32, device='cuda')
    sqrsum_grad = torch.empty(num_tokens, n_rms_group, dtype=torch.float32, device='cuda')

    kernel = _mhc_pre_norm_fn_bwd_norm(mhc_mult3, n_rms_group, rms_group_size, rms_eps)
    kernel(out_grad, out_mul, sqrsum, out_mul_grad, sqrsum_grad)
    torch.cuda.synchronize()

    ok = (
        out_mul_grad.shape == (num_tokens, n_rms_group, mhc_mult3)
        and sqrsum_grad.shape == (num_tokens, n_rms_group)
        and not torch.isnan(out_mul_grad).any().item()
        and not torch.isinf(out_mul_grad).any().item()
        and not torch.isnan(sqrsum_grad).any().item()
        and not torch.isinf(sqrsum_grad).any().item()
    )
    return ok


# ---------------------------------------------------------------------------
# Bench helper
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Platform: {"ROCm/HIP" if _IS_HIP else "CUDA"}')
    print(f'Wavefront size: {norm_fn_kernel._WAVEFRONT_SIZE}')
    print()

    all_pass = True

    # --- normw_merge_fwd / bwd correctness ---
    print('=== normw_merge_fwd correctness ===')
    for m, n in [(1, 256), (32, 512), (128, 1024), (256, 5120)]:
        diff = test_normw_merge_fwd(m, n)
        ok = diff < 1e-5
        all_pass &= ok
        print(f'  m={m:5d}, n={n:5d}: diff={diff:.8f} {"PASS" if ok else "FAIL"}')

    print('\n=== normw_merge_bwd correctness ===')
    for m, n in [(1, 256), (32, 512), (128, 1024), (256, 5120)]:
        diff = test_normw_merge_bwd(m, n)
        ok = diff < 1e-2
        all_pass &= ok
        print(f'  m={m:5d}, n={n:5d}: diff={diff:.8f} {"PASS" if ok else "FAIL"}')

    # --- fwd_norm / bwd_norm smoke tests ---
    mhc_mult3 = 24  # mhc_mult=4 -> 4*(2+4)=24
    n_rms_group = 4
    rms_group_size = 1280

    print('\n=== pre_norm_fn_fwd_norm smoke ===')
    for num_tokens in [1, 32, 256]:
        for n_splits in [1, 4]:
            ok = test_fwd_norm_smoke(num_tokens, mhc_mult3, n_rms_group, rms_group_size, n_splits)
            all_pass &= ok
            print(f'  N={num_tokens:5d}, splits={n_splits}: {"PASS" if ok else "FAIL"}')

    print('\n=== pre_norm_fn_bwd_norm smoke ===')
    for num_tokens in [1, 32, 256]:
        ok = test_bwd_norm_smoke(num_tokens, mhc_mult3, n_rms_group, rms_group_size)
        all_pass &= ok
        print(f'  N={num_tokens:5d}: {"PASS" if ok else "FAIL"}')

    # --- SKIPPED kernels ---
    print('\n=== SKIPPED: _mhc_pre_norm_fn_fwd_mul ===')
    print('  SKIP: tilelang ROCm GEMM lowering not supported (T.annotate_layout + make_swizzled_layout)')

    print('\n=== SKIPPED: _mhc_pre_norm_fn_bwd_mul ===')
    print('  SKIP: tilelang ROCm GEMM lowering not supported (T.annotate_layout + make_swizzled_layout)')

    # --- Performance ---
    print('\n=== Performance ===')
    m, n = 256, 5120
    fwd_k = _mhc_fn_normw_merge_fwd(m, n)
    bwd_k = _mhc_fn_normw_merge_bwd(m, n)

    fn_t = torch.randn(m, n, dtype=torch.float32, device='cuda')
    normw_t = torch.randn(n, dtype=torch.float32, device='cuda')
    out_fn_t = torch.empty(m, n, dtype=torch.float32, device='cuda')
    out_fn_grad_t = torch.randn(m, n, dtype=torch.float32, device='cuda')
    fn_grad_t = torch.zeros(m, n, dtype=torch.float32, device='cuda')
    normw_grad_t = torch.zeros(n, dtype=torch.float32, device='cuda')

    fwd_ms = bench(fwd_k, (fn_t, normw_t, out_fn_t))
    bwd_ms = bench(bwd_k, (fn_t, normw_t, out_fn_grad_t, fn_grad_t, normw_grad_t))
    print(f'  normw_merge_fwd (m={m}, n={n}): {fwd_ms:.4f} ms')
    print(f'  normw_merge_bwd (m={m}, n={n}): {bwd_ms:.4f} ms')

    print(f'\n{"ALL PASSED" if all_pass else "SOME FAILED"}')
