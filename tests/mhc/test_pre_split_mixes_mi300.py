"""MI300 validation for pre_split_mixes_kernel.py after HIP adaptation."""
import importlib.util
import os

# Direct import of kernel module to bypass package import issues
_kernel_path = os.path.join(
    os.path.dirname(__file__), '..', '..', 'tile_kernels', 'mhc', 'pre_split_mixes_kernel.py',
)
spec = importlib.util.spec_from_file_location('pre_split_mixes_kernel', _kernel_path)
pre_split_mixes_kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pre_split_mixes_kernel)

# Direct import of reference implementation
_ref_path = os.path.join(
    os.path.dirname(__file__), '..', '..', 'tile_kernels', 'torch', 'mhc.py',
)
spec_ref = importlib.util.spec_from_file_location('mhc_ref', _ref_path)
mhc_ref = importlib.util.module_from_spec(spec_ref)
spec_ref.loader.exec_module(mhc_ref)

import torch

_IS_HIP = pre_split_mixes_kernel._IS_HIP
_WARP_SIZE = pre_split_mixes_kernel._WARP_SIZE
_DEFAULT_NUM_SMS = pre_split_mixes_kernel._DEFAULT_NUM_SMS
_mhc_pre_split_mixes_fwd = pre_split_mixes_kernel._mhc_pre_split_mixes_fwd
_mhc_pre_split_mixes_bwd = pre_split_mixes_kernel._mhc_pre_split_mixes_bwd
mhc_pre_split_mixes_ref = mhc_ref.mhc_pre_split_mixes_ref


def test_constants():
    """Verify platform-aware constants are set correctly."""
    if _IS_HIP:
        assert _WARP_SIZE == 64, f'Expected WARP_SIZE=64 on HIP, got {_WARP_SIZE}'
        assert _DEFAULT_NUM_SMS == 304, f'Expected DEFAULT_NUM_SMS=304 on HIP, got {_DEFAULT_NUM_SMS}'
    else:
        assert _WARP_SIZE == 32, f'Expected WARP_SIZE=32 on CUDA, got {_WARP_SIZE}'
        assert _DEFAULT_NUM_SMS == 148, f'Expected DEFAULT_NUM_SMS=148 on CUDA, got {_DEFAULT_NUM_SMS}'
    return True


def test_fwd(num_tokens, mhc_mult, mhc_post_mult_value=2.0, mhc_pre_eps=1e-2, token_block_size=32):
    """Test FWD kernel against reference implementation."""
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2

    input_mixes = torch.randn(1, num_tokens, mhc_mult3, dtype=torch.float32, device='cuda')
    mhc_scale = torch.randn(3, dtype=torch.float32, device='cuda')
    mhc_base = torch.randn(mhc_mult3, dtype=torch.float32, device='cuda')

    # Reference
    pre_ref, post_ref, comb_ref = mhc_pre_split_mixes_ref(
        input_mixes, mhc_scale, mhc_base, mhc_mult, mhc_post_mult_value, mhc_pre_eps,
    )

    # Kernel
    input_mixes_flat = input_mixes.view(-1, mhc_mult3)
    n = input_mixes_flat.shape[0]
    pre_layer_mix = torch.empty(n, mhc_mult, dtype=torch.float32, device='cuda')
    post_layer_mix = torch.empty(n, mhc_mult, dtype=torch.float32, device='cuda')
    comb_res_mix = torch.empty(n, mhc_mult2, dtype=torch.float32, device='cuda')

    kernel = _mhc_pre_split_mixes_fwd(mhc_mult, mhc_post_mult_value, mhc_pre_eps, token_block_size)
    kernel(input_mixes_flat, mhc_scale, mhc_base, pre_layer_mix, post_layer_mix, comb_res_mix)
    torch.cuda.synchronize()

    # Reshape kernel outputs to match reference
    pre_tl = pre_layer_mix.view(1, num_tokens, mhc_mult, 1)
    post_tl = post_layer_mix.view(1, num_tokens, mhc_mult, 1)
    comb_tl = comb_res_mix.view(1, num_tokens, mhc_mult, mhc_mult)

    diffs = {
        'pre_layer_mix': (pre_tl - pre_ref).abs().max().item(),
        'post_layer_mix': (post_tl - post_ref).abs().max().item(),
        'comb_res_mix': (comb_tl - comb_ref).abs().max().item(),
    }
    return diffs


def test_bwd(num_tokens, mhc_mult, mhc_post_mult_value=2.0, mhc_pre_eps=1e-2, token_block_size=32, num_sms=None):
    """Test BWD kernel against PyTorch autograd reference."""
    if num_sms is None:
        num_sms = _DEFAULT_NUM_SMS

    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2

    # Generate inputs
    input_mixes_3d = torch.randn(1, num_tokens, mhc_mult3, dtype=torch.float32, device='cuda')
    mhc_scale = torch.randn(3, dtype=torch.float32, device='cuda')
    mhc_base = torch.randn(mhc_mult3, dtype=torch.float32, device='cuda')

    # Compute forward reference with autograd
    input_mixes_ref = input_mixes_3d.clone().requires_grad_()
    mhc_scale_ref = mhc_scale.clone().requires_grad_()
    mhc_base_ref = mhc_base.clone().requires_grad_()

    pre_ref, post_ref, comb_ref = mhc_pre_split_mixes_ref(
        input_mixes_ref, mhc_scale_ref, mhc_base_ref, mhc_mult, mhc_post_mult_value, mhc_pre_eps,
    )

    # Random grad outputs
    pre_grad = torch.randn_like(pre_ref)
    post_grad = torch.randn_like(post_ref)
    comb_grad = torch.randn_like(comb_ref)

    torch.autograd.backward([pre_ref, post_ref, comb_ref], [pre_grad, post_grad, comb_grad])

    # Now run kernel FWD to get cached activations
    input_mixes_flat = input_mixes_3d.view(-1, mhc_mult3)
    n = input_mixes_flat.shape[0]
    pre_layer_mix_k = torch.empty(n, mhc_mult, dtype=torch.float32, device='cuda')
    post_layer_mix_k = torch.empty(n, mhc_mult, dtype=torch.float32, device='cuda')
    comb_res_mix_k = torch.empty(n, mhc_mult2, dtype=torch.float32, device='cuda')

    fwd_kernel = _mhc_pre_split_mixes_fwd(mhc_mult, mhc_post_mult_value, mhc_pre_eps, token_block_size)
    fwd_kernel(input_mixes_flat, mhc_scale, mhc_base, pre_layer_mix_k, post_layer_mix_k, comb_res_mix_k)
    torch.cuda.synchronize()

    # Run BWD kernel
    input_mixes_grad = torch.empty_like(input_mixes_flat)
    mhc_scale_grad_partial = torch.empty(num_sms, 3, dtype=torch.float32, device='cuda')
    mhc_base_grad_partial = torch.empty(num_sms, mhc_mult3, dtype=torch.float32, device='cuda')

    bwd_kernel = _mhc_pre_split_mixes_bwd(mhc_mult, mhc_post_mult_value, token_block_size, num_sms=num_sms)
    bwd_kernel(
        pre_grad.view(n, mhc_mult),
        post_grad.view(n, mhc_mult),
        comb_grad.view(n, mhc_mult2),
        input_mixes_flat,
        post_layer_mix_k,
        mhc_scale,
        mhc_base,
        input_mixes_grad,
        mhc_scale_grad_partial,
        mhc_base_grad_partial,
    )
    torch.cuda.synchronize()

    mhc_scale_grad_tl = mhc_scale_grad_partial.sum(0)
    mhc_base_grad_tl = mhc_base_grad_partial.sum(0)
    input_mixes_grad_tl = input_mixes_grad.view_as(input_mixes_3d)

    diffs = {
        'input_mixes_grad': (input_mixes_grad_tl - input_mixes_ref.grad).abs().max().item(),
        'mhc_scale_grad': (mhc_scale_grad_tl - mhc_scale_ref.grad).abs().max().item(),
        'mhc_base_grad': (mhc_base_grad_tl - mhc_base_ref.grad).abs().max().item(),
    }
    return diffs


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
    print(f'_WARP_SIZE: {_WARP_SIZE}')
    print(f'_DEFAULT_NUM_SMS: {_DEFAULT_NUM_SMS}')
    print()

    mhc_mult = 4
    all_pass = True

    # --- Constants check ---
    print('=== Constants ===')
    ok = test_constants()
    print(f'  Platform constants: {"PASS" if ok else "FAIL"}')
    all_pass &= ok
    print()

    # --- FWD correctness ---
    print('=== FWD Correctness ===')
    fwd_tol = 1e-4
    for N in [32, 256, 1024, 4096]:
        diffs = test_fwd(N, mhc_mult)
        max_diff = max(diffs.values())
        ok = max_diff < fwd_tol
        all_pass &= ok
        print(f'  N={N:5d}: max_diff={max_diff:.2e} {"PASS" if ok else "FAIL"}  {diffs}')
    print()

    # --- BWD correctness ---
    print('=== BWD Correctness ===')
    # Use a small num_sms for testing to keep tensor sizes manageable
    test_num_sms = min(_DEFAULT_NUM_SMS, 32)
    bwd_tol = 5e-3
    for N in [32, 256, 1024, 4096]:
        diffs = test_bwd(N, mhc_mult, num_sms=test_num_sms)
        max_diff = max(diffs.values())
        ok = max_diff < bwd_tol
        all_pass &= ok
        print(f'  N={N:5d}: max_diff={max_diff:.2e} {"PASS" if ok else "FAIL"}  {diffs}')
    print()

    # --- Performance ---
    print('=== Performance (N=4096) ===')
    N = 4096
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2
    token_block_size = 32

    input_mixes = torch.randn(N, mhc_mult3, dtype=torch.float32, device='cuda')
    mhc_scale = torch.randn(3, dtype=torch.float32, device='cuda')
    mhc_base = torch.randn(mhc_mult3, dtype=torch.float32, device='cuda')
    pre_out = torch.empty(N, mhc_mult, dtype=torch.float32, device='cuda')
    post_out = torch.empty(N, mhc_mult, dtype=torch.float32, device='cuda')
    comb_out = torch.empty(N, mhc_mult2, dtype=torch.float32, device='cuda')

    fwd_k = _mhc_pre_split_mixes_fwd(mhc_mult, 2.0, 1e-2, token_block_size)
    fwd_ms = bench(fwd_k, (input_mixes, mhc_scale, mhc_base, pre_out, post_out, comb_out))
    print(f'  FWD: {fwd_ms:.4f} ms')

    perf_num_sms = test_num_sms
    pre_grad = torch.randn(N, mhc_mult, dtype=torch.float32, device='cuda')
    post_grad = torch.randn(N, mhc_mult, dtype=torch.float32, device='cuda')
    comb_grad = torch.randn(N, mhc_mult2, dtype=torch.float32, device='cuda')
    input_grad = torch.empty(N, mhc_mult3, dtype=torch.float32, device='cuda')
    scale_grad_p = torch.empty(perf_num_sms, 3, dtype=torch.float32, device='cuda')
    base_grad_p = torch.empty(perf_num_sms, mhc_mult3, dtype=torch.float32, device='cuda')

    bwd_k = _mhc_pre_split_mixes_bwd(mhc_mult, 2.0, token_block_size, num_sms=perf_num_sms)
    bwd_ms = bench(
        bwd_k,
        (pre_grad, post_grad, comb_grad, input_mixes, post_out, mhc_scale, mhc_base,
         input_grad, scale_grad_p, base_grad_p),
    )
    print(f'  BWD: {bwd_ms:.4f} ms')

    print(f'\n{"ALL PASSED" if all_pass else "SOME FAILED"}')
