"""Tests for the Triton mhc_pre_norm_fn implementation against mhc_pre_norm_fn_ref."""

import pytest
import torch
from tile_kernels.mhc.norm_fn_triton import mhc_pre_norm_fn_triton, norm_fn_fwd_triton
from tile_kernels.torch.mhc import mhc_pre_norm_fn_ref


def generate_norm_fn_test_data(
    n1: int,
    mhc_mult: int,
    hidden_size: int,
    generate_normw: bool,
) -> dict[str, torch.Tensor]:
    n0 = 1
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    mhc_hidden_size = mhc_mult * hidden_size
    device = 'cuda'

    residual = (
        torch.randn((n0, n1, mhc_mult, hidden_size), dtype=torch.float, device=device)
        .mul(1 + torch.arange(mhc_mult, device=device).mul(0.01).view(1, 1, -1, 1))
        .bfloat16()
    )

    fn = (
        torch.randn((mhc_mult3, mhc_mult, hidden_size), dtype=torch.float, device=device)
        * 1e-4
        * (1 + torch.arange(mhc_mult, device=device).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)

    if generate_normw:
        normw = torch.randn((mhc_hidden_size,), dtype=torch.float, device=device) * 0.1 + 1.0
    else:
        normw = None

    return {
        'residual': residual,
        'fn': fn,
        'normw': normw,
        'mhc_norm_eps': 1e-6,
    }


@pytest.mark.parametrize('n1', [1, 128, 4096])
@pytest.mark.parametrize('hidden_size', [1280, 2560, 4096])
@pytest.mark.parametrize('generate_normw', [False, True])
def test_norm_fn_triton_fwd(
    n1: int,
    hidden_size: int,
    generate_normw: bool,
) -> None:
    """Test Triton FWD matches mhc_pre_norm_fn_ref for various shapes."""
    mhc_mult = 4

    test_data = generate_norm_fn_test_data(
        n1=n1,
        mhc_mult=mhc_mult,
        hidden_size=hidden_size,
        generate_normw=generate_normw,
    )

    residual = test_data['residual']
    fn = test_data['fn']
    normw = test_data['normw']
    eps = test_data['mhc_norm_eps']

    out_triton = mhc_pre_norm_fn_triton(residual, fn, normw, eps)
    out_ref = mhc_pre_norm_fn_ref(residual, fn, normw, eps)

    torch.testing.assert_close(out_triton, out_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize('hidden_size', [512, 1024, 7680])
def test_norm_fn_triton_fwd_only_small(hidden_size: int) -> None:
    """Smoke test with small batch sizes."""
    mhc_mult = 4
    test_data = generate_norm_fn_test_data(
        n1=1,
        mhc_mult=mhc_mult,
        hidden_size=hidden_size,
        generate_normw=False,
    )

    residual = test_data['residual']
    fn = test_data['fn']
    eps = test_data['mhc_norm_eps']

    out_triton = mhc_pre_norm_fn_triton(residual, fn, None, eps)
    out_ref = mhc_pre_norm_fn_ref(residual, fn, None, eps)

    torch.testing.assert_close(out_triton, out_ref, atol=1e-3, rtol=1e-3)


def test_norm_fn_triton_smoke() -> None:
    """Minimal smoke test with the smallest reasonable dimensions."""
    mhc_mult = 4
    test_data = generate_norm_fn_test_data(
        n1=1,
        mhc_mult=mhc_mult,
        hidden_size=512,
        generate_normw=True,
    )

    residual = test_data['residual']
    fn = test_data['fn']
    normw = test_data['normw']
    eps = test_data['mhc_norm_eps']

    out_triton = mhc_pre_norm_fn_triton(residual, fn, normw, eps)
    out_ref = mhc_pre_norm_fn_ref(residual, fn, normw, eps)

    torch.testing.assert_close(out_triton, out_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize('n1', [13, 48, 512])
@pytest.mark.parametrize('hidden_size', [1280, 2560, 4096, 7168])
def test_norm_fn_triton_odd_sizes(n1: int, hidden_size: int) -> None:
    """Test with non-power-of-2 token counts."""
    mhc_mult = 4

    test_data = generate_norm_fn_test_data(
        n1=n1,
        mhc_mult=mhc_mult,
        hidden_size=hidden_size,
        generate_normw=False,
    )

    residual = test_data['residual']
    fn = test_data['fn']
    eps = test_data['mhc_norm_eps']

    out_triton = mhc_pre_norm_fn_triton(residual, fn, None, eps)
    out_ref = mhc_pre_norm_fn_ref(residual, fn, None, eps)

    torch.testing.assert_close(out_triton, out_ref, atol=1e-3, rtol=1e-3)


def test_norm_fn_triton_low_level() -> None:
    """Test the low-level norm_fn_fwd_triton function directly."""
    device = 'cuda'
    mhc_mult3 = 24
    total_hidden = 4096
    num_tokens = 64

    x = torch.randn(num_tokens, total_hidden, dtype=torch.bfloat16, device=device)
    fn = torch.randn(mhc_mult3, total_hidden, dtype=torch.float32, device=device) * 1e-4

    # rms_group_size = total_hidden (1 group, same as ref)
    rms_group_size = total_hidden
    n_rms_group = 1
    eps = 1e-6

    out_triton = norm_fn_fwd_triton(
        x, fn,
        mhc_mult3=mhc_mult3,
        n_rms_group=n_rms_group,
        rms_group_size=rms_group_size,
        eps=eps,
    )

    # Reference computation
    x_f = x.float()
    sqrsum = (x_f ** 2).sum(-1)
    inv_rms = torch.rsqrt(sqrsum / rms_group_size + eps)
    # dot_products: (num_tokens, mhc_mult3)
    dots = torch.einsum('nh,kh->nk', x_f, fn)
    out_ref = dots * inv_rms.unsqueeze(-1)

    torch.testing.assert_close(out_triton, out_ref, atol=1e-3, rtol=1e-3)
