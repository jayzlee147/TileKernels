"""Tests for the Triton mhc_pre_norm_fn implementation against mhc_pre_norm_fn_ref."""

import pytest
import torch
from tile_kernels.mhc.norm_fn_triton import (
    mhc_pre_norm_fn_triton,
    norm_fn_bwd_triton,
    norm_fn_fwd_triton,
)
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


# ---------------------------------------------------------------------------
# BWD tests
# ---------------------------------------------------------------------------


def _ref_bwd_manual(
    d_out: torch.Tensor,    # [N, K]
    x: torch.Tensor,        # [N, H]  fp32
    fn: torch.Tensor,       # [K, H]  fp32
    n_rms_group: int,
    rms_group_size: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for the BWD pass (manual math)."""
    N, H = x.shape
    K = fn.shape[0]
    x_grouped = x.view(N, n_rms_group, rms_group_size)
    sqrsum = (x_grouped ** 2).sum(-1)  # [N, G]
    inv_rms = torch.rsqrt(sqrsum / rms_group_size + eps)  # [N, G]

    d_x = torch.zeros_like(x)  # [N, H]

    for g in range(n_rms_group):
        s = g * rms_group_size
        e = s + rms_group_size
        x_g = x[:, s:e]                # [N, S]
        fn_g = fn[:, s:e]              # [K, S]
        irms = inv_rms[:, g]           # [N]

        # dot_g[n, k] = sum_h(x_g[n, h] * fn_g[k, h])
        dot_g = torch.einsum('nh,kh->nk', x_g, fn_g)  # [N, K]

        # c_g[n] = sum_k(d_out[n,k] * dot_g[n,k])
        c_g = (d_out * dot_g).sum(-1)  # [N]

        # dout_fn[n, h] = sum_k(d_out[n,k] * fn_g[k, h])
        dout_fn = torch.einsum('nk,kh->nh', d_out, fn_g)  # [N, S]

        # d_x_g[n, h] = irms * dout_fn - x_g * irms^3 * c_g / S
        rms_coeff = -irms ** 3 * c_g / rms_group_size  # [N]
        d_x[:, s:e] = irms.unsqueeze(-1) * dout_fn + rms_coeff.unsqueeze(-1) * x_g

    # d_fn[k, h] = sum_n(d_out[n,k] * x_normed[n,h])
    x_normed = (x_grouped * inv_rms.unsqueeze(-1)).view(N, H)
    d_fn = d_out.t() @ x_normed  # [K, H]

    return d_x, d_fn


def test_norm_fn_bwd_low_level() -> None:
    """Test the low-level norm_fn_bwd_triton against a manual reference."""
    device = 'cuda'
    mhc_mult3 = 24
    total_hidden = 4096
    num_tokens = 32
    n_rms_group = 1
    rms_group_size = total_hidden
    eps = 1e-6

    x = torch.randn(num_tokens, total_hidden, dtype=torch.bfloat16, device=device)
    fn = torch.randn(mhc_mult3, total_hidden, dtype=torch.float32, device=device) * 1e-4
    d_out = torch.randn(num_tokens, mhc_mult3, dtype=torch.float32, device=device)

    d_x_triton, d_fn_triton = norm_fn_bwd_triton(
        d_out, x, fn,
        mhc_mult3=mhc_mult3,
        n_rms_group=n_rms_group,
        rms_group_size=rms_group_size,
        eps=eps,
    )

    d_x_ref, d_fn_ref = _ref_bwd_manual(
        d_out, x.float(), fn,
        n_rms_group=n_rms_group,
        rms_group_size=rms_group_size,
        eps=eps,
    )

    torch.testing.assert_close(d_x_triton, d_x_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(d_fn_triton, d_fn_ref, atol=1e-2, rtol=1e-2)


def _generate_bwd_test_data(
    n1: int,
    mhc_mult: int,
    hidden_size: int,
    generate_normw: bool,
) -> dict:
    """Generate test data with requires_grad for backward testing."""
    n0 = 1
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    mhc_hidden_size = mhc_mult * hidden_size
    device = 'cuda'

    residual = (
        torch.randn((n0, n1, mhc_mult, hidden_size), dtype=torch.float, device=device)
        .mul(1 + torch.arange(mhc_mult, device=device).mul(0.01).view(1, 1, -1, 1))
        .bfloat16()
        .requires_grad_(True)
    )

    fn = (
        torch.randn((mhc_mult3, mhc_mult, hidden_size), dtype=torch.float, device=device)
        * 1e-4
        * (1 + torch.arange(mhc_mult, device=device).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2).requires_grad_(True)

    if generate_normw:
        normw = (
            torch.randn((mhc_hidden_size,), dtype=torch.float, device=device) * 0.1 + 1.0
        ).requires_grad_(True)
    else:
        normw = None

    return {
        'residual': residual,
        'fn': fn,
        'normw': normw,
        'mhc_norm_eps': 1e-6,
    }


@pytest.mark.parametrize('n1', [1, 64])
@pytest.mark.parametrize('hidden_size', [512, 1280])
@pytest.mark.parametrize('generate_normw', [False, True])
def test_norm_fn_triton_bwd_autograd(
    n1: int,
    hidden_size: int,
    generate_normw: bool,
) -> None:
    """Test backward pass via autograd: compare Triton grads vs PyTorch ref grads."""
    mhc_mult = 4
    data = _generate_bwd_test_data(
        n1=n1, mhc_mult=mhc_mult, hidden_size=hidden_size,
        generate_normw=generate_normw,
    )

    residual = data['residual']
    fn = data['fn']
    normw = data['normw']
    eps = data['mhc_norm_eps']

    # --- Triton path ---
    out_triton = mhc_pre_norm_fn_triton(residual, fn, normw, eps)
    # Use a random grad_output
    grad_out = torch.randn_like(out_triton)
    out_triton.backward(grad_out)

    g_res_triton = residual.grad.clone()
    g_fn_triton = fn.grad.clone()
    g_nw_triton = normw.grad.clone() if normw is not None else None

    # --- Reference path (pure PyTorch, should auto-diff correctly) ---
    residual.grad = None
    fn.grad = None
    if normw is not None:
        normw.grad = None

    out_ref = mhc_pre_norm_fn_ref(residual, fn, normw, eps)
    out_ref.backward(grad_out)

    g_res_ref = residual.grad.clone()
    g_fn_ref = fn.grad.clone()
    g_nw_ref = normw.grad.clone() if normw is not None else None

    # Compare
    torch.testing.assert_close(
        g_res_triton.float(), g_res_ref.float(), atol=1e-2, rtol=1e-2,
        msg='d_residual mismatch',
    )
    torch.testing.assert_close(
        g_fn_triton, g_fn_ref, atol=1e-2, rtol=1e-2,
        msg='d_fn mismatch',
    )
    if normw is not None:
        torch.testing.assert_close(
            g_nw_triton, g_nw_ref, atol=1e-2, rtol=1e-2,
            msg='d_norm_weight mismatch',
        )


def test_norm_fn_triton_bwd_smoke() -> None:
    """Minimal backward smoke test."""
    mhc_mult = 4
    data = _generate_bwd_test_data(
        n1=1, mhc_mult=mhc_mult, hidden_size=512, generate_normw=False,
    )
    residual = data['residual']
    fn = data['fn']
    eps = data['mhc_norm_eps']

    out = mhc_pre_norm_fn_triton(residual, fn, None, eps)
    loss = out.sum()
    loss.backward()

    assert residual.grad is not None
    assert fn.grad is not None
    assert residual.grad.shape == residual.shape
    assert fn.grad.shape == fn.shape
