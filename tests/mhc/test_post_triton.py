"""Tests for the Triton mhc_post implementation against mhc_post_ref."""

from typing import Callable

import pytest
import torch
from tile_kernels.mhc.post_triton import mhc_post_triton
from tile_kernels.torch.mhc import mhc_post_ref


def generate_mhc_post_test_data(
    n0: int,
    n1: int,
    h: int,
    mhc_mult: int,
    device: str = 'cuda',
) -> dict[str, torch.Tensor]:
    x = torch.randn((n0, n1, h), dtype=torch.bfloat16, device=device)
    residual = torch.randn((n0, n1, mhc_mult, h), dtype=torch.bfloat16, device=device)
    post_layer_mix = torch.randn((n0, n1, mhc_mult, 1), dtype=torch.float32, device=device)
    comb_res_mix = torch.randn((n0, n1, mhc_mult, mhc_mult), dtype=torch.float32, device=device)

    o_grad = torch.randn((n0, n1, mhc_mult, h), dtype=torch.bfloat16, device=device)

    return {
        'x': x,
        'residual': residual,
        'post_layer_mix': post_layer_mix,
        'comb_res_mix': comb_res_mix,
        'o_grad': o_grad,
    }


def _tester(
    impl: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    test_data: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    x_ = test_data['x'].clone().requires_grad_()
    residual_ = test_data['residual'].clone().requires_grad_()
    post_layer_mix_ = test_data['post_layer_mix'].clone().requires_grad_()
    comb_res_mix_ = test_data['comb_res_mix'].clone().requires_grad_()
    out_ = impl(x_, residual_, post_layer_mix_, comb_res_mix_)
    torch.autograd.backward([out_], [test_data['o_grad']])
    return out_, x_.grad, residual_.grad, post_layer_mix_.grad, comb_res_mix_.grad


@pytest.mark.parametrize('n0', [1, 2])
@pytest.mark.parametrize('n1', [1, 128, 4096])
@pytest.mark.parametrize('h', [1280, 2560, 4096])
def test_mhc_post_triton_comprehensive(n0: int, n1: int, h: int) -> None:
    """Test Triton implementation matches reference for various shapes."""
    test_data = generate_mhc_post_test_data(n0=n0, n1=n1, h=h, mhc_mult=4)

    out_triton, grad_x_triton, grad_res_triton, grad_plm_triton, grad_crm_triton = _tester(
        mhc_post_triton, test_data
    )
    out_ref, grad_x_ref, grad_res_ref, grad_plm_ref, grad_crm_ref = _tester(
        mhc_post_ref, test_data
    )

    # Forward
    torch.testing.assert_close(out_triton, out_ref)

    # Backward
    torch.testing.assert_close(grad_x_triton, grad_x_ref)
    torch.testing.assert_close(grad_res_triton, grad_res_ref)
    torch.testing.assert_close(grad_plm_triton, grad_plm_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(grad_crm_triton, grad_crm_ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize('h', [512, 1024, 7168])
def test_mhc_post_triton_fwd_only(h: int) -> None:
    """Test forward pass only (no backward)."""
    n0, n1, mhc_mult = 2, 512, 4
    test_data = generate_mhc_post_test_data(n0=n0, n1=n1, h=h, mhc_mult=mhc_mult)

    x = test_data['x']
    residual = test_data['residual']
    post_layer_mix = test_data['post_layer_mix']
    comb_res_mix = test_data['comb_res_mix']

    out_triton = mhc_post_triton(x, residual, post_layer_mix, comb_res_mix)
    out_ref = mhc_post_ref(x, residual, post_layer_mix, comb_res_mix)

    torch.testing.assert_close(out_triton, out_ref)


def test_mhc_post_triton_small() -> None:
    """Minimal smoke test with small dimensions."""
    test_data = generate_mhc_post_test_data(n0=1, n1=1, h=512, mhc_mult=4)

    out_triton, grad_x_triton, grad_res_triton, grad_plm_triton, grad_crm_triton = _tester(
        mhc_post_triton, test_data
    )
    out_ref, grad_x_ref, grad_res_ref, grad_plm_ref, grad_crm_ref = _tester(
        mhc_post_ref, test_data
    )

    torch.testing.assert_close(out_triton, out_ref)
    torch.testing.assert_close(grad_x_triton, grad_x_ref)
    torch.testing.assert_close(grad_res_triton, grad_res_ref)
    torch.testing.assert_close(grad_plm_triton, grad_plm_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(grad_crm_triton, grad_crm_ref, atol=1e-4, rtol=1e-4)
