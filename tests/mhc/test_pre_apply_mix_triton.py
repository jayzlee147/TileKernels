"""Tests for the Triton mhc_pre_apply_mix implementation against mhc_pre_apply_mix_ref."""

from typing import Callable

import pytest
import torch
from tile_kernels.mhc.pre_apply_mix_triton import mhc_pre_apply_mix_triton
from tile_kernels.torch.mhc import mhc_pre_apply_mix_ref


def generate_pre_apply_mix_test_data(
    n0: int, n1: int, mhc: int, h: int, device: str = 'cuda'
) -> dict[str, torch.Tensor]:
    x = torch.randn(n0, n1, mhc, h, dtype=torch.bfloat16, device=device).sigmoid()
    mix = torch.randn(n0, n1, mhc, 1, dtype=torch.float32, device=device).softmax(-2)
    o_grad = torch.randn(n0, n1, h, dtype=torch.bfloat16, device=device)

    return {
        'x': x,
        'mix': mix,
        'o_grad': o_grad,
    }


def _tester(
    impl: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    test_data: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_ = test_data['x'].clone().requires_grad_()
    mix_ = test_data['mix'].clone().requires_grad_()
    o_ = impl(x_, mix_)
    torch.autograd.backward([o_], [test_data['o_grad']])
    return o_, x_.grad, mix_.grad


@pytest.mark.parametrize('n0', [1, 2])
@pytest.mark.parametrize('n1', [1, 128, 4096])
@pytest.mark.parametrize('h', [1280, 2560, 4096])
def test_pre_apply_mix_triton_comprehensive(n0: int, n1: int, h: int) -> None:
    """Test Triton implementation matches reference for various shapes."""
    mhc = 4
    test_data = generate_pre_apply_mix_test_data(n0=n0, n1=n1, mhc=mhc, h=h)

    o_triton, x_grad_triton, mix_grad_triton = _tester(mhc_pre_apply_mix_triton, test_data)
    o_ref, x_grad_ref, mix_grad_ref = _tester(mhc_pre_apply_mix_ref, test_data)

    # Forward
    torch.testing.assert_close(o_triton, o_ref)

    # Backward
    torch.testing.assert_close(x_grad_triton, x_grad_ref)
    torch.testing.assert_close(mix_grad_triton, mix_grad_ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize('h', [512, 1024, 7680])
def test_pre_apply_mix_triton_fwd_only(h: int) -> None:
    """Test forward pass only (no backward)."""
    n0, n1, mhc = 2, 512, 4
    test_data = generate_pre_apply_mix_test_data(n0=n0, n1=n1, mhc=mhc, h=h)

    x = test_data['x']
    mix = test_data['mix']

    out_triton = mhc_pre_apply_mix_triton(x, mix)
    out_ref = mhc_pre_apply_mix_ref(x, mix)

    torch.testing.assert_close(out_triton, out_ref)


def test_pre_apply_mix_triton_small() -> None:
    """Minimal smoke test with small dimensions."""
    mhc = 4
    test_data = generate_pre_apply_mix_test_data(n0=1, n1=1, mhc=mhc, h=512)

    o_triton, x_grad_triton, mix_grad_triton = _tester(mhc_pre_apply_mix_triton, test_data)
    o_ref, x_grad_ref, mix_grad_ref = _tester(mhc_pre_apply_mix_ref, test_data)

    torch.testing.assert_close(o_triton, o_ref)
    torch.testing.assert_close(x_grad_triton, x_grad_ref)
    torch.testing.assert_close(mix_grad_triton, mix_grad_ref, atol=1e-4, rtol=1e-4)
