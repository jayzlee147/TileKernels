"""Tests for the Triton Sinkhorn implementation against sinkhorn_normalize_ref."""

from typing import Callable

import pytest
import torch
from tile_kernels.mhc.sinkhorn_triton import sinkhorn_normalize_triton
from tile_kernels.torch.mhc import sinkhorn_normalize_ref


def generate_sinkhorn_test_data(
    n0: int, n1: int, mhc: int, device: str = 'cuda'
) -> dict[str, torch.Tensor]:
    comb_res_mix = torch.randn((n0, n1, mhc, mhc), dtype=torch.float32, device=device)
    out_grad = torch.randn((n0, n1, mhc, mhc), dtype=torch.float32, device=device)
    return {
        'comb_res_mix': comb_res_mix,
        'out_grad': out_grad,
        'repeat': 10,
        'eps': 1e-6,
    }


def _tester(
    impl: Callable[[torch.Tensor, int, float], torch.Tensor],
    test_data: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    comb_res_mix_ = test_data['comb_res_mix'].clone().requires_grad_()
    out_ = impl(comb_res_mix_, test_data['repeat'], test_data['eps'])
    torch.autograd.backward([out_], [test_data['out_grad']])
    return out_, comb_res_mix_.grad


@pytest.mark.parametrize('n0', [1, 2])
@pytest.mark.parametrize('n1', [1, 1024, 4096])
@pytest.mark.parametrize('mhc', [4])
def test_sinkhorn_triton_comprehensive(n0: int, n1: int, mhc: int) -> None:
    """Test Triton implementation matches reference for various shapes."""
    test_data = generate_sinkhorn_test_data(n0=n0, n1=n1, mhc=mhc)

    out_triton, grad_triton = _tester(sinkhorn_normalize_triton, test_data)
    out_ref, grad_ref = _tester(sinkhorn_normalize_ref, test_data)

    torch.testing.assert_close(out_triton, out_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(grad_triton, grad_ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize('mhc', [1, 2, 4, 8])
def test_sinkhorn_triton_k_values(mhc: int) -> None:
    """Test that different K values are supported."""
    test_data = generate_sinkhorn_test_data(n0=1, n1=128, mhc=mhc)

    out_triton, grad_triton = _tester(sinkhorn_normalize_triton, test_data)
    out_ref, grad_ref = _tester(sinkhorn_normalize_ref, test_data)

    torch.testing.assert_close(out_triton, out_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(grad_triton, grad_ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize('repeat', [1, 2, 5, 10, 20])
def test_sinkhorn_triton_repeat_values(repeat: int) -> None:
    """Test different repeat (iteration) counts."""
    n0, n1, mhc = 1, 256, 4
    comb_res_mix = torch.randn((n0, n1, mhc, mhc), dtype=torch.float32, device='cuda')
    out_grad = torch.randn_like(comb_res_mix)
    test_data = {
        'comb_res_mix': comb_res_mix,
        'out_grad': out_grad,
        'repeat': repeat,
        'eps': 1e-6,
    }

    out_triton, grad_triton = _tester(sinkhorn_normalize_triton, test_data)
    out_ref, grad_ref = _tester(sinkhorn_normalize_ref, test_data)

    torch.testing.assert_close(out_triton, out_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(grad_triton, grad_ref, atol=1e-4, rtol=1e-4)


def test_sinkhorn_triton_fwd_only() -> None:
    """Test forward pass only (no backward)."""
    x = torch.randn(2, 512, 4, 4, dtype=torch.float32, device='cuda')

    out_triton = sinkhorn_normalize_triton(x, 10, 1e-6)
    out_ref = sinkhorn_normalize_ref(x, 10, 1e-6)

    torch.testing.assert_close(out_triton, out_ref, atol=1e-5, rtol=1e-5)


def test_sinkhorn_triton_3d_input() -> None:
    """Test with 3D input (N, K, K) -- no batch dim beyond leading."""
    x = torch.randn(256, 4, 4, dtype=torch.float32, device='cuda')

    out_triton = sinkhorn_normalize_triton(x, 10, 1e-6)
    out_ref = sinkhorn_normalize_ref(x, 10, 1e-6)

    torch.testing.assert_close(out_triton, out_ref, atol=1e-5, rtol=1e-5)
