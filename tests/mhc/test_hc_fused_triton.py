"""Tests for the fused HC Triton kernel (pre_split_mixes + sinkhorn).

Validates that the fused kernel matches the composition of the two
reference implementations: mhc_pre_split_mixes_ref followed by
sinkhorn_normalize_ref.
"""

import pytest
import torch
from tile_kernels.mhc.hc_fused_triton import hc_fused_triton
from tile_kernels.torch.mhc import mhc_pre_split_mixes_ref, sinkhorn_normalize_ref


def _reference(
    input_mixes: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_mult: int,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
    sinkhorn_repeat: int,
    sinkhorn_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose the two reference functions to get the expected result."""
    pre, post, comb = mhc_pre_split_mixes_ref(
        input_mixes, mhc_scale, mhc_base,
        mhc_mult, mhc_post_mult_value, mhc_pre_eps,
    )
    comb = sinkhorn_normalize_ref(comb, sinkhorn_repeat, sinkhorn_eps)
    return pre, post, comb


def generate_test_data(
    n0: int, n1: int, mhc_mult: int, device: str = 'cuda',
    mhc_post_mult_value: float = 2.0,
    mhc_pre_eps: float = 0.01,
    sinkhorn_repeat: int = 10,
    sinkhorn_eps: float = 1e-6,
) -> dict:
    mhc_mult3 = mhc_mult * 2 + mhc_mult * mhc_mult
    input_mixes = torch.randn((n0, n1, mhc_mult3), dtype=torch.float, device=device)
    mhc_scale = torch.randn((3,), dtype=torch.float, device=device)
    mhc_base = torch.randn((mhc_mult3,), dtype=torch.float, device=device)

    pre_grad = torch.randn((n0, n1, mhc_mult, 1), dtype=torch.float, device=device)
    post_grad = torch.randn((n0, n1, mhc_mult, 1), dtype=torch.float, device=device)
    comb_grad = torch.randn((n0, n1, mhc_mult, mhc_mult), dtype=torch.float, device=device)

    return {
        'input_mixes': input_mixes,
        'mhc_scale': mhc_scale,
        'mhc_base': mhc_base,
        'pre_grad': pre_grad,
        'post_grad': post_grad,
        'comb_grad': comb_grad,
        'mhc_mult': mhc_mult,
        'mhc_post_mult_value': mhc_post_mult_value,
        'mhc_pre_eps': mhc_pre_eps,
        'sinkhorn_repeat': sinkhorn_repeat,
        'sinkhorn_eps': sinkhorn_eps,
    }


def _run_impl(impl_fn, test_data, with_grad=True):
    """Run an implementation and optionally collect grads."""
    input_mixes_ = test_data['input_mixes'].clone().requires_grad_(with_grad)
    mhc_scale_ = test_data['mhc_scale'].clone().requires_grad_(with_grad)
    mhc_base_ = test_data['mhc_base'].clone().requires_grad_(with_grad)

    pre, post, comb = impl_fn(
        input_mixes_, mhc_scale_, mhc_base_,
        mhc_mult=test_data['mhc_mult'],
        mhc_post_mult_value=test_data['mhc_post_mult_value'],
        mhc_pre_eps=test_data['mhc_pre_eps'],
        sinkhorn_repeat=test_data['sinkhorn_repeat'],
        sinkhorn_eps=test_data['sinkhorn_eps'],
    )

    if with_grad:
        torch.autograd.backward(
            [pre, post, comb],
            [test_data['pre_grad'], test_data['post_grad'], test_data['comb_grad']],
        )

    result = {
        'pre': pre, 'post': post, 'comb': comb,
    }
    if with_grad:
        result['grad_input_mixes'] = input_mixes_.grad
        result['grad_mhc_scale'] = mhc_scale_.grad
        result['grad_mhc_base'] = mhc_base_.grad
    return result


def _reference_wrapper(input_mixes, mhc_scale, mhc_base, *,
                       mhc_mult, mhc_post_mult_value, mhc_pre_eps,
                       sinkhorn_repeat, sinkhorn_eps):
    """Wrap the two-step reference to match the fused API signature."""
    return _reference(
        input_mixes, mhc_scale, mhc_base,
        mhc_mult, mhc_post_mult_value, mhc_pre_eps,
        sinkhorn_repeat, sinkhorn_eps,
    )


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------


@pytest.mark.parametrize('n0', [1, 2])
@pytest.mark.parametrize('n1', [1, 128, 1024])
@pytest.mark.parametrize('mhc_mult', [4])
def test_hc_fused_comprehensive(n0: int, n1: int, mhc_mult: int) -> None:
    """Test fused kernel matches reference for various batch sizes."""
    test_data = generate_test_data(n0=n0, n1=n1, mhc_mult=mhc_mult)

    fused = _run_impl(hc_fused_triton, test_data)
    ref = _run_impl(_reference_wrapper, test_data)

    # Forward checks
    torch.testing.assert_close(fused['pre'], ref['pre'], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(fused['post'], ref['post'], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(fused['comb'], ref['comb'], atol=1e-5, rtol=1e-5)

    # Backward checks
    torch.testing.assert_close(
        fused['grad_input_mixes'], ref['grad_input_mixes'], atol=2e-4, rtol=1e-4)
    torch.testing.assert_close(
        fused['grad_mhc_scale'], ref['grad_mhc_scale'], atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(
        fused['grad_mhc_base'], ref['grad_mhc_base'], atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize('sinkhorn_repeat', [1, 2, 5, 10, 20])
def test_hc_fused_repeat_values(sinkhorn_repeat: int) -> None:
    """Test different sinkhorn repeat counts."""
    test_data = generate_test_data(
        n0=1, n1=256, mhc_mult=4, sinkhorn_repeat=sinkhorn_repeat)

    fused = _run_impl(hc_fused_triton, test_data)
    ref = _run_impl(_reference_wrapper, test_data)

    torch.testing.assert_close(fused['pre'], ref['pre'], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(fused['post'], ref['post'], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(fused['comb'], ref['comb'], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        fused['grad_input_mixes'], ref['grad_input_mixes'], atol=2e-4, rtol=1e-4)


@pytest.mark.parametrize('mhc_post_mult_value', [0.5, 1.0, 2.0, 4.0])
def test_hc_fused_post_mult_values(mhc_post_mult_value: float) -> None:
    """Test different post multiplier values."""
    test_data = generate_test_data(
        n0=1, n1=256, mhc_mult=4, mhc_post_mult_value=mhc_post_mult_value)

    fused = _run_impl(hc_fused_triton, test_data)
    ref = _run_impl(_reference_wrapper, test_data)

    torch.testing.assert_close(fused['pre'], ref['pre'], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(fused['post'], ref['post'], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(fused['comb'], ref['comb'], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        fused['grad_input_mixes'], ref['grad_input_mixes'], atol=2e-4, rtol=1e-4)


def test_hc_fused_fwd_only() -> None:
    """Test forward pass only (no backward)."""
    test_data = generate_test_data(n0=2, n1=512, mhc_mult=4)

    fused = _run_impl(hc_fused_triton, test_data, with_grad=False)
    ref = _run_impl(_reference_wrapper, test_data, with_grad=False)

    torch.testing.assert_close(fused['pre'], ref['pre'], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(fused['post'], ref['post'], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(fused['comb'], ref['comb'], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('mhc_mult', [1, 2, 4, 8])
def test_hc_fused_k_values(mhc_mult: int) -> None:
    """Test different K values."""
    test_data = generate_test_data(n0=1, n1=128, mhc_mult=mhc_mult)

    fused = _run_impl(hc_fused_triton, test_data)
    ref = _run_impl(_reference_wrapper, test_data)

    torch.testing.assert_close(fused['pre'], ref['pre'], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(fused['post'], ref['post'], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(fused['comb'], ref['comb'], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        fused['grad_input_mixes'], ref['grad_input_mixes'], atol=2e-4, rtol=1e-4)


def test_hc_fused_large_batch() -> None:
    """Test with a large batch that exercises multiple program tiles."""
    test_data = generate_test_data(n0=2, n1=4096, mhc_mult=4)

    fused = _run_impl(hc_fused_triton, test_data)
    ref = _run_impl(_reference_wrapper, test_data)

    torch.testing.assert_close(fused['pre'], ref['pre'], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(fused['post'], ref['post'], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(fused['comb'], ref['comb'], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        fused['grad_input_mixes'], ref['grad_input_mixes'], atol=2e-4, rtol=1e-4)
    torch.testing.assert_close(
        fused['grad_mhc_scale'], ref['grad_mhc_scale'], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        fused['grad_mhc_base'], ref['grad_mhc_base'], atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize('mhc_pre_eps', [0.0, 0.001, 0.01, 0.1])
def test_hc_fused_pre_eps_values(mhc_pre_eps: float) -> None:
    """Test different pre epsilon values."""
    test_data = generate_test_data(n0=1, n1=256, mhc_mult=4, mhc_pre_eps=mhc_pre_eps)

    fused = _run_impl(hc_fused_triton, test_data, with_grad=False)
    ref = _run_impl(_reference_wrapper, test_data, with_grad=False)

    torch.testing.assert_close(fused['pre'], ref['pre'], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(fused['post'], ref['post'], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(fused['comb'], ref['comb'], atol=1e-5, rtol=1e-5)
