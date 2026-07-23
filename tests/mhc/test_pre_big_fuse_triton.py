"""Tests for the Triton-fused mhc_pre_big_fuse kernel.

Validates the fused Triton kernel (pre_big_fuse_fwd_triton) against
step-by-step pure-torch reference functions.

The test does NOT require tilelang — it simulates the GEMM output in
pure torch and feeds it directly to the Triton fused kernel.
"""

import importlib.util
import sys

import pytest
import torch


# ---------------------------------------------------------------------------
# Import the Triton module directly (avoids tilelang dependency via __init__)
# ---------------------------------------------------------------------------
def _import_triton_module():
    spec = importlib.util.spec_from_file_location(
        'pre_big_fuse_triton',
        'tile_kernels/mhc/pre_big_fuse_triton.py',
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_triton_mod = _import_triton_module()
pre_big_fuse_fwd_triton = _triton_mod.pre_big_fuse_fwd_triton


# ---------------------------------------------------------------------------
# Pure-torch reference implementations
# ---------------------------------------------------------------------------


def _norm_fn_fwd_norm_ref(
    gemm_out_mul: torch.Tensor,     # [n_splits, N, mhc_mult3]
    gemm_out_sqrsum: torch.Tensor,  # [n_splits, N]
    mhc_mult: int,
    hidden_size: int,
    rms_eps: float,
) -> torch.Tensor:
    """Aggregate GEMM splits and RMS-normalize → mixes[N, mhc_mult3]."""
    # Sum over splits
    sqrsum = gemm_out_sqrsum.sum(dim=0)             # [N]
    mul_sum = gemm_out_mul.sum(dim=0)                # [N, mhc_mult3]
    inv_rms = torch.rsqrt(sqrsum / (mhc_mult * hidden_size) + rms_eps)  # [N]
    mixes = mul_sum * inv_rms.unsqueeze(-1)          # [N, mhc_mult3]
    return mixes


def _split_mixes_ref(
    mixes: torch.Tensor,       # [N, mhc_mult3]
    mhc_scale: torch.Tensor,   # [3]
    mhc_base: torch.Tensor,    # [mhc_mult3]
    mhc_mult: int,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split mixes → pre_mix, post_mix, comb_logit."""
    N = mixes.shape[0]
    mhc_mult2 = mhc_mult * mhc_mult

    # Build expanded scale
    scale = torch.cat([
        mhc_scale[0].expand(mhc_mult),
        mhc_scale[1].expand(mhc_mult),
        mhc_scale[2].expand(mhc_mult2),
    ])
    scaled = mixes * scale + mhc_base

    pre_logit = scaled[:, :mhc_mult]
    post_logit = scaled[:, mhc_mult:2 * mhc_mult]
    comb_logit = scaled[:, 2 * mhc_mult:]

    pre_mix = torch.sigmoid(pre_logit) + mhc_pre_eps           # [N, mhc_mult]
    post_mix = torch.sigmoid(post_logit) * mhc_post_mult_value  # [N, mhc_mult]
    return pre_mix, post_mix, comb_logit.view(N, mhc_mult, mhc_mult)


def _sinkhorn_ref(
    x: torch.Tensor,   # [N, K, K]
    repeat: int,
    eps: float,
) -> torch.Tensor:
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def _pre_apply_mix_ref(
    residual: torch.Tensor,  # [N, mhc_mult, H] bf16
    pre_mix: torch.Tensor,   # [N, mhc_mult] fp32
) -> torch.Tensor:
    """output[h] = Σ_m pre_mix[m] * residual[m, h]."""
    return (residual.float() * pre_mix.unsqueeze(-1)).sum(-2).bfloat16()


def stepwise_reference(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    residual: torch.Tensor,
    mhc_mult: int,
    hidden_size: int,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Complete step-by-step reference in pure torch."""
    mixes = _norm_fn_fwd_norm_ref(
        gemm_out_mul, gemm_out_sqrsum, mhc_mult, hidden_size, rms_eps,
    )
    pre_mix, post_mix, comb_logit = _split_mixes_ref(
        mixes, mhc_scale, mhc_base, mhc_mult, mhc_post_mult_value, mhc_pre_eps,
    )
    comb_mix = _sinkhorn_ref(comb_logit, sinkhorn_repeat, mhc_sinkhorn_eps)
    layer_input = _pre_apply_mix_ref(residual, pre_mix)

    return post_mix, comb_mix.flatten(-2, -1), layer_input


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_test_data(
    N: int,
    mhc_mult: int = 4,
    hidden_size: int = 4096,
    n_splits: int = 1,
    sinkhorn_repeat: int = 10,
    rms_eps: float = 1e-6,
    mhc_pre_eps: float = 1e-6,
    mhc_sinkhorn_eps: float = 1e-6,
    mhc_post_mult_value: float = 1.0,
) -> dict:
    device = 'cuda'
    mhc_mult3 = mhc_mult * 2 + mhc_mult * mhc_mult

    residual = torch.randn(N, mhc_mult, hidden_size, dtype=torch.float32, device=device).bfloat16()

    # Simulate GEMM output
    gemm_out_mul = torch.randn(n_splits, N, mhc_mult3, dtype=torch.float32, device=device) * 0.1
    gemm_out_sqrsum = torch.rand(n_splits, N, dtype=torch.float32, device=device) + 0.1

    mhc_scale = torch.randn(3, dtype=torch.float32, device=device) * 0.1
    mhc_base = torch.randn(mhc_mult3, dtype=torch.float32, device=device) * 0.1

    return dict(
        gemm_out_mul=gemm_out_mul,
        gemm_out_sqrsum=gemm_out_sqrsum,
        mhc_scale=mhc_scale,
        mhc_base=mhc_base,
        residual=residual,
        mhc_mult=mhc_mult,
        hidden_size=hidden_size,
        n_splits=n_splits,
        sinkhorn_repeat=sinkhorn_repeat,
        rms_eps=rms_eps,
        mhc_pre_eps=mhc_pre_eps,
        mhc_sinkhorn_eps=mhc_sinkhorn_eps,
        mhc_post_mult_value=mhc_post_mult_value,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('N', [512, 1024, 2048])
@pytest.mark.parametrize('hidden_size', [1280, 2560, 4096])
def test_correctness_vs_reference(N: int, hidden_size: int) -> None:
    td = generate_test_data(N=N, hidden_size=hidden_size)

    post_fused, comb_fused, layer_input_fused = pre_big_fuse_fwd_triton(
        td['gemm_out_mul'], td['gemm_out_sqrsum'],
        td['mhc_scale'], td['mhc_base'], td['residual'],
        hidden_size=td['hidden_size'],
        mhc_mult=td['mhc_mult'],
        n_splits=td['n_splits'],
        sinkhorn_repeat=td['sinkhorn_repeat'],
        rms_eps=td['rms_eps'],
        mhc_pre_eps=td['mhc_pre_eps'],
        mhc_sinkhorn_eps=td['mhc_sinkhorn_eps'],
        mhc_post_mult_value=td['mhc_post_mult_value'],
    )

    post_ref, comb_ref, layer_input_ref = stepwise_reference(
        td['gemm_out_mul'], td['gemm_out_sqrsum'],
        td['mhc_scale'], td['mhc_base'], td['residual'],
        td['mhc_mult'], td['hidden_size'],
        td['rms_eps'], td['mhc_pre_eps'], td['mhc_sinkhorn_eps'],
        td['mhc_post_mult_value'], td['sinkhorn_repeat'],
    )

    torch.testing.assert_close(post_fused, post_ref, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(comb_fused, comb_ref, atol=1e-5, rtol=1e-4)
    # bf16 output: FMA ordering may differ → allow 1 ULP
    torch.testing.assert_close(layer_input_fused, layer_input_ref, atol=0.0078125, rtol=0.008)


@pytest.mark.parametrize('N', [1, 7, 33, 128])
def test_small_token_counts(N: int) -> None:
    td = generate_test_data(N=N, hidden_size=1280)

    post_fused, comb_fused, layer_input_fused = pre_big_fuse_fwd_triton(
        td['gemm_out_mul'], td['gemm_out_sqrsum'],
        td['mhc_scale'], td['mhc_base'], td['residual'],
        hidden_size=td['hidden_size'],
        mhc_mult=td['mhc_mult'],
        n_splits=td['n_splits'],
        sinkhorn_repeat=td['sinkhorn_repeat'],
        rms_eps=td['rms_eps'],
        mhc_pre_eps=td['mhc_pre_eps'],
        mhc_sinkhorn_eps=td['mhc_sinkhorn_eps'],
        mhc_post_mult_value=td['mhc_post_mult_value'],
    )

    post_ref, comb_ref, layer_input_ref = stepwise_reference(
        td['gemm_out_mul'], td['gemm_out_sqrsum'],
        td['mhc_scale'], td['mhc_base'], td['residual'],
        td['mhc_mult'], td['hidden_size'],
        td['rms_eps'], td['mhc_pre_eps'], td['mhc_sinkhorn_eps'],
        td['mhc_post_mult_value'], td['sinkhorn_repeat'],
    )

    torch.testing.assert_close(post_fused, post_ref, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(comb_fused, comb_ref, atol=1e-5, rtol=1e-4)
    # bf16 output: FMA ordering may differ between Triton and torch -> allow 1 ULP
    torch.testing.assert_close(layer_input_fused, layer_input_ref, atol=0.0078125, rtol=0.008)


@pytest.mark.parametrize('sinkhorn_repeat', [1, 5, 10, 20])
def test_sinkhorn_repeat_values(sinkhorn_repeat: int) -> None:
    td = generate_test_data(N=256, hidden_size=1280, sinkhorn_repeat=sinkhorn_repeat)

    post_fused, comb_fused, layer_input_fused = pre_big_fuse_fwd_triton(
        td['gemm_out_mul'], td['gemm_out_sqrsum'],
        td['mhc_scale'], td['mhc_base'], td['residual'],
        hidden_size=td['hidden_size'],
        mhc_mult=td['mhc_mult'],
        n_splits=td['n_splits'],
        sinkhorn_repeat=sinkhorn_repeat,
        rms_eps=td['rms_eps'],
        mhc_pre_eps=td['mhc_pre_eps'],
        mhc_sinkhorn_eps=td['mhc_sinkhorn_eps'],
        mhc_post_mult_value=td['mhc_post_mult_value'],
    )

    post_ref, comb_ref, layer_input_ref = stepwise_reference(
        td['gemm_out_mul'], td['gemm_out_sqrsum'],
        td['mhc_scale'], td['mhc_base'], td['residual'],
        td['mhc_mult'], td['hidden_size'],
        td['rms_eps'], td['mhc_pre_eps'], td['mhc_sinkhorn_eps'],
        td['mhc_post_mult_value'], sinkhorn_repeat,
    )

    torch.testing.assert_close(post_fused, post_ref, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(comb_fused, comb_ref, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(layer_input_fused, layer_input_ref, atol=0.0078125, rtol=0.008)


@pytest.mark.parametrize('post_mult', [0.5, 1.0, 2.0])
def test_post_mult_values(post_mult: float) -> None:
    td = generate_test_data(N=256, hidden_size=1280, mhc_post_mult_value=post_mult)

    post_fused, comb_fused, _ = pre_big_fuse_fwd_triton(
        td['gemm_out_mul'], td['gemm_out_sqrsum'],
        td['mhc_scale'], td['mhc_base'], td['residual'],
        hidden_size=td['hidden_size'],
        mhc_mult=td['mhc_mult'],
        n_splits=td['n_splits'],
        sinkhorn_repeat=td['sinkhorn_repeat'],
        rms_eps=td['rms_eps'],
        mhc_pre_eps=td['mhc_pre_eps'],
        mhc_sinkhorn_eps=td['mhc_sinkhorn_eps'],
        mhc_post_mult_value=post_mult,
    )

    post_ref, comb_ref, _ = stepwise_reference(
        td['gemm_out_mul'], td['gemm_out_sqrsum'],
        td['mhc_scale'], td['mhc_base'], td['residual'],
        td['mhc_mult'], td['hidden_size'],
        td['rms_eps'], td['mhc_pre_eps'], td['mhc_sinkhorn_eps'],
        post_mult, td['sinkhorn_repeat'],
    )

    torch.testing.assert_close(post_fused, post_ref, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(comb_fused, comb_ref, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize('n_splits', [1, 4, 8, 16])
def test_multiple_splits(n_splits: int) -> None:
    td = generate_test_data(N=256, hidden_size=1280, n_splits=n_splits)

    post_fused, comb_fused, layer_input_fused = pre_big_fuse_fwd_triton(
        td['gemm_out_mul'], td['gemm_out_sqrsum'],
        td['mhc_scale'], td['mhc_base'], td['residual'],
        hidden_size=td['hidden_size'],
        mhc_mult=td['mhc_mult'],
        n_splits=n_splits,
        sinkhorn_repeat=td['sinkhorn_repeat'],
        rms_eps=td['rms_eps'],
        mhc_pre_eps=td['mhc_pre_eps'],
        mhc_sinkhorn_eps=td['mhc_sinkhorn_eps'],
        mhc_post_mult_value=td['mhc_post_mult_value'],
    )

    post_ref, comb_ref, layer_input_ref = stepwise_reference(
        td['gemm_out_mul'], td['gemm_out_sqrsum'],
        td['mhc_scale'], td['mhc_base'], td['residual'],
        td['mhc_mult'], td['hidden_size'],
        td['rms_eps'], td['mhc_pre_eps'], td['mhc_sinkhorn_eps'],
        td['mhc_post_mult_value'], td['sinkhorn_repeat'],
    )

    torch.testing.assert_close(post_fused, post_ref, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(comb_fused, comb_ref, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(layer_input_fused, layer_input_ref, atol=0.0078125, rtol=0.008)
