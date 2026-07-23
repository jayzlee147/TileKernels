"""Triton-fused pre_split_mixes FWD/BWD for TileKernels.

Math (FWD):
    mhc_scale_expanded = cat([scale[0].expand(K), scale[1].expand(K), scale[2].expand(K*K)])
    input_mixes = input_mixes * mhc_scale_expanded + mhc_base

    pre_layer_mix  = sigmoid(input_mixes[:, :K]) + mhc_pre_eps
    post_layer_mix = sigmoid(input_mixes[:, K:2K]) * mhc_post_mult_value
    comb_res_mix   = input_mixes[:, 2K:]  (identity, just view as [K, K])

Math (BWD):
    d_pre_logit  = d_pre  * sig_pre * (1 - sig_pre)
    d_post_logit = d_post * sig_post * (1 - sig_post) * mhc_post_mult_value
                 = d_post * post_out * (1 - post_out / mhc_post_mult_value)
    d_comb_logit = d_comb  (identity)

    d_input_mixes[:, :K]   = d_pre_logit  * scale[0]
    d_input_mixes[:, K:2K] = d_post_logit * scale[1]
    d_input_mixes[:, 2K:]  = d_comb_logit * scale[2]

    d_base = sum_n(d_*_logit)
    d_scale[i] = sum(input_mixes_slice_i * d_*_logit_i)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# FWD kernel
# ---------------------------------------------------------------------------


@triton.jit
def _pre_split_mixes_fwd_kernel(
    INPUT_MIXES_PTR,  # [N, (2+K)*K] contiguous, fp32
    SCALE_PTR,        # [3] fp32
    BASE_PTR,         # [(2+K)*K] fp32
    PRE_PTR,          # [N, K] output fp32
    POST_PTR,         # [N, K] output fp32
    COMB_PTR,         # [N, K*K] output fp32
    N,
    MHC_POST_MULT_VALUE: tl.constexpr,
    MHC_PRE_EPS: tl.constexpr,
    K: tl.constexpr,
    BLOCK_LEADING: tl.constexpr,
):
    """One program tile of BLOCK_LEADING rows.

    Reads input_mixes[BLOCK_LEADING, (2+K)*K], applies scale+base FMA,
    sigmoid for pre/post slices, identity for comb, writes three outputs.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_LEADING + tl.arange(0, BLOCK_LEADING)
    mask_leading = offs < N

    k_idx = tl.arange(0, K)
    KK: tl.constexpr = K * K
    KK_TOTAL: tl.constexpr = (2 + K) * K

    # Load the three scale scalars.
    scale0 = tl.load(SCALE_PTR + 0)
    scale1 = tl.load(SCALE_PTR + 1)
    scale2 = tl.load(SCALE_PTR + 2)

    # Load base partitions.
    base_pre = tl.load(BASE_PTR + k_idx)
    base_post = tl.load(BASE_PTR + K + k_idx)
    # base_comb is K*K elements stored contiguously.
    base_comb = tl.load(BASE_PTR + 2 * K + tl.arange(0, KK))

    # Load input_mixes slices.
    pre_logit = tl.load(
        INPUT_MIXES_PTR + offs[:, None] * KK_TOTAL + k_idx[None, :],
        mask=mask_leading[:, None],
        other=0.0,
    )
    post_logit = tl.load(
        INPUT_MIXES_PTR + offs[:, None] * KK_TOTAL + K + k_idx[None, :],
        mask=mask_leading[:, None],
        other=0.0,
    )
    comb_logit = tl.load(
        INPUT_MIXES_PTR + offs[:, None] * KK_TOTAL + 2 * K + tl.arange(0, KK)[None, :],
        mask=mask_leading[:, None],
        other=0.0,
    )

    # Apply scale + base (FMA).
    pre_logit = pre_logit * scale0 + base_pre[None, :]
    post_logit = post_logit * scale1 + base_post[None, :]
    comb_logit = comb_logit * scale2 + base_comb[None, :]

    # Sigmoid for pre and post.
    pre_out = tl.sigmoid(pre_logit) + MHC_PRE_EPS
    post_out = tl.sigmoid(post_logit) * MHC_POST_MULT_VALUE

    # Comb is identity (just the affine-transformed logit).
    comb_out = comb_logit

    # Store outputs.
    tl.store(
        PRE_PTR + offs[:, None] * K + k_idx[None, :],
        pre_out,
        mask=mask_leading[:, None],
    )
    tl.store(
        POST_PTR + offs[:, None] * K + k_idx[None, :],
        post_out,
        mask=mask_leading[:, None],
    )
    tl.store(
        COMB_PTR + offs[:, None] * KK + tl.arange(0, KK)[None, :],
        comb_out,
        mask=mask_leading[:, None],
    )


# ---------------------------------------------------------------------------
# BWD kernel
# ---------------------------------------------------------------------------


@triton.jit
def _pre_split_mixes_bwd_kernel(
    DPRE_PTR,         # [N, K] upstream grad fp32
    DPOST_PTR,        # [N, K] upstream grad fp32
    DCOMB_PTR,        # [N, K*K] upstream grad fp32
    INPUT_MIXES_PTR,  # [N, (2+K)*K] original input fp32
    POST_OUT_PTR,     # [N, K] saved post_layer_mix fp32
    SCALE_PTR,        # [3] fp32
    BASE_PTR,         # [(2+K)*K] fp32
    DLOGITS_PTR,      # [N, (2+K)*K] output grad fp32
    DBASE_PTR,        # [N, (2+K)*K] per-row partials for base grad
    N,
    MHC_POST_MULT_VALUE: tl.constexpr,
    K: tl.constexpr,
    BLOCK_LEADING: tl.constexpr,
):
    """VJP through the elemwise pre_split_mixes.

    Forward:
        pre_logit  = input_mixes[:, :K]   * s0 + base[:K]
        post_logit = input_mixes[:, K:2K] * s1 + base[K:2K]
        comb_logit = input_mixes[:, 2K:]  * s2 + base[2K:]

        pre  = sigmoid(pre_logit) + eps
        post = sigmoid(post_logit) * post_mult_value
        comb = comb_logit  (identity)

    Backward:
        d_pre_logit  = d_pre * sig_pre * (1 - sig_pre)
        d_post_logit = d_post * post_out * (1 - post_out / post_mult_value)
        d_comb_logit = d_comb

        d_input_mixes = d_*_logit * scale
        d_base = d_*_logit  (per-row partials, host reduces)
        d_scale = input_mixes * d_*_logit  (host reduces)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_LEADING + tl.arange(0, BLOCK_LEADING)
    mask_leading = offs < N

    k_idx = tl.arange(0, K)
    KK: tl.constexpr = K * K
    KK_TOTAL: tl.constexpr = (2 + K) * K

    # Read scale.
    scale0 = tl.load(SCALE_PTR + 0)
    scale1 = tl.load(SCALE_PTR + 1)
    scale2 = tl.load(SCALE_PTR + 2)

    # Load base to recompute sigmoid for pre slice.
    base_pre = tl.load(BASE_PTR + k_idx)

    # Load input_mixes (original, pre-scale) for d_scale computation.
    input_pre = tl.load(
        INPUT_MIXES_PTR + offs[:, None] * KK_TOTAL + k_idx[None, :],
        mask=mask_leading[:, None],
        other=0.0,
    )
    input_post = tl.load(
        INPUT_MIXES_PTR + offs[:, None] * KK_TOTAL + K + k_idx[None, :],
        mask=mask_leading[:, None],
        other=0.0,
    )
    input_comb = tl.load(
        INPUT_MIXES_PTR + offs[:, None] * KK_TOTAL + 2 * K + tl.arange(0, KK)[None, :],
        mask=mask_leading[:, None],
        other=0.0,
    )

    # Load saved post_out for sigmoid VJP.
    post_out = tl.load(
        POST_OUT_PTR + offs[:, None] * K + k_idx[None, :],
        mask=mask_leading[:, None],
        other=0.0,
    )

    # Load upstream grads.
    d_pre = tl.load(
        DPRE_PTR + offs[:, None] * K + k_idx[None, :],
        mask=mask_leading[:, None],
        other=0.0,
    )
    d_post = tl.load(
        DPOST_PTR + offs[:, None] * K + k_idx[None, :],
        mask=mask_leading[:, None],
        other=0.0,
    )
    d_comb = tl.load(
        DCOMB_PTR + offs[:, None] * KK + tl.arange(0, KK)[None, :],
        mask=mask_leading[:, None],
        other=0.0,
    )

    # Sigmoid VJP for pre: recompute sigmoid from original input.
    pre_sig = tl.sigmoid(input_pre * scale0 + base_pre[None, :])
    d_pre_logit = d_pre * pre_sig * (1.0 - pre_sig)

    # Sigmoid VJP for post: use saved post_out.
    # post_out = sigmoid(post_logit) * post_mult_value
    # sigmoid(post_logit) = post_out / post_mult_value
    # d_post_logit = d_post * post_mult_value * sig * (1 - sig)
    #              = d_post * post_out * (1 - post_out / post_mult_value)
    d_post_logit = d_post * post_out * (1.0 - post_out / MHC_POST_MULT_VALUE)

    # Identity VJP for comb.
    d_comb_logit = d_comb

    # Write d_input_mixes = d_*_logit * scale.
    tl.store(
        DLOGITS_PTR + offs[:, None] * KK_TOTAL + k_idx[None, :],
        d_pre_logit * scale0,
        mask=mask_leading[:, None],
    )
    tl.store(
        DLOGITS_PTR + offs[:, None] * KK_TOTAL + K + k_idx[None, :],
        d_post_logit * scale1,
        mask=mask_leading[:, None],
    )
    tl.store(
        DLOGITS_PTR + offs[:, None] * KK_TOTAL + 2 * K + tl.arange(0, KK)[None, :],
        d_comb_logit * scale2,
        mask=mask_leading[:, None],
    )

    # Write d_base per-row partials (= d_*_logit, no scale factor).
    tl.store(
        DBASE_PTR + offs[:, None] * KK_TOTAL + k_idx[None, :],
        d_pre_logit,
        mask=mask_leading[:, None],
    )
    tl.store(
        DBASE_PTR + offs[:, None] * KK_TOTAL + K + k_idx[None, :],
        d_post_logit,
        mask=mask_leading[:, None],
    )
    tl.store(
        DBASE_PTR + offs[:, None] * KK_TOTAL + 2 * K + tl.arange(0, KK)[None, :],
        d_comb_logit,
        mask=mask_leading[:, None],
    )


# ---------------------------------------------------------------------------
# Block-leading heuristic
# ---------------------------------------------------------------------------


def _pick_block_leading(n: int, k: int) -> int:
    """Pick BLOCK_LEADING based on K and work-axis size."""
    if k <= 4:
        cap = 64
    elif k <= 8:
        cap = 32
    else:
        cap = 8
    if n < cap:
        return max(1, triton.next_power_of_2(n))
    return cap


# ---------------------------------------------------------------------------
# Python wrapper functions
# ---------------------------------------------------------------------------


def pre_split_mixes_fwd_triton(
    input_mixes: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_mult: int,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the Triton FWD kernel for pre_split_mixes.

    Args:
        input_mixes: [N, (2+K)*K] contiguous fp32 tensor (N = num_tokens).
        mhc_scale: [3] fp32 tensor.
        mhc_base: [(2+K)*K] fp32 tensor.
        mhc_mult: K (number of heads).
        mhc_post_mult_value: scalar multiplier for post sigmoid.
        mhc_pre_eps: epsilon added to pre sigmoid output.

    Returns:
        (pre_layer_mix, post_layer_mix, comb_res_mix)
        pre:  [N, K] fp32
        post: [N, K] fp32
        comb: [N, K*K] fp32
    """
    K = mhc_mult
    KK_TOTAL = (2 + K) * K

    input_mixes = input_mixes.contiguous().to(torch.float32)
    mhc_scale = mhc_scale.contiguous().to(torch.float32)
    mhc_base = mhc_base.contiguous().to(torch.float32)

    assert input_mixes.ndim == 2 and input_mixes.shape[1] == KK_TOTAL
    N = input_mixes.shape[0]

    device = input_mixes.device
    pre = torch.empty((N, K), dtype=torch.float32, device=device)
    post = torch.empty((N, K), dtype=torch.float32, device=device)
    comb = torch.empty((N, K * K), dtype=torch.float32, device=device)

    block_leading = _pick_block_leading(N, K)
    grid = (triton.cdiv(N, block_leading),)
    _pre_split_mixes_fwd_kernel[grid](
        input_mixes,
        mhc_scale,
        mhc_base,
        pre,
        post,
        comb,
        N,
        MHC_POST_MULT_VALUE=float(mhc_post_mult_value),
        MHC_PRE_EPS=float(mhc_pre_eps),
        K=K,
        BLOCK_LEADING=block_leading,
    )

    return pre, post, comb


def pre_split_mixes_bwd_triton(
    d_pre: torch.Tensor,
    d_post: torch.Tensor,
    d_comb: torch.Tensor,
    input_mixes: torch.Tensor,
    post_out: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_mult: int,
    mhc_post_mult_value: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the Triton BWD kernel for pre_split_mixes.

    Args:
        d_pre:  [N, K] upstream grad fp32.
        d_post: [N, K] upstream grad fp32.
        d_comb: [N, K*K] upstream grad fp32.
        input_mixes: [N, (2+K)*K] original input fp32.
        post_out: [N, K] saved post_layer_mix from FWD.
        mhc_scale: [3] fp32.
        mhc_base: [(2+K)*K] fp32.
        mhc_mult: K.
        mhc_post_mult_value: scalar multiplier.

    Returns:
        (d_input_mixes, d_scale, d_base)
    """
    K = mhc_mult
    KK_TOTAL = (2 + K) * K
    N = input_mixes.shape[0]

    d_pre = d_pre.contiguous().to(torch.float32)
    d_post = d_post.contiguous().to(torch.float32)
    d_comb = d_comb.contiguous().to(torch.float32)

    device = input_mixes.device
    d_logits = torch.empty_like(input_mixes)
    d_base_partials = torch.empty((N, KK_TOTAL), dtype=torch.float32, device=device)

    block_leading = _pick_block_leading(N, K)
    grid = (triton.cdiv(N, block_leading),)
    _pre_split_mixes_bwd_kernel[grid](
        d_pre,
        d_post,
        d_comb,
        input_mixes,
        post_out,
        mhc_scale,
        mhc_base,
        d_logits,
        d_base_partials,
        N,
        MHC_POST_MULT_VALUE=float(mhc_post_mult_value),
        K=K,
        BLOCK_LEADING=block_leading,
    )

    # Host-side reductions.
    d_base = d_base_partials.sum(dim=0)
    d_scale_0 = (input_mixes[:, :K] * d_base_partials[:, :K]).sum()
    d_scale_1 = (input_mixes[:, K:2 * K] * d_base_partials[:, K:2 * K]).sum()
    d_scale_2 = (input_mixes[:, 2 * K:] * d_base_partials[:, 2 * K:]).sum()
    d_scale = torch.stack([d_scale_0, d_scale_1, d_scale_2])

    return d_logits, d_scale, d_base


# ---------------------------------------------------------------------------
# torch.autograd.Function wrapper
# ---------------------------------------------------------------------------


class PreSplitMixesTritonFn(torch.autograd.Function):
    """Autograd wrapper for the Triton pre_split_mixes FWD/BWD kernels."""

    @staticmethod
    def forward(
        ctx,
        input_mixes: torch.Tensor,
        mhc_scale: torch.Tensor,
        mhc_base: torch.Tensor,
        mhc_mult: int,
        mhc_post_mult_value: float,
        mhc_pre_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        K = mhc_mult
        mhc_mult2 = K * K
        mhc_mult3 = K * 2 + mhc_mult2

        assert input_mixes.ndim == 3
        tokens_shape = input_mixes.shape[:2]

        input_mixes_flat = input_mixes.view(-1, mhc_mult3).contiguous().to(torch.float32)
        mhc_scale_c = mhc_scale.contiguous().to(torch.float32)
        mhc_base_c = mhc_base.contiguous().to(torch.float32)

        pre, post, comb = pre_split_mixes_fwd_triton(
            input_mixes_flat, mhc_scale_c, mhc_base_c,
            K, mhc_post_mult_value, mhc_pre_eps,
        )

        ctx.save_for_backward(input_mixes_flat, post, mhc_scale_c, mhc_base_c)
        ctx.mhc_mult = K
        ctx.mhc_post_mult_value = mhc_post_mult_value
        ctx.tokens_shape = tuple(tokens_shape)

        pre = pre.view(*tokens_shape, K, 1)
        post = post.view(*tokens_shape, K, 1)
        comb = comb.view(*tokens_shape, K, K)

        return pre, post, comb

    @staticmethod
    def backward(ctx, d_pre, d_post, d_comb):
        input_mixes_flat, post_out, mhc_scale_c, mhc_base_c = ctx.saved_tensors
        K = ctx.mhc_mult
        num_tokens = input_mixes_flat.shape[0]

        d_logits, d_scale, d_base = pre_split_mixes_bwd_triton(
            d_pre.reshape(num_tokens, K),
            d_post.reshape(num_tokens, K),
            d_comb.reshape(num_tokens, K * K),
            input_mixes_flat,
            post_out,
            mhc_scale_c,
            mhc_base_c,
            K,
            ctx.mhc_post_mult_value,
        )

        d_logits = d_logits.view(*ctx.tokens_shape, d_logits.shape[-1])

        return d_logits, d_scale, d_base, None, None, None


def mhc_pre_split_mixes_triton(
    input_mixes: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_mult: int,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton-fused pre_split_mixes (TileKernels math).

    Args:
        input_mixes: [a, b, (2+K)*K] input tensor.
        mhc_scale: [3] scale parameters.
        mhc_base: [(2+K)*K] base/bias parameters.
        mhc_mult: K (number of heads).
        mhc_post_mult_value: scalar multiplier for post sigmoid.
        mhc_pre_eps: epsilon added to pre sigmoid output.

    Returns:
        (pre_layer_mix, post_layer_mix, comb_res_mix)
        pre:  [a, b, K, 1]
        post: [a, b, K, 1]
        comb: [a, b, K, K]
    """
    return PreSplitMixesTritonFn.apply(
        input_mixes,
        mhc_scale,
        mhc_base,
        mhc_mult,
        mhc_post_mult_value,
        mhc_pre_eps,
    )


__all__ = [
    "pre_split_mixes_fwd_triton",
    "pre_split_mixes_bwd_triton",
    "mhc_pre_split_mixes_triton",
    "PreSplitMixesTritonFn",
]
