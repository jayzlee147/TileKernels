"""Triton-fused mhc_pre_big_fuse FWD for TileKernels.

Fuses the following pipeline into a single kernel (one program per token):
  1. norm_fn_fwd_norm:  aggregate GEMM splits → RMS normalize → mixes[mhc_mult3]
  2. pre_split_mixes_fwd (post & comb): sigmoid → post_mix, comb logits
  3. sinkhorn_fwd:      softmax + Sinkhorn normalization → comb_mix
  4. pre_split_mixes_fwd (pre): sigmoid + eps → pre_mix
  5. pre_apply_mix_fwd:  weighted sum of residual → layer_input

This is a forward-only inference kernel (no backward).

Design:
  - mhc_mult3 = 24 fits comfortably in registers (K_BLK = 32, next power of 2).
  - mhc_mult = 4, so sinkhorn operates on a 4×4 matrix in registers.
  - The hidden-dimension loop for pre_apply_mix is tiled with H_BLK.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused FWD kernel
# ---------------------------------------------------------------------------


@triton.jit
def _pre_big_fuse_fwd_kernel(
    # Inputs
    GEMM_MUL_PTR,     # [n_splits, N, mhc_mult3]  fp32
    GEMM_SQRSUM_PTR,  # [n_splits, N]              fp32
    SCALE_PTR,         # [3]                        fp32
    BASE_PTR,          # [mhc_mult3]                fp32
    RESIDUAL_PTR,      # [N, mhc_mult, hidden_size] bf16
    # Outputs
    POST_MIX_PTR,      # [N, mhc_mult]              fp32
    COMB_MIX_PTR,      # [N, mhc_mult * mhc_mult]   fp32
    LAYER_INPUT_PTR,   # [N, hidden_size]            bf16
    # Scalars
    N,
    HIDDEN_SIZE: tl.constexpr,
    MHC_MULT: tl.constexpr,       # 4
    MHC_MULT3: tl.constexpr,      # 24 = 4*2 + 4*4
    K_BLK: tl.constexpr,          # next_power_of_2(MHC_MULT3), e.g. 32
    N_SPLITS: tl.constexpr,
    SINKHORN_REPEAT: tl.constexpr,
    H_BLK: tl.constexpr,
    RMS_EPS: tl.constexpr,
    MHC_PRE_EPS: tl.constexpr,
    MHC_SINKHORN_EPS: tl.constexpr,
    MHC_POST_MULT_VALUE: tl.constexpr,
):
    pid_n = tl.program_id(0)

    # ================================================================
    # Stage 1: norm_fn_fwd_norm — aggregate splits → mixes
    # ================================================================

    # Sum sqrsum across splits
    rms_acc = tl.zeros([], dtype=tl.float32)
    for i_split in range(N_SPLITS):
        sqrsum_val = tl.load(
            GEMM_SQRSUM_PTR + i_split * N + pid_n,
        )
        rms_acc += sqrsum_val

    # Compute rsqrt normalization
    # total_hidden = mhc_mult * hidden_size
    inv_rms = tl.rsqrt(rms_acc / (MHC_MULT * HIDDEN_SIZE) + RMS_EPS)

    # Sum gemm_out_mul across splits → mixes
    k_offs = tl.arange(0, K_BLK)
    k_mask = k_offs < MHC_MULT3
    mixes = tl.zeros([K_BLK], dtype=tl.float32)
    for i_split in range(N_SPLITS):
        mul_vals = tl.load(
            GEMM_MUL_PTR + i_split * N * MHC_MULT3 + pid_n * MHC_MULT3 + k_offs,
            mask=k_mask,
            other=0.0,
        )
        mixes += mul_vals
    mixes *= inv_rms

    # Load scale and base
    scale0 = tl.load(SCALE_PTR + 0)
    scale1 = tl.load(SCALE_PTR + 1)
    scale2 = tl.load(SCALE_PTR + 2)

    base_vals = tl.load(BASE_PTR + k_offs, mask=k_mask, other=0.0)

    # ================================================================
    # Stage 2: pre_split_mixes_fwd (post & comb parts)
    # ================================================================

    # Layout of mixes[mhc_mult3]:
    #   [0..MHC_MULT)              → pre logits
    #   [MHC_MULT..2*MHC_MULT)     → post logits
    #   [2*MHC_MULT..mhc_mult3)    → comb logits (MHC_MULT * MHC_MULT values)

    # --- Post mix: sigmoid(logit * scale1 + base) * post_mult ---
    m_offs = tl.arange(0, MHC_MULT)  # [MHC_MULT]

    # Extract post logits from mixes
    post_logit_raw = tl.zeros([MHC_MULT], dtype=tl.float32)
    for j in tl.static_range(MHC_MULT):
        idx = MHC_MULT + j
        # Extract the j-th post logit from mixes
        post_logit_raw = tl.where(m_offs == j,
                                  tl.sum(tl.where(k_offs == idx, mixes, 0.0)),
                                  post_logit_raw)

    # Extract post base
    post_base = tl.zeros([MHC_MULT], dtype=tl.float32)
    for j in tl.static_range(MHC_MULT):
        idx = MHC_MULT + j
        post_base = tl.where(m_offs == j,
                             tl.load(BASE_PTR + idx),
                             post_base)

    post_out = tl.sigmoid(post_logit_raw * scale1 + post_base) * MHC_POST_MULT_VALUE

    # Store post_mix: [N, MHC_MULT]
    tl.store(
        POST_MIX_PTR + pid_n * MHC_MULT + m_offs,
        post_out,
        mask=m_offs < MHC_MULT,
    )

    # --- Comb mix: extract comb logits → sinkhorn ---
    MHC_MULT2: tl.constexpr = MHC_MULT * MHC_MULT
    c_offs = tl.arange(0, MHC_MULT2)  # [MHC_MULT2] = [16]

    # Extract comb logits from mixes
    comb_logit = tl.zeros([MHC_MULT2], dtype=tl.float32)
    for j in tl.static_range(MHC_MULT2):
        idx = 2 * MHC_MULT + j
        comb_logit = tl.where(c_offs == j,
                              tl.sum(tl.where(k_offs == idx, mixes, 0.0)),
                              comb_logit)

    # Extract comb base
    comb_base = tl.zeros([MHC_MULT2], dtype=tl.float32)
    for j in tl.static_range(MHC_MULT2):
        idx = 2 * MHC_MULT + j
        comb_base = tl.where(c_offs == j,
                             tl.load(BASE_PTR + idx),
                             comb_base)

    comb_logit = comb_logit * scale2 + comb_base

    # ================================================================
    # Stage 3: Sinkhorn normalization on comb_logit [MHC_MULT, MHC_MULT]
    # ================================================================
    # comb_logit is flat [MHC_MULT2]. We treat rows/cols explicitly.
    # Row j elements: c_offs // MHC_MULT == j
    # Col k elements: c_offs % MHC_MULT == k

    row_idx = c_offs // MHC_MULT  # which row each element belongs to
    col_idx = c_offs % MHC_MULT   # which col each element belongs to

    # --- softmax per row ---
    # For each row, subtract max then exp then divide by sum
    cm = comb_logit

    # Row max
    for j in tl.static_range(MHC_MULT):
        row_mask_j = (row_idx == j)
        row_max_j = tl.max(tl.where(row_mask_j, cm, float('-inf')))
        cm = tl.where(row_mask_j, tl.exp(cm - row_max_j), cm)

    # Row sum and normalize
    for j in tl.static_range(MHC_MULT):
        row_mask_j = (row_idx == j)
        row_sum_j = tl.sum(tl.where(row_mask_j, cm, 0.0))
        cm = tl.where(row_mask_j, cm / row_sum_j + MHC_SINKHORN_EPS, cm)

    # Col normalize: x / (col_sum + eps)
    for k in tl.static_range(MHC_MULT):
        col_mask_k = (col_idx == k)
        col_sum_k = tl.sum(tl.where(col_mask_k, cm, 0.0))
        cm = tl.where(col_mask_k, cm / (col_sum_k + MHC_SINKHORN_EPS), cm)

    # Remaining sinkhorn iterations
    for _ in tl.static_range(SINKHORN_REPEAT - 1):
        # Row normalize: x / (row_sum + eps)
        for j in tl.static_range(MHC_MULT):
            row_mask_j = (row_idx == j)
            row_sum_j = tl.sum(tl.where(row_mask_j, cm, 0.0))
            cm = tl.where(row_mask_j, cm / (row_sum_j + MHC_SINKHORN_EPS), cm)

        # Col normalize: x / (col_sum + eps)
        for k in tl.static_range(MHC_MULT):
            col_mask_k = (col_idx == k)
            col_sum_k = tl.sum(tl.where(col_mask_k, cm, 0.0))
            cm = tl.where(col_mask_k, cm / (col_sum_k + MHC_SINKHORN_EPS), cm)

    # Store comb_mix: [N, MHC_MULT2]
    tl.store(
        COMB_MIX_PTR + pid_n * MHC_MULT2 + c_offs,
        cm,
        mask=c_offs < MHC_MULT2,
    )

    # ================================================================
    # Stage 4: pre_split_mixes_fwd (pre part)
    # ================================================================

    # Extract pre logits from mixes (first MHC_MULT elements)
    pre_logit = tl.zeros([MHC_MULT], dtype=tl.float32)
    for j in tl.static_range(MHC_MULT):
        pre_logit = tl.where(m_offs == j,
                             tl.sum(tl.where(k_offs == j, mixes, 0.0)),
                             pre_logit)

    # Extract pre base
    pre_base = tl.zeros([MHC_MULT], dtype=tl.float32)
    for j in tl.static_range(MHC_MULT):
        pre_base = tl.where(m_offs == j,
                            tl.load(BASE_PTR + j),
                            pre_base)

    pre_mix = tl.sigmoid(pre_logit * scale0 + pre_base) + MHC_PRE_EPS

    # ================================================================
    # Stage 5: pre_apply_mix_fwd
    #   layer_input[h] = Σ_m pre_mix[m] * residual[m, h]
    # ================================================================

    # Load pre_mix scalars individually for use in the loop
    # pre_mix is [MHC_MULT] vector; extract scalars
    pm0 = tl.sum(tl.where(m_offs == 0, pre_mix, 0.0))
    pm1 = tl.sum(tl.where(m_offs == 1, pre_mix, 0.0))
    pm2 = tl.sum(tl.where(m_offs == 2, pre_mix, 0.0))
    pm3 = tl.sum(tl.where(m_offs == 3, pre_mix, 0.0))

    res_base = pid_n * MHC_MULT * HIDDEN_SIZE
    out_base = pid_n * HIDDEN_SIZE

    for h_start in tl.static_range(0, HIDDEN_SIZE, H_BLK):
        h_offs = h_start + tl.arange(0, H_BLK)

        # Load residual[pid_n, m, h_offs] for each m
        x0 = tl.load(RESIDUAL_PTR + res_base + 0 * HIDDEN_SIZE + h_offs).to(tl.float32)
        x1 = tl.load(RESIDUAL_PTR + res_base + 1 * HIDDEN_SIZE + h_offs).to(tl.float32)
        x2 = tl.load(RESIDUAL_PTR + res_base + 2 * HIDDEN_SIZE + h_offs).to(tl.float32)
        x3 = tl.load(RESIDUAL_PTR + res_base + 3 * HIDDEN_SIZE + h_offs).to(tl.float32)

        o = pm0 * x0 + pm1 * x1 + pm2 * x2 + pm3 * x3

        tl.store(LAYER_INPUT_PTR + out_base + h_offs, o.to(tl.bfloat16))


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _pick_h_blk(hidden_size: int, default: int = 512) -> int:
    return math.gcd(hidden_size, default)


def pre_big_fuse_fwd_triton(
    gemm_out_mul: torch.Tensor,     # [n_splits, N, mhc_mult3]  fp32
    gemm_out_sqrsum: torch.Tensor,  # [n_splits, N]              fp32
    mhc_scale: torch.Tensor,        # [3]                        fp32
    mhc_base: torch.Tensor,         # [mhc_mult3]                fp32
    residual: torch.Tensor,         # [N, mhc_mult, hidden_size] bf16
    *,
    hidden_size: int,
    mhc_mult: int = 4,
    n_splits: int = 1,
    sinkhorn_repeat: int = 10,
    rms_eps: float = 1e-6,
    mhc_pre_eps: float = 1e-6,
    mhc_sinkhorn_eps: float = 1e-6,
    mhc_post_mult_value: float = 1.0,
    post_mix: torch.Tensor | None = None,
    comb_mix: torch.Tensor | None = None,
    layer_input: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the fused FWD Triton kernel for mhc_pre_big_fuse."""
    N = residual.shape[0]
    mhc_mult3 = mhc_mult * 2 + mhc_mult * mhc_mult
    mhc_mult2 = mhc_mult * mhc_mult

    assert gemm_out_mul.shape == (n_splits, N, mhc_mult3)
    assert gemm_out_sqrsum.shape == (n_splits, N)
    assert mhc_scale.shape == (3,)
    assert mhc_base.shape == (mhc_mult3,)
    assert residual.shape == (N, mhc_mult, hidden_size)

    if post_mix is None:
        post_mix = torch.empty(N, mhc_mult, dtype=torch.float32, device=residual.device)
    if comb_mix is None:
        comb_mix = torch.empty(N, mhc_mult2, dtype=torch.float32, device=residual.device)
    if layer_input is None:
        layer_input = torch.empty(N, hidden_size, dtype=torch.bfloat16, device=residual.device)

    K_BLK = _next_power_of_2(mhc_mult3)
    H_BLK = _pick_h_blk(hidden_size)

    grid = (N,)

    _pre_big_fuse_fwd_kernel[grid](
        gemm_out_mul, gemm_out_sqrsum,
        mhc_scale, mhc_base,
        residual,
        post_mix, comb_mix, layer_input,
        N,
        HIDDEN_SIZE=hidden_size,
        MHC_MULT=mhc_mult,
        MHC_MULT3=mhc_mult3,
        K_BLK=K_BLK,
        N_SPLITS=n_splits,
        SINKHORN_REPEAT=sinkhorn_repeat,
        H_BLK=H_BLK,
        RMS_EPS=rms_eps,
        MHC_PRE_EPS=mhc_pre_eps,
        MHC_SINKHORN_EPS=mhc_sinkhorn_eps,
        MHC_POST_MULT_VALUE=mhc_post_mult_value,
    )

    return post_mix, comb_mix, layer_input


# ---------------------------------------------------------------------------
# High-level wrapper matching the existing mhc_pre_big_fuse interface
# ---------------------------------------------------------------------------


def mhc_pre_big_fuse_triton(
    residual: torch.Tensor,
    fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton-fused mhc_pre_big_fuse — drop-in replacement for the tilelang version.

    This replicates the full pipeline: GEMM (via tilelang) → fused norm+split+sinkhorn+apply.

    Args:
        residual: [*, mhc_mult, hidden_size] bf16
        fn: [mhc_mult3, mhc_mult * hidden_size] fp32
        mhc_scale: [3] fp32
        mhc_base: [mhc_mult3] fp32
        rms_eps, mhc_pre_eps, mhc_sinkhorn_eps: float
        mhc_post_mult_value: float
        sinkhorn_repeat: int
        n_splits: int

    Returns:
        post_mix:    [*, mhc_mult, 1]           fp32
        comb_mix:    [*, mhc_mult, mhc_mult]    fp32
        layer_input: [*, hidden_size]            bf16
    """
    from tile_kernels.mhc.norm_fn_kernel import _mhc_pre_norm_fn_fwd_mul

    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32

    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2
    mhc_hidden_size = mhc_mult * hidden_size

    outer_shape = residual.shape[:-2]
    residual_flat = residual.reshape(-1, mhc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    # --- Stage 0: GEMM (reuse tilelang) ---
    # TileLang doesn't support split-k, force n_splits=1
    n_splits_actual = 1
    gemm_out_mul = torch.empty(
        n_splits_actual, num_tokens, mhc_mult3,
        dtype=torch.float32, device=residual.device,
    )
    gemm_out_sqrsum = torch.empty(
        n_splits_actual, num_tokens,
        dtype=torch.float32, device=residual.device,
    )

    fn_bf16 = fn.bfloat16()
    fwd_mul_kernel = _mhc_pre_norm_fn_fwd_mul(mhc_mult3, 1, mhc_hidden_size)
    fwd_mul_kernel(
        residual_flat.reshape(-1, mhc_hidden_size),
        fn_bf16,
        gemm_out_mul.reshape(-1, 1, mhc_mult3),
        gemm_out_sqrsum.reshape(-1, 1),
    )

    # --- Stage 1-5: Fused Triton kernel ---
    post_mix, comb_mix, layer_input = pre_big_fuse_fwd_triton(
        gemm_out_mul, gemm_out_sqrsum,
        mhc_scale, mhc_base,
        residual_flat,
        hidden_size=hidden_size,
        mhc_mult=mhc_mult,
        n_splits=n_splits_actual,
        sinkhorn_repeat=sinkhorn_repeat,
        rms_eps=rms_eps,
        mhc_pre_eps=mhc_pre_eps,
        mhc_sinkhorn_eps=mhc_sinkhorn_eps,
        mhc_post_mult_value=mhc_post_mult_value,
    )

    post_mix = post_mix.reshape(*outer_shape, mhc_mult, 1)
    comb_mix = comb_mix.reshape(*outer_shape, mhc_mult, mhc_mult)
    layer_input = layer_input.reshape(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input


__all__ = [
    "pre_big_fuse_fwd_triton",
    "mhc_pre_big_fuse_triton",
]
