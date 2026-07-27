"""Triton-fused HC kernel: pre_split_mixes + sinkhorn in a single kernel call.

Fused forward math:
    # --- pre_split_mixes ---
    mhc_scale_expanded = cat([scale[0].expand(K), scale[1].expand(K), scale[2].expand(K*K)])
    input_mixes = input_mixes * mhc_scale_expanded + mhc_base

    pre  = sigmoid(input_mixes[:, :K]) + mhc_pre_eps           # [N, K]
    post = sigmoid(input_mixes[:, K:2K]) * mhc_post_mult_value  # [N, K]
    comb = input_mixes[:, 2K:].view(N, K, K)                    # [N, K, K]

    # --- sinkhorn on comb ---
    comb = softmax(comb, dim=-1) + sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + sinkhorn_eps)

State trajectory saved by FWD for BWD (total = 2 * REPEAT + 1 states per sample):
    state 0: sm = softmax(comb_logit)         (no eps, needed for softmax VJP)
    state 1: sm + eps                         (input to first col normalize)
    state 2: after first col normalize
    state 3+2*i: after row normalize i        (i = 0..REPEAT-2)
    state 4+2*i: after col normalize i        (i = 0..REPEAT-2)

BWD VJPs:
    Normalization y = x / s, s = sum(x, axis) + eps:
        dx = (dy - sum(dy * y, axis, keepdims)) / s

    Softmax sm = softmax(x):
        dx = (dy - sum(dy * sm, -1, keepdims)) * sm

    Sigmoid + eps for pre:
        dx = dy * sig * (1 - sig)

    Sigmoid * mult for post:
        dx = dy * post_out * (1 - post_out / mult)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_SUPPORTED_K = (1, 2, 4, 8, 16)

_TORCH_TO_TL_DTYPE = {
    torch.float64: tl.float64,
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


def _triton_dtype(t: torch.dtype):
    try:
        return _TORCH_TO_TL_DTYPE[t]
    except KeyError as exc:
        raise TypeError(
            f"hc_fused_triton: unsupported dtype {t}; expected one of {list(_TORCH_TO_TL_DTYPE)}"
        ) from exc


# ---------------------------------------------------------------------------
# FWD kernel
# ---------------------------------------------------------------------------


@triton.jit
def _hc_fused_fwd_kernel(
    INPUT_MIXES_PTR,  # [N, (2+K)*K] contiguous, fp32
    SCALE_PTR,        # [3] fp32
    BASE_PTR,         # [(2+K)*K] fp32
    PRE_PTR,          # [N, K] output fp32
    POST_PTR,         # [N, K] output fp32
    COMB_PTR,         # [N, K, K] output fp32 (sinkhorn result)
    STATES_PTR,       # [N, N_STATES, K, K] contiguous trajectory cache
    N,
    K: tl.constexpr,
    REPEAT: tl.constexpr,
    MHC_POST_MULT_VALUE: tl.constexpr,
    MHC_PRE_EPS: tl.constexpr,
    SINKHORN_EPS: tl.constexpr,
    BLOCK_LEADING: tl.constexpr,
):
    """Fused FWD: pre_split_mixes + sinkhorn in one kernel.

    Each program handles BLOCK_LEADING rows of the [N, ...] batch.
    K=4 => all 24 mixes values + 4x4 comb matrix + sinkhorn states in registers.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_LEADING + tl.arange(0, BLOCK_LEADING)
    mask_leading = offs < N

    k_idx = tl.arange(0, K)
    r_offs = tl.arange(0, K)
    c_offs = tl.arange(0, K)
    KK: tl.constexpr = K * K
    KK_TOTAL: tl.constexpr = (2 + K) * K

    # ---- Load scale & base ----
    scale0 = tl.load(SCALE_PTR + 0)
    scale1 = tl.load(SCALE_PTR + 1)
    scale2 = tl.load(SCALE_PTR + 2)

    base_pre = tl.load(BASE_PTR + k_idx)
    base_post = tl.load(BASE_PTR + K + k_idx)
    # base_comb as [K, K] for 3D indexing
    base_comb = tl.load(BASE_PTR + 2 * K + r_offs[:, None] * K + c_offs[None, :])

    # ---- Load input_mixes slices ----
    # pre & post as [BLOCK_LEADING, K]
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
    # comb directly as [BLOCK_LEADING, K, K] — matches sinkhorn kernel's 3D layout
    m = tl.load(
        INPUT_MIXES_PTR
        + offs[:, None, None] * KK_TOTAL
        + 2 * K
        + r_offs[None, :, None] * K + c_offs[None, None, :],
        mask=mask_leading[:, None, None],
        other=0.0,
    )

    # ---- pre_split_mixes: scale + base FMA ----
    pre_logit = pre_logit * scale0 + base_pre[None, :]
    post_logit = post_logit * scale1 + base_post[None, :]
    m = m * scale2 + base_comb[None, :, :]

    # ---- pre & post: sigmoid activations ----
    pre_out = tl.sigmoid(pre_logit) + MHC_PRE_EPS
    post_out = tl.sigmoid(post_logit) * MHC_POST_MULT_VALUE

    # Store pre and post outputs
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

    # ---- Sinkhorn on m [BLOCK_LEADING, K, K] ----
    full_mask = mask_leading[:, None, None]

    # State buffer layout: [N, N_STATES, K, K] row-major
    n_states: tl.constexpr = 2 * REPEAT + 1
    state_row_stride: tl.constexpr = n_states * KK
    state_base = (offs[:, None, None] * state_row_stride
                  + r_offs[None, :, None] * K + c_offs[None, None, :])

    # --- softmax(-1): max subtract, exp, normalize ---
    row_max = tl.max(m, axis=2, keep_dims=True)
    m = tl.exp(m - row_max)
    row_sum = tl.sum(m, axis=2, keep_dims=True)
    sm = m / row_sum
    # state 0: softmax output (no eps)
    tl.store(STATES_PTR + state_base + 0 * KK, sm, mask=full_mask)

    # sm + eps
    m = sm + SINKHORN_EPS
    # state 1: softmax + eps (input to first col normalize)
    tl.store(STATES_PTR + state_base + 1 * KK, m, mask=full_mask)

    # --- first col normalize ---
    s = tl.sum(m, axis=1, keep_dims=True) + SINKHORN_EPS
    m = m / s
    # state 2: after first col normalize
    tl.store(STATES_PTR + state_base + 2 * KK, m, mask=full_mask)

    # --- (REPEAT-1) iterations of (row, col) normalize ---
    for it in tl.static_range(REPEAT - 1):
        # row normalize
        s = tl.sum(m, axis=2, keep_dims=True) + SINKHORN_EPS
        m = m / s
        tl.store(STATES_PTR + state_base + (3 + 2 * it) * KK, m, mask=full_mask)

        # col normalize
        s = tl.sum(m, axis=1, keep_dims=True) + SINKHORN_EPS
        m = m / s
        tl.store(STATES_PTR + state_base + (4 + 2 * it) * KK, m, mask=full_mask)

    # Store final sinkhorn output as [N, K, K]
    comb_base = (offs[:, None, None] * KK
                 + r_offs[None, :, None] * K + c_offs[None, None, :])
    tl.store(COMB_PTR + comb_base, m, mask=full_mask)


# ---------------------------------------------------------------------------
# BWD kernel
# ---------------------------------------------------------------------------


@triton.jit
def _hc_fused_bwd_kernel(
    STATES_PTR,       # [N, N_STATES, K, K] contiguous trajectory cache
    DPRE_PTR,         # [N, K] upstream grad fp32
    DPOST_PTR,        # [N, K] upstream grad fp32
    DCOMB_PTR,        # [N, K, K] upstream grad fp32
    INPUT_MIXES_PTR,  # [N, (2+K)*K] original input fp32
    POST_OUT_PTR,     # [N, K] saved post_layer_mix from FWD fp32
    SCALE_PTR,        # [3] fp32
    BASE_PTR,         # [(2+K)*K] fp32
    DLOGITS_PTR,      # [N, (2+K)*K] output grad for input_mixes fp32
    DBASE_PTR,        # [N, (2+K)*K] per-row partials for base grad
    N,
    K: tl.constexpr,
    REPEAT: tl.constexpr,
    MHC_POST_MULT_VALUE: tl.constexpr,
    SINKHORN_EPS: tl.constexpr,
    BLOCK_LEADING: tl.constexpr,
):
    """Fused BWD: sinkhorn VJP -> pre_split_mixes VJP in one kernel.

    Walks backward through sinkhorn states to get d_comb_logit,
    then applies sigmoid/identity VJPs for pre/post/comb,
    and writes d_input_mixes and d_base partials.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_LEADING + tl.arange(0, BLOCK_LEADING)
    mask_leading = offs < N

    k_idx = tl.arange(0, K)
    r_offs = tl.arange(0, K)
    c_offs = tl.arange(0, K)
    KK: tl.constexpr = K * K
    KK_TOTAL: tl.constexpr = (2 + K) * K
    n_states: tl.constexpr = 2 * REPEAT + 1
    state_row_stride: tl.constexpr = n_states * KK

    full_mask = mask_leading[:, None, None]
    state_base = (offs[:, None, None] * state_row_stride
                  + r_offs[None, :, None] * K + c_offs[None, None, :])
    comb_base = (offs[:, None, None] * KK
                 + r_offs[None, :, None] * K + c_offs[None, None, :])

    # =====================================================================
    # Part 1: Sinkhorn BWD — walk backward through states
    # =====================================================================
    dm = tl.load(DCOMB_PTR + comb_base, mask=full_mask, other=0.0)

    # Walk backward through (REPEAT-1) pairs of (col, row) normalize steps.
    for i in range(REPEAT - 1):
        # col normalize VJP
        col_out_idx = 2 * REPEAT - 2 * i
        col_in_idx = col_out_idx - 1
        m_before = tl.load(STATES_PTR + state_base + col_in_idx * KK,
                           mask=full_mask, other=0.0)
        m_after = tl.load(STATES_PTR + state_base + col_out_idx * KK,
                          mask=full_mask, other=0.0)
        s = tl.sum(m_before, axis=1, keep_dims=True) + SINKHORN_EPS
        dot = tl.sum(dm * m_after, axis=1, keep_dims=True)
        dm = (dm - dot) / s

        # row normalize VJP
        row_out_idx = col_in_idx
        row_in_idx = row_out_idx - 1
        m_before = tl.load(STATES_PTR + state_base + row_in_idx * KK,
                           mask=full_mask, other=0.0)
        m_after = tl.load(STATES_PTR + state_base + row_out_idx * KK,
                          mask=full_mask, other=0.0)
        s = tl.sum(m_before, axis=2, keep_dims=True) + SINKHORN_EPS
        dot = tl.sum(dm * m_after, axis=2, keep_dims=True)
        dm = (dm - dot) / s

    # First col normalize VJP
    m_before = tl.load(STATES_PTR + state_base + 1 * KK, mask=full_mask, other=0.0)
    m_after = tl.load(STATES_PTR + state_base + 2 * KK, mask=full_mask, other=0.0)
    s = tl.sum(m_before, axis=1, keep_dims=True) + SINKHORN_EPS
    dot = tl.sum(dm * m_after, axis=1, keep_dims=True)
    dm = (dm - dot) / s

    # eps addition VJP: pass-through (d/dx (x + eps) = 1)

    # Softmax VJP
    sm = tl.load(STATES_PTR + state_base + 0 * KK, mask=full_mask, other=0.0)
    dot_sm = tl.sum(dm * sm, axis=2, keep_dims=True)
    dm = (dm - dot_sm) * sm

    # dm is now d_comb_logit as [BLOCK_LEADING, K, K]

    # =====================================================================
    # Part 2: pre_split_mixes BWD
    # =====================================================================

    # Load scale
    scale0 = tl.load(SCALE_PTR + 0)
    scale1 = tl.load(SCALE_PTR + 1)
    scale2 = tl.load(SCALE_PTR + 2)

    # Load base_pre for recomputing sigmoid
    base_pre = tl.load(BASE_PTR + k_idx)

    # Load original input_mixes
    input_pre = tl.load(
        INPUT_MIXES_PTR + offs[:, None] * KK_TOTAL + k_idx[None, :],
        mask=mask_leading[:, None],
        other=0.0,
    )

    # Load saved post_out for sigmoid VJP
    post_out = tl.load(
        POST_OUT_PTR + offs[:, None] * K + k_idx[None, :],
        mask=mask_leading[:, None],
        other=0.0,
    )

    # Load upstream grads for pre and post
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

    # Sigmoid VJP for pre: recompute sigmoid from original input
    pre_sig = tl.sigmoid(input_pre * scale0 + base_pre[None, :])
    d_pre_logit = d_pre * pre_sig * (1.0 - pre_sig)

    # Sigmoid VJP for post: use saved post_out
    d_post_logit = d_post * post_out * (1.0 - post_out / MHC_POST_MULT_VALUE)

    # Identity VJP for comb: dm is d_comb_logit from sinkhorn BWD

    # Write d_input_mixes = d_*_logit * scale
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
    # Store comb grad using 3D indexing into the flat [N, (2+K)*K] layout
    comb_store_idx = (offs[:, None, None] * KK_TOTAL
                      + 2 * K
                      + r_offs[None, :, None] * K + c_offs[None, None, :])
    tl.store(
        DLOGITS_PTR + comb_store_idx,
        dm * scale2,
        mask=full_mask,
    )

    # Write d_base per-row partials (= d_*_logit, no scale factor)
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
        DBASE_PTR + comb_store_idx,
        dm,
        mask=full_mask,
    )


# ---------------------------------------------------------------------------
# Block-leading heuristic
# ---------------------------------------------------------------------------


def _pick_block_leading(n: int, k: int) -> int:
    """Pick BLOCK_LEADING based on K and work-axis size (Primus pattern)."""
    if k <= 4:
        cap = 128
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


def hc_fused_fwd_triton(
    input_mixes: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_mult: int,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
    sinkhorn_repeat: int,
    sinkhorn_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the fused FWD kernel (pre_split_mixes + sinkhorn).

    Args:
        input_mixes: [N, (2+K)*K] contiguous fp32.
        mhc_scale: [3] fp32.
        mhc_base: [(2+K)*K] fp32.
        mhc_mult: K.
        mhc_post_mult_value: post sigmoid multiplier.
        mhc_pre_eps: pre sigmoid epsilon.
        sinkhorn_repeat: sinkhorn iteration count.
        sinkhorn_eps: sinkhorn numerical stability epsilon.

    Returns:
        (pre, post, comb, states_buf)
        pre:  [N, K] fp32
        post: [N, K] fp32
        comb: [N, K, K] fp32 (after sinkhorn)
        states_buf: [N, 2*repeat+1, K, K] fp32 trajectory cache for BWD
    """
    K = mhc_mult
    KK_TOTAL = (2 + K) * K

    input_mixes = input_mixes.contiguous().to(torch.float32)
    mhc_scale = mhc_scale.contiguous().to(torch.float32)
    mhc_base = mhc_base.contiguous().to(torch.float32)

    if K not in _SUPPORTED_K:
        raise ValueError(f"hc_fused_triton: unsupported K={K}; expected one of {_SUPPORTED_K}")
    assert input_mixes.ndim == 2 and input_mixes.shape[1] == KK_TOTAL
    if sinkhorn_repeat < 1:
        raise ValueError(f"sinkhorn_repeat must be >= 1, got {sinkhorn_repeat}")

    N = input_mixes.shape[0]
    device = input_mixes.device

    pre = torch.empty((N, K), dtype=torch.float32, device=device)
    post = torch.empty((N, K), dtype=torch.float32, device=device)
    comb = torch.empty((N, K, K), dtype=torch.float32, device=device)

    n_states = 2 * sinkhorn_repeat + 1
    states_buf = torch.empty(
        (N, n_states, K, K),
        dtype=torch.float32,
        device=device,
    )

    block_leading = _pick_block_leading(N, K)
    grid = (triton.cdiv(N, block_leading),)
    _hc_fused_fwd_kernel[grid](
        input_mixes,
        mhc_scale,
        mhc_base,
        pre,
        post,
        comb,
        states_buf,
        N,
        K=K,
        REPEAT=int(sinkhorn_repeat),
        MHC_POST_MULT_VALUE=float(mhc_post_mult_value),
        MHC_PRE_EPS=float(mhc_pre_eps),
        SINKHORN_EPS=float(sinkhorn_eps),
        BLOCK_LEADING=block_leading,
    )

    return pre, post, comb, states_buf


def hc_fused_bwd_triton(
    states_buf: torch.Tensor,
    d_pre: torch.Tensor,
    d_post: torch.Tensor,
    d_comb: torch.Tensor,
    input_mixes: torch.Tensor,
    post_out: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_mult: int,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    sinkhorn_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the fused BWD kernel (sinkhorn VJP + pre_split_mixes VJP).

    Args:
        states_buf: [N, 2*repeat+1, K, K] trajectory cache from FWD.
        d_pre:  [N, K] upstream grad fp32.
        d_post: [N, K] upstream grad fp32.
        d_comb: [N, K, K] upstream grad fp32.
        input_mixes: [N, (2+K)*K] original input fp32.
        post_out: [N, K] saved post_layer_mix from FWD fp32.
        mhc_scale: [3] fp32.
        mhc_base: [(2+K)*K] fp32.
        mhc_mult: K.
        mhc_post_mult_value: post sigmoid multiplier.
        sinkhorn_repeat: sinkhorn iteration count.
        sinkhorn_eps: sinkhorn numerical stability epsilon.

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
    _hc_fused_bwd_kernel[grid](
        states_buf,
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
        K=K,
        REPEAT=int(sinkhorn_repeat),
        MHC_POST_MULT_VALUE=float(mhc_post_mult_value),
        SINKHORN_EPS=float(sinkhorn_eps),
        BLOCK_LEADING=block_leading,
    )

    # Host-side reductions
    d_base = d_base_partials.sum(dim=0)
    d_scale_0 = (input_mixes[:, :K] * d_base_partials[:, :K]).sum()
    d_scale_1 = (input_mixes[:, K:2 * K] * d_base_partials[:, K:2 * K]).sum()
    d_scale_2 = (input_mixes[:, 2 * K:] * d_base_partials[:, 2 * K:]).sum()
    d_scale = torch.stack([d_scale_0, d_scale_1, d_scale_2])

    return d_logits, d_scale, d_base


# ---------------------------------------------------------------------------
# torch.autograd.Function wrapper
# ---------------------------------------------------------------------------


class HCFusedTritonFn(torch.autograd.Function):
    """Autograd wrapper for the fused HC (pre_split_mixes + sinkhorn) kernel."""

    @staticmethod
    def forward(
        ctx,
        input_mixes: torch.Tensor,
        mhc_scale: torch.Tensor,
        mhc_base: torch.Tensor,
        mhc_mult: int,
        mhc_post_mult_value: float,
        mhc_pre_eps: float,
        sinkhorn_repeat: int,
        sinkhorn_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        K = mhc_mult
        mhc_mult3 = K * 2 + K * K

        assert input_mixes.ndim == 3
        tokens_shape = input_mixes.shape[:2]

        input_mixes_flat = input_mixes.view(-1, mhc_mult3).contiguous().to(torch.float32)
        mhc_scale_c = mhc_scale.contiguous().to(torch.float32)
        mhc_base_c = mhc_base.contiguous().to(torch.float32)

        pre, post, comb, states_buf = hc_fused_fwd_triton(
            input_mixes_flat, mhc_scale_c, mhc_base_c,
            K, mhc_post_mult_value, mhc_pre_eps,
            sinkhorn_repeat, sinkhorn_eps,
        )

        ctx.save_for_backward(input_mixes_flat, post, mhc_scale_c, mhc_base_c, states_buf)
        ctx.mhc_mult = K
        ctx.mhc_post_mult_value = mhc_post_mult_value
        ctx.sinkhorn_repeat = sinkhorn_repeat
        ctx.sinkhorn_eps = sinkhorn_eps
        ctx.tokens_shape = tuple(tokens_shape)

        pre = pre.view(*tokens_shape, K, 1)
        post = post.view(*tokens_shape, K, 1)
        comb = comb.view(*tokens_shape, K, K)

        return pre, post, comb

    @staticmethod
    def backward(ctx, d_pre, d_post, d_comb):
        input_mixes_flat, post_out, mhc_scale_c, mhc_base_c, states_buf = ctx.saved_tensors
        K = ctx.mhc_mult
        num_tokens = input_mixes_flat.shape[0]

        d_logits, d_scale, d_base = hc_fused_bwd_triton(
            states_buf,
            d_pre.reshape(num_tokens, K),
            d_post.reshape(num_tokens, K),
            d_comb.reshape(num_tokens, K, K),
            input_mixes_flat,
            post_out,
            mhc_scale_c,
            mhc_base_c,
            K,
            ctx.mhc_post_mult_value,
            ctx.sinkhorn_repeat,
            ctx.sinkhorn_eps,
        )

        d_logits = d_logits.view(*ctx.tokens_shape, d_logits.shape[-1])

        return d_logits, d_scale, d_base, None, None, None, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def hc_fused_triton(
    input_mixes: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    *,
    mhc_mult: int = 4,
    mhc_post_mult_value: float = 1.0,
    mhc_pre_eps: float = 0.01,
    sinkhorn_repeat: int = 10,
    sinkhorn_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused Triton kernel: pre_split_mixes + sinkhorn in one call.

    Args:
        input_mixes: [a, b, (2+K)*K] input tensor.
        mhc_scale: [3] scale parameters.
        mhc_base: [(2+K)*K] base/bias parameters.
        mhc_mult: K (number of heads), default 4.
        mhc_post_mult_value: scalar multiplier for post sigmoid.
        mhc_pre_eps: epsilon added to pre sigmoid output.
        sinkhorn_repeat: number of sinkhorn iterations.
        sinkhorn_eps: sinkhorn numerical stability epsilon.

    Returns:
        (pre, post, comb) where comb has been through sinkhorn.
        pre:  [a, b, K, 1]
        post: [a, b, K, 1]
        comb: [a, b, K, K]
    """
    return HCFusedTritonFn.apply(
        input_mixes,
        mhc_scale,
        mhc_base,
        mhc_mult,
        mhc_post_mult_value,
        mhc_pre_eps,
        sinkhorn_repeat,
        sinkhorn_eps,
    )


__all__ = [
    "hc_fused_fwd_triton",
    "hc_fused_bwd_triton",
    "hc_fused_triton",
    "HCFusedTritonFn",
]
