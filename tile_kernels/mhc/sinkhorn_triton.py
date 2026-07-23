"""Triton-fused Sinkhorn-Knopp FWD/BWD for TileKernels.

Math (TileKernels version):
    x = x.softmax(-1) + eps                    # softmax then add eps
    x = x / (x.sum(-2, keepdim=True) + eps)    # col normalize
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)  # row normalize
        x = x / (x.sum(-2, keepdim=True) + eps)  # col normalize
    return x

State trajectory saved by FWD for BWD (total = 2 + 2 * REPEAT states):
    state 0: softmax output (BEFORE adding eps) -- needed for softmax VJP
    state 1: softmax output + eps  -- input to first col-normalize
    state 2: after first col-normalize
    state 3: input to first row-normalize (same as state 2)
    ...pattern: pairs of (input, output) for each normalize step

Actually, we simplify to the Primus pattern but with 2*REPEAT+1 states:
    state 0: sm = softmax(x)           (softmax output, no eps)
    state 1: sm + eps                  (input to first col normalize)
    state 2: after first col normalize
    for step i in 0..REPEAT-2:
        state 3+2*i: after row normalize  (these equal the x before next col)
        state 4+2*i: after col normalize
    Total states: 2 * REPEAT + 1

Per-step VJP for normalization:
    forward:  y = x / s, where s = sum(x, axis=axis) + eps
    backward: dx = (dy - sum(dy * y, axis=axis, keepdims=True)) / s

Softmax VJP:
    forward:  sm = softmax(x)
    backward: dx = (dy - sum(dy * sm, axis=-1, keepdims=True)) * sm
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
            f"sinkhorn_triton: unsupported dtype {t}; expected one of {list(_TORCH_TO_TL_DTYPE)}"
        ) from exc


# ---------------------------------------------------------------------------
# FWD kernel
# ---------------------------------------------------------------------------


@triton.jit
def _sinkhorn_fwd_kernel(
    X_PTR,       # [N, K, K] contiguous input
    Y_PTR,       # [N, K, K] contiguous output
    STATES_PTR,  # [N, N_STATES, K, K] contiguous trajectory cache
    N,
    K: tl.constexpr,
    REPEAT: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_LEADING: tl.constexpr,
    DTYPE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    """FWD: softmax+eps → col_normalize → (row_normalize → col_normalize) × (REPEAT-1).

    Saves 2*REPEAT+1 states to STATES_PTR for backward.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_LEADING + tl.arange(0, BLOCK_LEADING)
    mask_leading = offs < N

    r_offs = tl.arange(0, K)
    c_offs = tl.arange(0, K)

    KK: tl.constexpr = K * K
    base = offs[:, None, None] * KK + r_offs[None, :, None] * K + c_offs[None, None, :]
    full_mask = mask_leading[:, None, None]

    # State buffer layout: [N, N_STATES, K, K] row-major
    n_states: tl.constexpr = 2 * REPEAT + 1
    state_row_stride: tl.constexpr = n_states * KK
    state_base = offs[:, None, None] * state_row_stride + r_offs[None, :, None] * K + c_offs[None, None, :]

    # Load input
    m = tl.load(X_PTR + base, mask=full_mask, other=0.0).to(COMPUTE_DTYPE)

    # --- softmax(-1): max subtract, exp, normalize ---
    row_max = tl.max(m, axis=2, keep_dims=True)
    m = tl.exp(m - row_max)
    row_sum = tl.sum(m, axis=2, keep_dims=True)
    sm = m / row_sum
    # state 0: softmax output (no eps) -- needed for softmax VJP
    tl.store(STATES_PTR + state_base + 0 * KK, sm, mask=full_mask)

    # sm + eps
    m = sm + EPS
    # state 1: softmax + eps (input to first col normalize)
    tl.store(STATES_PTR + state_base + 1 * KK, m, mask=full_mask)

    # --- first col normalize: x / (x.sum(-2, keepdim=True) + eps) ---
    s = tl.sum(m, axis=1, keep_dims=True) + EPS
    m = m / s
    # state 2: after first col normalize
    tl.store(STATES_PTR + state_base + 2 * KK, m, mask=full_mask)

    # --- (REPEAT-1) iterations of (row, col) normalize ---
    for it in tl.static_range(REPEAT - 1):
        # row normalize: x / (x.sum(-1, keepdim=True) + eps)
        s = tl.sum(m, axis=2, keep_dims=True) + EPS
        m = m / s
        tl.store(STATES_PTR + state_base + (3 + 2 * it) * KK, m, mask=full_mask)

        # col normalize: x / (x.sum(-2, keepdim=True) + eps)
        s = tl.sum(m, axis=1, keep_dims=True) + EPS
        m = m / s
        tl.store(STATES_PTR + state_base + (4 + 2 * it) * KK, m, mask=full_mask)

    # Store final output
    tl.store(Y_PTR + base, m.to(DTYPE), mask=full_mask)


# ---------------------------------------------------------------------------
# BWD kernel
# ---------------------------------------------------------------------------


@triton.jit
def _sinkhorn_bwd_kernel(
    STATES_PTR,  # [N, N_STATES, K, K] contiguous trajectory cache
    DY_PTR,      # [N, K, K] contiguous upstream grad
    DX_PTR,      # [N, K, K] contiguous output grad
    N,
    K: tl.constexpr,
    REPEAT: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_LEADING: tl.constexpr,
    DTYPE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    """BWD: walk the FWD trajectory backward, applying per-step VJP.

    State layout (2*REPEAT+1 states):
        0: sm (softmax output, no eps)
        1: sm + eps (input to first col normalize)
        2: after first col normalize
        3: after first row normalize (iteration 0)
        4: after first col normalize (iteration 0)
        ...
        2*REPEAT: final output

    Normalization VJP (for y = x / s where s = sum(x, axis) + eps):
        dx = (dy - sum(dy * y, axis, keepdims)) / s

    Softmax VJP (for sm = softmax(x)):
        dx = (dy - sum(dy * sm, axis=-1, keepdims)) * sm
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_LEADING + tl.arange(0, BLOCK_LEADING)
    mask_leading = offs < N

    r_offs = tl.arange(0, K)
    c_offs = tl.arange(0, K)

    KK: tl.constexpr = K * K
    n_states: tl.constexpr = 2 * REPEAT + 1
    state_row_stride: tl.constexpr = n_states * KK

    base = offs[:, None, None] * KK + r_offs[None, :, None] * K + c_offs[None, None, :]
    full_mask = mask_leading[:, None, None]
    state_base = offs[:, None, None] * state_row_stride + r_offs[None, :, None] * K + c_offs[None, None, :]

    dm = tl.load(DY_PTR + base, mask=full_mask, other=0.0).to(COMPUTE_DTYPE)

    # Walk backward through (REPEAT-1) pairs of (col, row) normalize steps.
    # The last state index is 2*REPEAT. We go backward.
    for i in range(REPEAT - 1):
        # --- col normalize VJP (the later step in fwd) ---
        # fwd step wrote state at index: 2*REPEAT - 2*i
        # input to this col step was state at: 2*REPEAT - 2*i - 1
        # output of this col step was state at: 2*REPEAT - 2*i
        col_out_idx = 2 * REPEAT - 2 * i
        col_in_idx = col_out_idx - 1
        m_before = tl.load(STATES_PTR + state_base + col_in_idx * KK, mask=full_mask, other=0.0)
        m_after = tl.load(STATES_PTR + state_base + col_out_idx * KK, mask=full_mask, other=0.0)
        s = tl.sum(m_before, axis=1, keep_dims=True) + EPS
        dot = tl.sum(dm * m_after, axis=1, keep_dims=True)
        dm = (dm - dot) / s

        # --- row normalize VJP (the earlier step in fwd) ---
        # fwd step wrote state at index: col_in_idx = 2*REPEAT - 2*i - 1
        # input to this row step was state at: col_in_idx - 1
        # output of this row step was state at: col_in_idx
        row_out_idx = col_in_idx
        row_in_idx = row_out_idx - 1
        m_before = tl.load(STATES_PTR + state_base + row_in_idx * KK, mask=full_mask, other=0.0)
        m_after = tl.load(STATES_PTR + state_base + row_out_idx * KK, mask=full_mask, other=0.0)
        s = tl.sum(m_before, axis=2, keep_dims=True) + EPS
        dot = tl.sum(dm * m_after, axis=2, keep_dims=True)
        dm = (dm - dot) / s

    # --- first col normalize VJP ---
    # fwd: state 1 (sm+eps) -> state 2 (after col normalize)
    m_before = tl.load(STATES_PTR + state_base + 1 * KK, mask=full_mask, other=0.0)
    m_after = tl.load(STATES_PTR + state_base + 2 * KK, mask=full_mask, other=0.0)
    s = tl.sum(m_before, axis=1, keep_dims=True) + EPS
    dot = tl.sum(dm * m_after, axis=1, keep_dims=True)
    dm = (dm - dot) / s

    # --- eps addition VJP: trivially pass-through (d/dx (x + eps) = 1) ---
    # dm is already correct after col normalize VJP

    # --- softmax VJP ---
    # sm is state 0
    sm = tl.load(STATES_PTR + state_base + 0 * KK, mask=full_mask, other=0.0)
    dot_sm = tl.sum(dm * sm, axis=2, keep_dims=True)
    dm = (dm - dot_sm) * sm

    tl.store(DX_PTR + base, dm.to(DTYPE), mask=full_mask)


# ---------------------------------------------------------------------------
# Block-leading heuristic
# ---------------------------------------------------------------------------


def _pick_block_leading(n: int, k: int) -> int:
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


def sinkhorn_fwd_triton(
    x: torch.Tensor,
    repeat: int = 10,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the Triton FWD kernel.

    Args:
        x: [..., K, K] input tensor.
        repeat: number of Sinkhorn iterations.
        eps: numerical stability epsilon.

    Returns:
        (output, states_buf) where states_buf is needed for backward.
    """
    if x.dim() < 2:
        raise ValueError(f"sinkhorn_fwd_triton: input must be >= 2-D, got shape {tuple(x.shape)}")
    K = x.shape[-1]
    if x.shape[-2] != K:
        raise ValueError(f"sinkhorn_fwd_triton: last two dims must be square, got {tuple(x.shape)}")
    if K not in _SUPPORTED_K:
        raise ValueError(f"sinkhorn_fwd_triton: unsupported K={K}; expected one of {_SUPPORTED_K}")
    if repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")

    x = x.contiguous()
    leading_shape = x.shape[:-2]
    N = 1
    for s in leading_shape:
        N *= s

    out = torch.empty_like(x)
    n_states = 2 * repeat + 1
    states_buf = torch.empty(
        (N, n_states, K, K),
        dtype=torch.float32,
        device=x.device,
    )

    block_leading = _pick_block_leading(N, K)
    grid = (triton.cdiv(N, block_leading),)
    _sinkhorn_fwd_kernel[grid](
        x.view(N, K, K),
        out.view(N, K, K),
        states_buf,
        N,
        K=K,
        REPEAT=int(repeat),
        EPS=float(eps),
        BLOCK_LEADING=block_leading,
        DTYPE=_triton_dtype(x.dtype),
        COMPUTE_DTYPE=tl.float32,
    )

    return out, states_buf


def sinkhorn_bwd_triton(
    states_buf: torch.Tensor,
    dy: torch.Tensor,
    K: int,
    repeat: int = 10,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Run the Triton BWD kernel.

    Args:
        states_buf: [N, 2*repeat+1, K, K] trajectory cache from FWD.
        dy: [..., K, K] upstream gradient.
        K: spatial dimension size.
        repeat: number of Sinkhorn iterations (must match FWD).
        eps: numerical stability epsilon.

    Returns:
        dx: gradient w.r.t. input, same shape as dy.
    """
    dy = dy.contiguous()
    leading_shape = dy.shape[:-2]
    N = 1
    for s in leading_shape:
        N *= s

    dx = torch.empty_like(dy)
    block_leading = _pick_block_leading(N, K)
    grid = (triton.cdiv(N, block_leading),)
    _sinkhorn_bwd_kernel[grid](
        states_buf,
        dy.view(N, K, K),
        dx.view(N, K, K),
        N,
        K=K,
        REPEAT=int(repeat),
        EPS=float(eps),
        BLOCK_LEADING=block_leading,
        DTYPE=_triton_dtype(dy.dtype),
        COMPUTE_DTYPE=tl.float32,
    )

    return dx


# ---------------------------------------------------------------------------
# torch.autograd.Function wrapper
# ---------------------------------------------------------------------------


class SinkhornNormalizeTritonFn(torch.autograd.Function):
    """Autograd wrapper for the Triton Sinkhorn FWD/BWD kernels."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        repeat: int,
        eps: float,
    ) -> torch.Tensor:
        out, states_buf = sinkhorn_fwd_triton(x, repeat, eps)
        ctx.save_for_backward(states_buf)
        ctx.repeat = repeat
        ctx.eps = eps
        ctx.K = x.shape[-1]
        ctx.leading_shape = tuple(x.shape[:-2])
        return out

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        (states_buf,) = ctx.saved_tensors
        dx = sinkhorn_bwd_triton(states_buf, dy, ctx.K, ctx.repeat, ctx.eps)
        return dx, None, None


def sinkhorn_normalize_triton(
    x: torch.Tensor,
    repeat: int = 10,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Triton-fused Sinkhorn normalize (TileKernels math).

    Args:
        x: [..., K, K] input tensor (K must be power-of-2 in {1,2,4,8,16}).
        repeat: number of Sinkhorn iterations.
        eps: numerical stability epsilon.

    Returns:
        Doubly-stochastic-ish matrix, same shape as x.
    """
    return SinkhornNormalizeTritonFn.apply(
        x.contiguous().view(-1, *x.shape[-2:]),
        repeat,
        eps,
    ).view_as(x)


__all__ = [
    "sinkhorn_fwd_triton",
    "sinkhorn_bwd_triton",
    "sinkhorn_normalize_triton",
    "SinkhornNormalizeTritonFn",
]
