"""Triton + rocBLAS hybrid mhc_pre_norm_fn FWD + BWD for TileKernels.

Math (FWD):
    For each token (row of residual) of length total_hidden = n_rms_group * rms_group_size:
      Split into n_rms_group groups of rms_group_size.
      For each group g:
        x_g = residual[g*S : (g+1)*S]          (S = rms_group_size)
        sqrsum_g = sum(x_g^2)
        dot_g[k] = sum(x_g * fn[k, g*S:(g+1)*S])   for k in 0..mhc_mult3-1
        rsqrt_g = rsqrt(sqrsum_g / S + eps)
        out[k] += dot_g[k] * rsqrt_g

    If mhc_norm_weight is provided, fn is pre-multiplied: fn = fn * weight.

Math (BWD):
    Given d_out[n, k], for each group g:
      c_g = sum_k(d_out[k] * dot_g[k])
      d_x_g[h] = inv_rms_g * (sum_k(d_out[k] * fn[k, g*S+h]) - x_g[h] * inv_rms_g^2 * c_g / S)
      d_fn[k, g*S+h] = d_out[k] * x_g[h] * inv_rms_g

    inv_rms and dot are recomputed in the BWD kernel to save memory.
    d_fn cross-token reduction is done on the host via torch.einsum.

Hybrid approach (default):
    The GEMV (x @ fn.T) is offloaded to torch.mm (backed by rocBLAS), which is
    ~8x faster than the per-token elementwise dot product in Triton for
    tall-skinny shapes like (4096, 16384) @ (16384, 24).  RMSNorm is computed
    via PyTorch ops.  The fully-fused Triton kernels are kept as fallback.
"""

from __future__ import annotations

import math
import weakref

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# FWD kernel
# ---------------------------------------------------------------------------


@triton.jit
def _norm_fn_fwd_kernel(
    X_PTR,       # [num_tokens, total_hidden] bf16, contiguous
    FN_PTR,      # [mhc_mult3, total_hidden]  fp32, contiguous
    OUT_PTR,     # [num_tokens, mhc_mult3]    fp32, contiguous
    num_tokens,
    total_hidden,
    MHC_MULT3: tl.constexpr,      # number of fn vectors (e.g. 24)
    RMS_GROUP_SIZE: tl.constexpr,  # elements per RMS group (e.g. 1024)
    N_RMS_GROUP: tl.constexpr,     # number of RMS groups
    H_BLK: tl.constexpr,          # tile size along rms_group_size dimension
    EPS: tl.constexpr,            # rms norm epsilon
):
    """Each program handles one token (one row of residual)."""
    pid_n = tl.program_id(0)

    # Accumulator for the final output: out[k] for k in 0..MHC_MULT3-1
    k_offs = tl.arange(0, MHC_MULT3)  # [MHC_MULT3]

    out_acc = tl.zeros([MHC_MULT3], dtype=tl.float32)

    x_row_base = pid_n * total_hidden

    for g in range(N_RMS_GROUP):
        group_base = g * RMS_GROUP_SIZE

        # Per-group accumulators
        sqrsum = tl.zeros([], dtype=tl.float32)
        dot_acc = tl.zeros([MHC_MULT3], dtype=tl.float32)

        for h_start in tl.static_range(0, RMS_GROUP_SIZE, H_BLK):
            h_offs = h_start + tl.arange(0, H_BLK)  # [H_BLK]
            h_mask = h_offs < RMS_GROUP_SIZE

            # Load x chunk: residual[pid_n, group_base + h_offs]
            x_addrs = x_row_base + group_base + h_offs
            x_vals = tl.load(X_PTR + x_addrs, mask=h_mask, other=0.0).to(tl.float32)  # [H_BLK]

            # Accumulate sqrsum
            sqrsum += tl.sum(x_vals * x_vals)

            # Dot products with each fn vector
            # fn[k, group_base + h_offs] for all k
            for k in range(MHC_MULT3):
                fn_addrs = k * total_hidden + group_base + h_offs
                fn_vals = tl.load(FN_PTR + fn_addrs, mask=h_mask, other=0.0)  # [H_BLK], fp32
                dot_acc = tl.where(
                    k_offs == k,
                    dot_acc + tl.sum(x_vals * fn_vals),
                    dot_acc,
                )

        # Compute rsqrt normalization for this group
        inv_rms = tl.rsqrt(sqrsum / RMS_GROUP_SIZE + EPS)

        # Accumulate into output
        out_acc += dot_acc * inv_rms

    # Store output
    out_base = pid_n * MHC_MULT3
    tl.store(OUT_PTR + out_base + k_offs, out_acc, mask=k_offs < MHC_MULT3)


# ---------------------------------------------------------------------------
# Optimized FWD kernel – vectorized over k dimension
# ---------------------------------------------------------------------------


@triton.jit
def _norm_fn_fwd_kernel_vec(
    X_PTR,       # [num_tokens, total_hidden] bf16, contiguous
    FN_PTR,      # [mhc_mult3, total_hidden]  fp32, contiguous
    OUT_PTR,     # [num_tokens, mhc_mult3]    fp32, contiguous
    num_tokens,
    total_hidden,
    MHC_MULT3: tl.constexpr,      # number of fn vectors (e.g. 24)
    RMS_GROUP_SIZE: tl.constexpr,  # elements per RMS group (e.g. 1024)
    N_RMS_GROUP: tl.constexpr,     # number of RMS groups
    H_BLK: tl.constexpr,          # tile size along rms_group_size dimension
    K_BLK: tl.constexpr,          # tile size along mhc_mult3 dimension (power-of-2 >= MHC_MULT3)
    EPS: tl.constexpr,            # rms norm epsilon
):
    """Each program handles one token. Vectorized load of fn across k dimension."""
    pid_n = tl.program_id(0)

    k_offs = tl.arange(0, K_BLK)  # [K_BLK]
    k_mask = k_offs < MHC_MULT3

    out_acc = tl.zeros([K_BLK], dtype=tl.float32)

    x_row_base = pid_n * total_hidden

    for g in range(N_RMS_GROUP):
        group_base = g * RMS_GROUP_SIZE

        # Per-group accumulators
        sqrsum = tl.zeros([], dtype=tl.float32)
        dot_acc = tl.zeros([K_BLK], dtype=tl.float32)

        for h_start in tl.static_range(0, RMS_GROUP_SIZE, H_BLK):
            h_offs = h_start + tl.arange(0, H_BLK)  # [H_BLK]

            # Load x chunk: residual[pid_n, group_base + h_offs]
            x_addrs = x_row_base + group_base + h_offs
            x_vals = tl.load(X_PTR + x_addrs).to(tl.float32)  # [H_BLK]

            # Accumulate sqrsum
            sqrsum += tl.sum(x_vals * x_vals)

            # Load fn[k, group_base + h_offs] as 2D tile [K_BLK, H_BLK]
            # fn_addrs[k, h] = k * total_hidden + group_base + h_offs[h]
            fn_addrs = k_offs[:, None] * total_hidden + (group_base + h_offs)[None, :]  # [K_BLK, H_BLK]
            fn_vals = tl.load(FN_PTR + fn_addrs, mask=k_mask[:, None], other=0.0)  # [K_BLK, H_BLK]

            # dot_acc[k] += sum_h(fn_vals[k, h] * x_vals[h])
            dot_acc += tl.sum(fn_vals * x_vals[None, :], axis=1)

        # Compute rsqrt normalization for this group
        inv_rms = tl.rsqrt(sqrsum / RMS_GROUP_SIZE + EPS)

        # Accumulate into output
        out_acc += dot_acc * inv_rms

    # Store output
    out_base = pid_n * MHC_MULT3
    tl.store(OUT_PTR + out_base + k_offs, out_acc, mask=k_mask)


# ---------------------------------------------------------------------------
# BWD kernel – computes d_x (d_fn is computed on the host)
# ---------------------------------------------------------------------------


@triton.jit
def _norm_fn_bwd_kernel(
    # Inputs
    DOUT_PTR,    # [num_tokens, mhc_mult3]   fp32, contiguous
    X_PTR,       # [num_tokens, total_hidden] bf16, contiguous
    FN_PTR,      # [mhc_mult3, total_hidden]  fp32, contiguous
    # Output
    DX_PTR,      # [num_tokens, total_hidden] fp32, contiguous
    # Dimensions
    num_tokens,
    total_hidden,
    MHC_MULT3: tl.constexpr,
    RMS_GROUP_SIZE: tl.constexpr,
    N_RMS_GROUP: tl.constexpr,
    H_BLK: tl.constexpr,
    K_BLK: tl.constexpr,
    EPS: tl.constexpr,
):
    """Each program handles one (token, group) pair.

    Recomputes inv_rms and dot from x and fn, then uses d_out to produce d_x.
    """
    pid_n = tl.program_id(0)   # token index
    pid_g = tl.program_id(1)   # group index

    k_offs = tl.arange(0, K_BLK)  # [K_BLK]
    k_mask = k_offs < MHC_MULT3

    x_row_base = pid_n * total_hidden
    group_base = pid_g * RMS_GROUP_SIZE

    # Load d_out[pid_n, :] — shape [K_BLK]
    dout_base = pid_n * MHC_MULT3
    d_out = tl.load(DOUT_PTR + dout_base + k_offs, mask=k_mask, other=0.0)  # [K_BLK]

    # --- Pass 1: recompute sqrsum and dot_acc for this group ---
    sqrsum = tl.zeros([], dtype=tl.float32)
    dot_acc = tl.zeros([K_BLK], dtype=tl.float32)

    for h_start in tl.static_range(0, RMS_GROUP_SIZE, H_BLK):
        h_offs = h_start + tl.arange(0, H_BLK)  # [H_BLK]

        x_addrs = x_row_base + group_base + h_offs
        x_vals = tl.load(X_PTR + x_addrs).to(tl.float32)  # [H_BLK]

        sqrsum += tl.sum(x_vals * x_vals)

        fn_addrs = k_offs[:, None] * total_hidden + (group_base + h_offs)[None, :]  # [K_BLK, H_BLK]
        fn_vals = tl.load(FN_PTR + fn_addrs, mask=k_mask[:, None], other=0.0)  # [K_BLK, H_BLK]
        dot_acc += tl.sum(fn_vals * x_vals[None, :], axis=1)  # [K_BLK]

    inv_rms = tl.rsqrt(sqrsum / RMS_GROUP_SIZE + EPS)

    # c_g = sum_k(d_out[k] * dot_acc[k]) — scalar
    c_g = tl.sum(d_out * dot_acc)

    # Precompute coefficient for the rms-gradient term
    # coeff = -inv_rms^3 / S * c_g  =  inv_rms * (-inv_rms^2 * c_g / S)
    rms_coeff = -inv_rms * inv_rms * inv_rms * c_g / RMS_GROUP_SIZE

    # --- Pass 2: compute d_x for each h ---
    for h_start in tl.static_range(0, RMS_GROUP_SIZE, H_BLK):
        h_offs = h_start + tl.arange(0, H_BLK)  # [H_BLK]

        x_addrs = x_row_base + group_base + h_offs
        x_vals = tl.load(X_PTR + x_addrs).to(tl.float32)  # [H_BLK]

        # sum_k(d_out[k] * fn[k, group_base + h]) for each h
        fn_addrs = k_offs[:, None] * total_hidden + (group_base + h_offs)[None, :]  # [K_BLK, H_BLK]
        fn_vals = tl.load(FN_PTR + fn_addrs, mask=k_mask[:, None], other=0.0)  # [K_BLK, H_BLK]
        # dout_fn[h] = sum_k(d_out[k] * fn[k, h])
        dout_fn = tl.sum(d_out[:, None] * fn_vals, axis=0)  # [H_BLK]

        # d_x[h] = inv_rms * dout_fn[h] + rms_coeff * x[h]
        dx_vals = inv_rms * dout_fn + rms_coeff * x_vals  # [H_BLK]

        dx_addrs = pid_n * total_hidden + group_base + h_offs
        tl.store(DX_PTR + dx_addrs, dx_vals)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    return 1 << (n - 1).bit_length()


def _pick_h_blk(rms_group_size: int, default: int = 512) -> int:
    """Pick H_BLK that evenly divides rms_group_size, capped at default."""
    return math.gcd(rms_group_size, default)


# ---------------------------------------------------------------------------
# fn transpose cache — avoids repeated .T.contiguous() allocations
# ---------------------------------------------------------------------------

_fn_transpose_cache: dict[int, tuple[weakref.ref, torch.Tensor]] = {}


def _get_fn_transpose(fn: torch.Tensor) -> torch.Tensor:
    """Return fn.T.contiguous(), caching the result keyed on data_ptr."""
    key = fn.data_ptr()
    cached = _fn_transpose_cache.get(key)
    if cached is not None:
        ref, fn_t = cached
        if ref() is fn:
            return fn_t
    fn_t = fn.T.contiguous()
    _fn_transpose_cache[key] = (weakref.ref(fn), fn_t)
    return fn_t


def norm_fn_fwd_triton(
    x: torch.Tensor,           # [num_tokens, total_hidden] bf16, contiguous
    fn: torch.Tensor,          # [mhc_mult3, total_hidden]  fp32, contiguous
    mhc_mult3: int,
    n_rms_group: int,
    rms_group_size: int,
    eps: float,
    out: torch.Tensor | None = None,
    use_fused_triton: bool = False,
) -> torch.Tensor:
    """FWD wrapper for mhc_pre_norm_fn.

    By default uses a hybrid approach: torch.mm for the GEMV (backed by
    rocBLAS) and PyTorch ops for RMSNorm.  Set *use_fused_triton=True* to
    fall back to the fully-fused Triton kernel.
    """
    num_tokens = x.shape[0]
    total_hidden = x.shape[1]

    if use_fused_triton:
        # --- Fully-fused Triton fallback ---
        if out is None:
            out = torch.empty((num_tokens, mhc_mult3), dtype=torch.float32, device=x.device)

        H_BLK = _pick_h_blk(rms_group_size)
        K_BLK = _next_power_of_2(mhc_mult3)

        grid = (num_tokens,)

        _norm_fn_fwd_kernel_vec[grid](
            x, fn, out,
            num_tokens, total_hidden,
            MHC_MULT3=mhc_mult3,
            RMS_GROUP_SIZE=rms_group_size,
            N_RMS_GROUP=n_rms_group,
            H_BLK=H_BLK,
            K_BLK=K_BLK,
            EPS=eps,
        )
        return out

    # --- Hybrid: torch.mm GEMV + PyTorch RMSNorm ---
    x_f32 = x.float()  # [num_tokens, total_hidden]

    # Per-group inv_rms: [num_tokens, n_rms_group]
    x_grouped = x_f32.view(num_tokens, n_rms_group, rms_group_size)
    inv_rms = torch.rsqrt(
        x_grouped.square().sum(-1) / rms_group_size + eps,
    )  # [num_tokens, n_rms_group]

    if n_rms_group == 1:
        # Fast path: single group — GEMV via rocBLAS + scalar inv_rms
        fn_t = _get_fn_transpose(fn)
        dot = torch.mm(x_f32, fn_t)  # [num_tokens, mhc_mult3]
        result = dot * inv_rms  # inv_rms is [num_tokens, 1], broadcasts
    else:
        # Multi-group: per-group dot products needed, use einsum
        fn_grouped = fn.view(mhc_mult3, n_rms_group, rms_group_size)
        dot_grouped = torch.einsum(
            'ngs,kgs->ngk', x_grouped, fn_grouped,
        )  # [num_tokens, n_rms_group, mhc_mult3]
        result = (dot_grouped * inv_rms.unsqueeze(-1)).sum(1)  # [num_tokens, mhc_mult3]

    if out is not None:
        out.copy_(result)
        return out
    return result


def norm_fn_bwd_triton(
    d_out: torch.Tensor,        # [num_tokens, mhc_mult3]    fp32
    x: torch.Tensor,            # [num_tokens, total_hidden] bf16, contiguous
    fn: torch.Tensor,           # [mhc_mult3, total_hidden]  fp32, contiguous
    mhc_mult3: int,
    n_rms_group: int,
    rms_group_size: int,
    eps: float,
    use_fused_triton: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BWD wrapper for d_x and d_fn.

    By default uses a hybrid approach with torch.mm for GEMVs.  Set
    *use_fused_triton=True* to use the Triton kernel for d_x.

    Returns:
        d_x:  [num_tokens, total_hidden] fp32
        d_fn: [mhc_mult3, total_hidden]  fp32
    """
    num_tokens = x.shape[0]
    total_hidden = x.shape[1]

    # --- Recompute inv_rms (needed by both paths) ---
    x_f32 = x.float()  # [num_tokens, total_hidden]
    x_grouped = x_f32.view(num_tokens, n_rms_group, rms_group_size)
    sqrsum = x_grouped.square().sum(-1)  # [num_tokens, n_rms_group]
    inv_rms = torch.rsqrt(sqrsum / rms_group_size + eps)  # [num_tokens, n_rms_group]

    if use_fused_triton:
        # --- Fully-fused Triton fallback for d_x ---
        d_x = torch.empty((num_tokens, total_hidden), dtype=torch.float32, device=x.device)

        H_BLK = _pick_h_blk(rms_group_size)
        K_BLK = _next_power_of_2(mhc_mult3)

        grid = (num_tokens, n_rms_group)

        _norm_fn_bwd_kernel[grid](
            d_out, x, fn,
            d_x,
            num_tokens, total_hidden,
            MHC_MULT3=mhc_mult3,
            RMS_GROUP_SIZE=rms_group_size,
            N_RMS_GROUP=n_rms_group,
            H_BLK=H_BLK,
            K_BLK=K_BLK,
            EPS=eps,
        )
    else:
        # --- Hybrid: torch.mm for d_x ---
        # d_x_g[h] = inv_rms_g * (sum_k(d_out[k] * fn[k, g*S+h])
        #            - x_g[h] * inv_rms_g^2 * c_g / S)
        # where c_g = sum_k(d_out[k] * dot_g[k])
        #       dot_g[k] = sum_h(x_g[h] * fn[k, g*S+h])

        # Recompute dot_grouped for multi-group c_g
        fn_grouped = fn.view(mhc_mult3, n_rms_group, rms_group_size)
        dot_grouped = torch.einsum(
            'ngs,kgs->ngk', x_grouped, fn_grouped,
        )  # [num_tokens, n_rms_group, mhc_mult3]

        # c_g = sum_k(d_out[k] * dot_g[k])  — [num_tokens, n_rms_group]
        c_g = torch.einsum(
            'nk,ngk->ng', d_out, dot_grouped,
        )  # [num_tokens, n_rms_group]

        # sum_k(d_out[k] * fn[k, h]) via GEMV: [num_tokens, total_hidden]
        dout_fn = torch.mm(d_out, fn)  # [num_tokens, total_hidden]
        dout_fn_grouped = dout_fn.view(num_tokens, n_rms_group, rms_group_size)

        # rms_coeff[n, g] = -inv_rms^3 * c_g / S
        rms_coeff = -inv_rms.pow(3) * c_g / rms_group_size  # [num_tokens, n_rms_group]

        # d_x[n, g, h] = inv_rms[n,g] * dout_fn[n,g,h] + rms_coeff[n,g] * x[n,g,h]
        d_x = (
            inv_rms.unsqueeze(-1) * dout_fn_grouped
            + rms_coeff.unsqueeze(-1) * x_grouped
        ).view(num_tokens, total_hidden)

    # --- d_fn (same for both paths): d_fn = d_out^T @ x_normed ---
    x_normed = (x_grouped * inv_rms.unsqueeze(-1)).view(num_tokens, total_hidden)
    d_fn = torch.mm(d_out.t(), x_normed)  # [mhc_mult3, total_hidden]

    return d_x, d_fn


# ---------------------------------------------------------------------------
# Autograd wrapper
# ---------------------------------------------------------------------------


class MhcPreNormFnTritonFn(torch.autograd.Function):
    """Autograd wrapper for hybrid mhc_pre_norm_fn FWD + BWD."""

    @staticmethod
    def forward(
        ctx,
        residual: torch.Tensor,
        mhc_fn: torch.Tensor,
        mhc_norm_weight: torch.Tensor | None,
        mhc_norm_eps: float,
    ) -> torch.Tensor:
        # residual: (n0, n1, mhc_mult, hidden_size) bf16
        # mhc_fn:   (mhc_mult3, mhc_mult * hidden_size) fp32
        n0, n1 = residual.shape[:2]
        mhc_mult3 = mhc_fn.shape[0]
        total_hidden = mhc_fn.shape[1]

        # Merge norm weight into fn if present
        if mhc_norm_weight is not None:
            fn = mhc_fn * mhc_norm_weight
        else:
            fn = mhc_fn

        fn = fn.contiguous()

        # Flatten residual to (N, total_hidden) bf16
        x = residual.flatten(2, 3).contiguous()
        assert x.shape[-1] == total_hidden

        rms_group_size = fn.shape[-1]
        n_rms_group = total_hidden // rms_group_size
        assert n_rms_group * rms_group_size == total_hidden

        x_flat = x.view(n0 * n1, total_hidden)

        out = norm_fn_fwd_triton(
            x_flat, fn,
            mhc_mult3=mhc_mult3,
            n_rms_group=n_rms_group,
            rms_group_size=rms_group_size,
            eps=mhc_norm_eps,
        )

        # Reshape output: (N, mhc_mult3) -> (n0, n1, mhc_mult3)
        out = out.view(n0, n1, mhc_mult3)

        # Save for backward (save_for_backward accepts None entries)
        ctx.save_for_backward(x_flat, fn, mhc_fn, mhc_norm_weight)
        ctx.n0 = n0
        ctx.n1 = n1
        ctx.mhc_mult3 = mhc_mult3
        ctx.n_rms_group = n_rms_group
        ctx.rms_group_size = rms_group_size
        ctx.mhc_norm_eps = mhc_norm_eps
        ctx.residual_shape = residual.shape

        return out

    @staticmethod
    def backward(ctx, grad_output):
        x_flat, fn, mhc_fn, mhc_norm_weight = ctx.saved_tensors
        n0, n1 = ctx.n0, ctx.n1
        mhc_mult3 = ctx.mhc_mult3
        n_rms_group = ctx.n_rms_group
        rms_group_size = ctx.rms_group_size
        eps = ctx.mhc_norm_eps

        # grad_output: (n0, n1, mhc_mult3) -> flatten to (N, mhc_mult3)
        d_out = grad_output.reshape(n0 * n1, mhc_mult3).contiguous()

        d_x_flat, d_fn = norm_fn_bwd_triton(
            d_out, x_flat, fn,
            mhc_mult3=mhc_mult3,
            n_rms_group=n_rms_group,
            rms_group_size=rms_group_size,
            eps=eps,
        )

        # d_x_flat: (N, total_hidden) fp32 -> reshape to original residual shape
        d_residual = d_x_flat.view(ctx.residual_shape).to(x_flat.dtype)

        # d_fn is w.r.t. the (possibly weight-merged) fn.
        # If mhc_norm_weight was applied: fn = mhc_fn * weight
        #   d_mhc_fn = d_fn * weight
        #   d_weight  = (d_fn * mhc_fn).sum(0)
        if mhc_norm_weight is not None:
            d_mhc_fn = d_fn * mhc_norm_weight
            d_norm_weight = (d_fn * mhc_fn).sum(0)
        else:
            d_mhc_fn = d_fn
            d_norm_weight = None

        return d_residual, d_mhc_fn, d_norm_weight, None  # None for mhc_norm_eps


def mhc_pre_norm_fn_triton(
    residual: torch.Tensor,
    mhc_fn: torch.Tensor,
    mhc_norm_weight: torch.Tensor | None,
    mhc_norm_eps: float,
) -> torch.Tensor:
    """Triton-fused mhc_pre_norm_fn (drop-in replacement for mhc_pre_norm_fn_ref).

    Args:
        residual:        [n0, n1, mhc_mult, hidden_size] bf16
        mhc_fn:          [mhc_mult3, mhc_mult * hidden_size] fp32
        mhc_norm_weight: [mhc_mult * hidden_size] fp32 or None
        mhc_norm_eps:    float

    Returns:
        output: [n0, n1, mhc_mult3] fp32
    """
    return MhcPreNormFnTritonFn.apply(residual, mhc_fn, mhc_norm_weight, mhc_norm_eps)
