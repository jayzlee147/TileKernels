"""Triton-fused mhc_pre_norm_fn FWD for TileKernels.

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

This is a fused RMSNorm + small GEMM, replacing the tilelang multi-kernel
approach (normw_merge, fwd_mul, fwd_norm) with a single Triton kernel.

BWD: TODO
"""

from __future__ import annotations

import math

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
# Python wrappers
# ---------------------------------------------------------------------------


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    return 1 << (n - 1).bit_length()


def _pick_h_blk(rms_group_size: int, default: int = 512) -> int:
    """Pick H_BLK that evenly divides rms_group_size, capped at default."""
    return math.gcd(rms_group_size, default)


def norm_fn_fwd_triton(
    x: torch.Tensor,           # [num_tokens, total_hidden] bf16, contiguous
    fn: torch.Tensor,          # [mhc_mult3, total_hidden]  fp32, contiguous
    mhc_mult3: int,
    n_rms_group: int,
    rms_group_size: int,
    eps: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch the FWD Triton kernel for mhc_pre_norm_fn."""
    num_tokens = x.shape[0]
    total_hidden = x.shape[1]

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


# ---------------------------------------------------------------------------
# Autograd wrapper (FWD only; BWD = TODO)
# ---------------------------------------------------------------------------


class MhcPreNormFnTritonFn(torch.autograd.Function):
    """Autograd wrapper for Triton mhc_pre_norm_fn FWD."""

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

        # Determine n_rms_group and rms_group_size from shapes
        # mhc_fn is (mhc_mult3, n_rms_group * rms_group_size)
        # The original ref reshapes residual to (-1, 1, rms_group_size)
        # and fn to (mhc_mult, 1, rms_group_size).
        # Here mhc_mult = mhc_fn.shape[0] = mhc_mult3 and rms_group_size = mhc_fn.shape[-1]
        # Wait, re-read the ref:
        #   mhc_mult = mhc_fn.shape[0]   => this is mhc_mult3 in the tilelang version
        #   rms_group_size = mhc_fn.shape[-1]  => fn shape is (mhc_mult3, rms_group_size)??
        # Actually no. In the ref:
        #   mhc_fn shape is (mhc_mult3, n_rms_group, rms_group_size) originally,
        #   then .flatten(1,2) -> (mhc_mult3, total_hidden)
        #   In the ref code: mhc_mult = mhc_fn.shape[0], rms_group_size = mhc_fn.shape[-1]
        #   This means mhc_fn has at least 2 dims. Looking at mhc_fn.shape[-1] => rms_group_size.
        #   But mhc_fn is already (mhc_mult3, total_hidden) after flatten...
        #
        # Let me re-read the ref:
        #   mhc_mult = mhc_fn.shape[0]      # = mhc_mult3
        #   rms_group_size = mhc_fn.shape[-1]  # This works if fn is (mhc_mult3, rms_group_size)
        #                                       # but fn is (mhc_mult3, n_rms_group*rms_group_size)
        #
        # The ref passes fn as (mhc_mult3, n_rms_group, rms_group_size) NOT flattened.
        # The test flatten it: fn.flatten(1,2). But the ref expects non-flattened fn.
        #
        # So actually: in the ref, mhc_fn.shape = (mhc_mult3, n_rms_group, rms_group_size)
        # but in the test, fn is created as (mhc_mult3, mhc_mult, hidden_size).flatten(1,2)
        # = (mhc_mult3, mhc_mult*hidden_size) = (mhc_mult3, total_hidden)
        # Then in the ref: rms_group_size = mhc_fn.shape[-1] = total_hidden??
        #
        # That can't be right. Let me re-read more carefully.
        # In test_norm_fn.py, fn is passed directly to mhc_pre_norm_fn_ref as
        # (mhc_mult3, total_hidden) where total_hidden = mhc_mult * hidden_size.
        # In the ref: rms_group_size = mhc_fn.shape[-1] = total_hidden.
        # Then residual.view(-1, 1, rms_group_size) => each row IS one group.
        # And n_rms_group = total_hidden / rms_group_size = 1??
        #
        # Wait, but the tilelang kernel has n_rms_group as a parameter and it's 4.
        # Let me look at how the tilelang ops call it.

        # We need to figure out the actual intended n_rms_group.
        # The safest approach: accept it as a parameter from the caller.
        # For now, infer from shapes. If mhc_fn is 2D (mhc_mult3, total_hidden),
        # then rms_group_size = total_hidden and n_rms_group = 1 (as the ref does).
        # If mhc_fn is 3D (mhc_mult3, n_rms_group, rms_group_size), use those dims.

        # Based on the ref code: rms_group_size = mhc_fn.shape[-1], and the fn
        # is used as-is. So if fn is 2D, rms_group_size = fn.shape[-1] = total_hidden,
        # and there's only 1 group. If fn is 3D, rms_group_size = fn.shape[-1].
        # For our Triton wrapper, we use the ORIGINAL fn shape before flatten.
        # But we receive it already flattened... So we need rms_group_size as a param.

        # Actually, re-reading the ref more carefully:
        #   residual is (n0, n1, mhc*hidden) after flatten(2,3).float()
        #   mhc_mult = mhc_fn.shape[0]
        #   rms_group_size = mhc_fn.shape[-1]
        #   residual.view(-1, 1, rms_group_size) => (n0*n1 * n_rms_group, 1, rms_group_size)
        #     where n_rms_group = total_hidden / rms_group_size
        #   mhc_fn.view(mhc_mult, 1, rms_group_size) => fn must be (mhc_mult, total_hidden)
        #     and view as (mhc_mult, 1, rms_group_size) => this only works if total_hidden == rms_group_size
        #
        # Hmm, that means fn.view(mhc_mult, 1, rms_group_size) requires total_hidden = rms_group_size.
        # So in the ref, fn.shape[-1] IS rms_group_size, and the view(-1,1,rms_group_size)
        # splits residual into chunks, each of size rms_group_size.
        # n_rms_group = total_hidden / rms_group_size.
        #
        # Wait, fn.view(mhc_mult, 1, rms_group_size) — this doesn't change the data,
        # just adds a middle dim of 1. The fn shape is (mhc_mult, rms_group_size) before this.
        # But fn was created as (mhc_mult3, mhc_mult * hidden_size) in the test.
        # So rms_group_size = mhc_mult * hidden_size = total_hidden.
        #
        # Then residual.view(-1, 1, rms_group_size) = (n0*n1, 1, total_hidden).
        # That means there IS only 1 group in the ref.
        #
        # But the tilelang kernel has n_rms_group = 4. So the tilelang version
        # must receive fn in a different shape. Let me check the modeling/mhc/ops.py.

        # For this Triton implementation, let's follow the REF exactly:
        # rms_group_size = fn.shape[-1] (= total_hidden if fn is 2D)
        # n_rms_group = total_hidden // rms_group_size

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

        # TODO: save tensors for backward
        # ctx.save_for_backward(...)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: implement backward pass
        raise NotImplementedError('norm_fn_triton backward not yet implemented')


def mhc_pre_norm_fn_triton(
    residual: torch.Tensor,
    mhc_fn: torch.Tensor,
    mhc_norm_weight: torch.Tensor | None,
    mhc_norm_eps: float,
) -> torch.Tensor:
    """Triton-fused mhc_pre_norm_fn (drop-in replacement for mhc_pre_norm_fn_ref, FWD only).

    Args:
        residual:        [n0, n1, mhc_mult, hidden_size] bf16
        mhc_fn:          [mhc_mult3, mhc_mult * hidden_size] fp32
        mhc_norm_weight: [mhc_mult * hidden_size] fp32 or None
        mhc_norm_eps:    float

    Returns:
        output: [n0, n1, mhc_mult3] fp32
    """
    return MhcPreNormFnTritonFn.apply(residual, mhc_fn, mhc_norm_weight, mhc_norm_eps)
