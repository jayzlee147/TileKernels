"""Triton-fused mhc_pre_apply_mix FWD/BWD for TileKernels.

Math (FWD):
    output[h] = Σ_m mix[m] * x[m, h]

    where: x   = (N, mhc, H) bf16
           mix = (N, mhc)    fp32
           out = (N, H)      bf16

Math (BWD):
    d_mix[m]   = Σ_h o_grad[h] * x[m, h]   (reduction over h)
    d_x[m, h]  = mix[m] * o_grad[h]         (broadcast multiply)
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
def _pre_apply_mix_fwd_kernel(
    X_PTR,    # [N, 4, H] bf16, contiguous
    MIX_PTR,  # [N, 4]    fp32, contiguous
    O_PTR,    # [N, H]    bf16, contiguous
    N,
    H: tl.constexpr,
    H_BLK: tl.constexpr,
):
    pid_n = tl.program_id(0)

    # Load mix[4] into registers
    mix_base = pid_n * 4
    m0 = tl.load(MIX_PTR + mix_base + 0).to(tl.float32)
    m1 = tl.load(MIX_PTR + mix_base + 1).to(tl.float32)
    m2 = tl.load(MIX_PTR + mix_base + 2).to(tl.float32)
    m3 = tl.load(MIX_PTR + mix_base + 3).to(tl.float32)

    x_base = pid_n * 4 * H
    o_base = pid_n * H

    for h_start in range(0, H, H_BLK):
        h_offs = h_start + tl.arange(0, H_BLK)
        mask = h_offs < H

        # Load x[4, h_blk]
        x0 = tl.load(X_PTR + x_base + 0 * H + h_offs, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(X_PTR + x_base + 1 * H + h_offs, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(X_PTR + x_base + 2 * H + h_offs, mask=mask, other=0.0).to(tl.float32)
        x3 = tl.load(X_PTR + x_base + 3 * H + h_offs, mask=mask, other=0.0).to(tl.float32)

        # output[h] = Σ_m mix[m] * x[m, h]
        o = m0 * x0 + m1 * x1 + m2 * x2 + m3 * x3

        tl.store(O_PTR + o_base + h_offs, o.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# BWD kernel
# ---------------------------------------------------------------------------


@triton.jit
def _pre_apply_mix_bwd_kernel(
    OG_PTR,   # [N, H]    o_grad, bf16, contiguous
    X_PTR,    # [N, 4, H] x, bf16, contiguous
    MIX_PTR,  # [N, 4]    mix, fp32, contiguous
    XG_PTR,   # [N, 4, H] x_grad, bf16, contiguous (read-modify-write)
    MG_PTR,   # [N, 4]    mix_grad, fp32, contiguous (output)
    N,
    H: tl.constexpr,
    H_BLK: tl.constexpr,
):
    pid_n = tl.program_id(0)

    # Load mix[4]
    mix_base = pid_n * 4
    m0 = tl.load(MIX_PTR + mix_base + 0).to(tl.float32)
    m1 = tl.load(MIX_PTR + mix_base + 1).to(tl.float32)
    m2 = tl.load(MIX_PTR + mix_base + 2).to(tl.float32)
    m3 = tl.load(MIX_PTR + mix_base + 3).to(tl.float32)

    # Accumulators for d_mix[m] = Σ_h o_grad[h] * x[m, h]
    dm0 = tl.zeros([], dtype=tl.float32)
    dm1 = tl.zeros([], dtype=tl.float32)
    dm2 = tl.zeros([], dtype=tl.float32)
    dm3 = tl.zeros([], dtype=tl.float32)

    og_base = pid_n * H
    x_base = pid_n * 4 * H
    xg_base = pid_n * 4 * H

    for h_start in range(0, H, H_BLK):
        h_offs = h_start + tl.arange(0, H_BLK)
        mask = h_offs < H

        # Load o_grad[h_blk]
        og = tl.load(OG_PTR + og_base + h_offs, mask=mask, other=0.0).to(tl.float32)

        # Load x[4, h_blk]
        x0 = tl.load(X_PTR + x_base + 0 * H + h_offs, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(X_PTR + x_base + 1 * H + h_offs, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(X_PTR + x_base + 2 * H + h_offs, mask=mask, other=0.0).to(tl.float32)
        x3 = tl.load(X_PTR + x_base + 3 * H + h_offs, mask=mask, other=0.0).to(tl.float32)

        # d_mix[m] += Σ_h o_grad[h] * x[m, h]
        dm0 += tl.sum(og * x0)
        dm1 += tl.sum(og * x1)
        dm2 += tl.sum(og * x2)
        dm3 += tl.sum(og * x3)

        # d_x[m, h] = mix[m] * o_grad[h]  (accumulate into x_grad)
        xg0 = tl.load(XG_PTR + xg_base + 0 * H + h_offs, mask=mask, other=0.0).to(tl.float32)
        xg1 = tl.load(XG_PTR + xg_base + 1 * H + h_offs, mask=mask, other=0.0).to(tl.float32)
        xg2 = tl.load(XG_PTR + xg_base + 2 * H + h_offs, mask=mask, other=0.0).to(tl.float32)
        xg3 = tl.load(XG_PTR + xg_base + 3 * H + h_offs, mask=mask, other=0.0).to(tl.float32)

        xg0 += m0 * og
        xg1 += m1 * og
        xg2 += m2 * og
        xg3 += m3 * og

        tl.store(XG_PTR + xg_base + 0 * H + h_offs, xg0.to(tl.bfloat16), mask=mask)
        tl.store(XG_PTR + xg_base + 1 * H + h_offs, xg1.to(tl.bfloat16), mask=mask)
        tl.store(XG_PTR + xg_base + 2 * H + h_offs, xg2.to(tl.bfloat16), mask=mask)
        tl.store(XG_PTR + xg_base + 3 * H + h_offs, xg3.to(tl.bfloat16), mask=mask)

    # Store d_mix[4]
    mg_base = pid_n * 4
    tl.store(MG_PTR + mg_base + 0, dm0)
    tl.store(MG_PTR + mg_base + 1, dm1)
    tl.store(MG_PTR + mg_base + 2, dm2)
    tl.store(MG_PTR + mg_base + 3, dm3)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def _pick_h_blk(H: int, default: int = 512) -> int:
    """Pick H_BLK that evenly divides H, capped at default."""
    return math.gcd(H, default)


def pre_apply_mix_fwd_triton(
    x: torch.Tensor,    # [N, 4, H] bf16
    mix: torch.Tensor,  # [N, 4]    fp32
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch the FWD Triton kernel."""
    N = x.shape[0]
    H = x.shape[2]

    if out is None:
        out = torch.empty((N, H), dtype=torch.bfloat16, device=x.device)

    H_BLK = _pick_h_blk(H)
    grid = (N,)

    _pre_apply_mix_fwd_kernel[grid](
        x, mix, out,
        N, H=H, H_BLK=H_BLK,
    )
    return out


def pre_apply_mix_bwd_triton(
    o_grad: torch.Tensor,  # [N, H]    bf16
    x: torch.Tensor,       # [N, 4, H] bf16
    mix: torch.Tensor,     # [N, 4]    fp32
    x_grad: torch.Tensor,  # [N, 4, H] bf16 (read-modify-write)
) -> torch.Tensor:
    """Launch the BWD Triton kernel.

    Returns:
        mix_grad: [N, 4] fp32
    """
    N = o_grad.shape[0]
    H = o_grad.shape[1]

    mix_grad = torch.empty((N, 4), dtype=torch.float32, device=o_grad.device)

    H_BLK = _pick_h_blk(H)
    grid = (N,)

    _pre_apply_mix_bwd_kernel[grid](
        o_grad, x, mix, x_grad, mix_grad,
        N, H=H, H_BLK=H_BLK,
    )
    return mix_grad


# ---------------------------------------------------------------------------
# Autograd wrapper
# ---------------------------------------------------------------------------


class MhcPreApplyMixTritonFn(torch.autograd.Function):
    """Autograd wrapper for Triton mhc_pre_apply_mix FWD/BWD."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        mix: torch.Tensor,
    ) -> torch.Tensor:
        h = x.shape[-1]
        mhc = mix.shape[-2]

        assert x.dtype == torch.bfloat16
        assert mix.dtype == torch.float32
        assert mhc == 4
        assert mix.shape[-1] == 1

        x_c = x.contiguous()
        mix_c = mix.contiguous()

        out = pre_apply_mix_fwd_triton(
            x_c.view(-1, mhc, h),
            mix_c.view(-1, mhc),
        )
        out = out.view(*x.shape[:-2], h)

        ctx.save_for_backward(x_c, mix_c)
        ctx.h = h
        ctx.mhc = mhc
        ctx.outer_shape = x.shape[:-2]

        return out

    @staticmethod
    def backward(ctx, o_grad):
        x, mix = ctx.saved_tensors
        h = ctx.h
        mhc = ctx.mhc
        N = x.view(-1, mhc, h).shape[0]

        # Match the tilelang version: support grad_from_mhc_post accumulation
        if hasattr(x.untyped_storage(), 'grad_from_mhc_post'):
            x_grad_tensor = x.untyped_storage().grad_from_mhc_post
            mix_grad = pre_apply_mix_bwd_triton(
                o_grad.contiguous().view(N, h),
                x.view(N, mhc, h),
                mix.view(N, mhc),
                x_grad_tensor.view(N, mhc, h),
            )
            x_grad_out = None
        else:
            x_grad_tensor = torch.zeros_like(x)
            mix_grad = pre_apply_mix_bwd_triton(
                o_grad.contiguous().view(N, h),
                x.view(N, mhc, h),
                mix.view(N, mhc),
                x_grad_tensor.view(N, mhc, h),
            )
            x_grad_out = x_grad_tensor

        return x_grad_out, mix_grad.view_as(mix)


def mhc_pre_apply_mix_triton(
    x: torch.Tensor,
    mix: torch.Tensor,
) -> torch.Tensor:
    """Triton-fused mhc_pre_apply_mix (drop-in replacement for mhc_pre_apply_mix_ref).

    Args:
        x:   [*, mhc, H] bf16
        mix: [*, mhc, 1] fp32

    Returns:
        output: [*, H] bf16
    """
    return MhcPreApplyMixTritonFn.apply(x, mix)
