"""Triton-fused mhc_post FWD/BWD for TileKernels.

Math (FWD):
    x_out[m, h] = c[m] * d[h] + Σ_k a[k, m] * b[k, h]

    where: a = comb_res_mix  (N, mhc, mhc) float32
           b = residual      (N, mhc, H)   bf16
           c = post_layer_mix(N, mhc)       float32
           d = x_input       (N, H)         bf16
           x_out             (N, mhc, H)    bf16

Math (BWD):
    da[i,j] = Σ_h b[i,h] * dx[j,h]      (reduction over h)
    db[i,h] = Σ_j a[i,j] * dx[j,h]      (matmul)
    dc[m]   = Σ_h d[h] * dx[m,h]         (reduction over h)
    dd[h]   = Σ_m c[m] * dx[m,h]         (weighted sum)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# FWD kernel
# ---------------------------------------------------------------------------


@triton.jit
def _mhc_post_fwd_kernel(
    A_PTR,   # [N, 4, 4] comb_res_mix, fp32, contiguous
    B_PTR,   # [N, 4, H] residual, bf16, contiguous
    C_PTR,   # [N, 4]    post_layer_mix, fp32, contiguous
    D_PTR,   # [N, H]    x_input, bf16, contiguous
    X_PTR,   # [N, 4, H] output, bf16, contiguous
    N,
    H: tl.constexpr,
    H_BLK: tl.constexpr,
):
    pid_n = tl.program_id(0)

    # Load a[4,4] and c[4] into registers (small, fits easily)
    a_base = pid_n * 16  # 4*4
    a00 = tl.load(A_PTR + a_base + 0).to(tl.float32)
    a01 = tl.load(A_PTR + a_base + 1).to(tl.float32)
    a02 = tl.load(A_PTR + a_base + 2).to(tl.float32)
    a03 = tl.load(A_PTR + a_base + 3).to(tl.float32)
    a10 = tl.load(A_PTR + a_base + 4).to(tl.float32)
    a11 = tl.load(A_PTR + a_base + 5).to(tl.float32)
    a12 = tl.load(A_PTR + a_base + 6).to(tl.float32)
    a13 = tl.load(A_PTR + a_base + 7).to(tl.float32)
    a20 = tl.load(A_PTR + a_base + 8).to(tl.float32)
    a21 = tl.load(A_PTR + a_base + 9).to(tl.float32)
    a22 = tl.load(A_PTR + a_base + 10).to(tl.float32)
    a23 = tl.load(A_PTR + a_base + 11).to(tl.float32)
    a30 = tl.load(A_PTR + a_base + 12).to(tl.float32)
    a31 = tl.load(A_PTR + a_base + 13).to(tl.float32)
    a32 = tl.load(A_PTR + a_base + 14).to(tl.float32)
    a33 = tl.load(A_PTR + a_base + 15).to(tl.float32)

    c_base = pid_n * 4
    c0 = tl.load(C_PTR + c_base + 0).to(tl.float32)
    c1 = tl.load(C_PTR + c_base + 1).to(tl.float32)
    c2 = tl.load(C_PTR + c_base + 2).to(tl.float32)
    c3 = tl.load(C_PTR + c_base + 3).to(tl.float32)

    # FWD: x[m, h] = c[m]*d[h] + sum_k a[k, m]*b[k, h]
    # Note: a is (mhc_in, mhc_out) = a[k, m], but in the tilelang kernel
    # the inner loop is a_local[i_mhci, i_mhco] * b_local[i_mhci, i1_h]
    # which means a is stored row-major as [k, m] and we sum over k.
    # The einsum in ref is 'abmn,abmc->abnc' where m=k_in, n=k_out, c=h
    # So a[k,m] means: for output index m, we sum a[k,m]*b[k,h] over k.
    # In tilelang: a_local[i_mhci, i_mhco] => a[i_mhci, i_mhco].
    # The memory layout is [N, mhc, mhc] = [N, mhc_row, mhc_col].
    # tilelang loads a[pid_n, 0, 0] into a_local[mhc, mhc], so
    # a_local[i_mhci, i_mhco] = a[pid_n, i_mhci, i_mhco].
    # x[i_mhco, h] += a[i_mhci, i_mhco] * b[i_mhci, h]
    # So for output row m: x[m,h] = c[m]*d[h] + Σ_k a[k,m]*b[k,h]

    b_base = pid_n * 4 * H  # b[pid_n, 0, 0]
    d_base = pid_n * H      # d[pid_n, 0]
    x_base = pid_n * 4 * H  # x[pid_n, 0, 0]

    for h_start in range(0, H, H_BLK):
        h_offs = h_start + tl.arange(0, H_BLK)
        mask = h_offs < H

        # Load d[h_blk]
        d_vec = tl.load(D_PTR + d_base + h_offs, mask=mask, other=0.0).to(tl.float32)

        # Load b[4, h_blk]
        b0 = tl.load(B_PTR + b_base + 0 * H + h_offs, mask=mask, other=0.0).to(tl.float32)
        b1 = tl.load(B_PTR + b_base + 1 * H + h_offs, mask=mask, other=0.0).to(tl.float32)
        b2 = tl.load(B_PTR + b_base + 2 * H + h_offs, mask=mask, other=0.0).to(tl.float32)
        b3 = tl.load(B_PTR + b_base + 3 * H + h_offs, mask=mask, other=0.0).to(tl.float32)

        # x[0, h] = c0*d + a[0,0]*b0 + a[1,0]*b1 + a[2,0]*b2 + a[3,0]*b3
        x0 = c0 * d_vec + a00 * b0 + a10 * b1 + a20 * b2 + a30 * b3
        x1 = c1 * d_vec + a01 * b0 + a11 * b1 + a21 * b2 + a31 * b3
        x2 = c2 * d_vec + a02 * b0 + a12 * b1 + a22 * b2 + a32 * b3
        x3 = c3 * d_vec + a03 * b0 + a13 * b1 + a23 * b2 + a33 * b3

        tl.store(X_PTR + x_base + 0 * H + h_offs, x0.to(tl.bfloat16), mask=mask)
        tl.store(X_PTR + x_base + 1 * H + h_offs, x1.to(tl.bfloat16), mask=mask)
        tl.store(X_PTR + x_base + 2 * H + h_offs, x2.to(tl.bfloat16), mask=mask)
        tl.store(X_PTR + x_base + 3 * H + h_offs, x3.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# BWD kernel
# ---------------------------------------------------------------------------


@triton.jit
def _mhc_post_bwd_kernel(
    DX_PTR,  # [N, 4, H] dx (grad of output), bf16
    A_PTR,   # [N, 4, 4] comb_res_mix, fp32
    B_PTR,   # [N, 4, H] residual, bf16
    C_PTR,   # [N, 4]    post_layer_mix, fp32
    D_PTR,   # [N, H]    x_input, bf16
    DA_PTR,  # [N, 4, 4] output grad for a, fp32
    DB_PTR,  # [N, 4, H] output grad for b, bf16
    DC_PTR,  # [N, 4]    output grad for c, fp32
    DD_PTR,  # [N, H]    output grad for d, bf16
    N,
    H: tl.constexpr,
    H_BLK: tl.constexpr,
):
    pid_n = tl.program_id(0)

    # Load a[4,4] and c[4]
    a_base = pid_n * 16
    a00 = tl.load(A_PTR + a_base + 0).to(tl.float32)
    a01 = tl.load(A_PTR + a_base + 1).to(tl.float32)
    a02 = tl.load(A_PTR + a_base + 2).to(tl.float32)
    a03 = tl.load(A_PTR + a_base + 3).to(tl.float32)
    a10 = tl.load(A_PTR + a_base + 4).to(tl.float32)
    a11 = tl.load(A_PTR + a_base + 5).to(tl.float32)
    a12 = tl.load(A_PTR + a_base + 6).to(tl.float32)
    a13 = tl.load(A_PTR + a_base + 7).to(tl.float32)
    a20 = tl.load(A_PTR + a_base + 8).to(tl.float32)
    a21 = tl.load(A_PTR + a_base + 9).to(tl.float32)
    a22 = tl.load(A_PTR + a_base + 10).to(tl.float32)
    a23 = tl.load(A_PTR + a_base + 11).to(tl.float32)
    a30 = tl.load(A_PTR + a_base + 12).to(tl.float32)
    a31 = tl.load(A_PTR + a_base + 13).to(tl.float32)
    a32 = tl.load(A_PTR + a_base + 14).to(tl.float32)
    a33 = tl.load(A_PTR + a_base + 15).to(tl.float32)

    c_base = pid_n * 4
    c0 = tl.load(C_PTR + c_base + 0).to(tl.float32)
    c1 = tl.load(C_PTR + c_base + 1).to(tl.float32)
    c2 = tl.load(C_PTR + c_base + 2).to(tl.float32)
    c3 = tl.load(C_PTR + c_base + 3).to(tl.float32)

    # Accumulators for da[4,4] and dc[4] (reductions over h)
    da00 = tl.zeros([], dtype=tl.float32)
    da01 = tl.zeros([], dtype=tl.float32)
    da02 = tl.zeros([], dtype=tl.float32)
    da03 = tl.zeros([], dtype=tl.float32)
    da10 = tl.zeros([], dtype=tl.float32)
    da11 = tl.zeros([], dtype=tl.float32)
    da12 = tl.zeros([], dtype=tl.float32)
    da13 = tl.zeros([], dtype=tl.float32)
    da20 = tl.zeros([], dtype=tl.float32)
    da21 = tl.zeros([], dtype=tl.float32)
    da22 = tl.zeros([], dtype=tl.float32)
    da23 = tl.zeros([], dtype=tl.float32)
    da30 = tl.zeros([], dtype=tl.float32)
    da31 = tl.zeros([], dtype=tl.float32)
    da32 = tl.zeros([], dtype=tl.float32)
    da33 = tl.zeros([], dtype=tl.float32)

    dc0 = tl.zeros([], dtype=tl.float32)
    dc1 = tl.zeros([], dtype=tl.float32)
    dc2 = tl.zeros([], dtype=tl.float32)
    dc3 = tl.zeros([], dtype=tl.float32)

    dx_base = pid_n * 4 * H
    b_base = pid_n * 4 * H
    d_base = pid_n * H
    db_base = pid_n * 4 * H
    dd_base = pid_n * H

    for h_start in range(0, H, H_BLK):
        h_offs = h_start + tl.arange(0, H_BLK)
        mask = h_offs < H

        # Load dx[4, h_blk]
        dx0 = tl.load(DX_PTR + dx_base + 0 * H + h_offs, mask=mask, other=0.0).to(tl.float32)
        dx1 = tl.load(DX_PTR + dx_base + 1 * H + h_offs, mask=mask, other=0.0).to(tl.float32)
        dx2 = tl.load(DX_PTR + dx_base + 2 * H + h_offs, mask=mask, other=0.0).to(tl.float32)
        dx3 = tl.load(DX_PTR + dx_base + 3 * H + h_offs, mask=mask, other=0.0).to(tl.float32)

        # Load b[4, h_blk]
        b0 = tl.load(B_PTR + b_base + 0 * H + h_offs, mask=mask, other=0.0).to(tl.float32)
        b1 = tl.load(B_PTR + b_base + 1 * H + h_offs, mask=mask, other=0.0).to(tl.float32)
        b2 = tl.load(B_PTR + b_base + 2 * H + h_offs, mask=mask, other=0.0).to(tl.float32)
        b3 = tl.load(B_PTR + b_base + 3 * H + h_offs, mask=mask, other=0.0).to(tl.float32)

        # Load d[h_blk]
        d_vec = tl.load(D_PTR + d_base + h_offs, mask=mask, other=0.0).to(tl.float32)

        # da[i,j] = Σ_h b[i,h] * dx[j,h]
        # In the tilelang kernel: da_reducer[i_mhci, i_mhco] += b_local[i_mhci, i1_h] * dx_local[i_mhco, i1_h]
        # So da[i,j] += b[i,h] * dx[j,h] accumulated across h blocks
        da00 += tl.sum(b0 * dx0)
        da01 += tl.sum(b0 * dx1)
        da02 += tl.sum(b0 * dx2)
        da03 += tl.sum(b0 * dx3)
        da10 += tl.sum(b1 * dx0)
        da11 += tl.sum(b1 * dx1)
        da12 += tl.sum(b1 * dx2)
        da13 += tl.sum(b1 * dx3)
        da20 += tl.sum(b2 * dx0)
        da21 += tl.sum(b2 * dx1)
        da22 += tl.sum(b2 * dx2)
        da23 += tl.sum(b2 * dx3)
        da30 += tl.sum(b3 * dx0)
        da31 += tl.sum(b3 * dx1)
        da32 += tl.sum(b3 * dx2)
        da33 += tl.sum(b3 * dx3)

        # db[i,h] = Σ_j a[i,j] * dx[j,h]
        # From tilelang: db_local[i_mhci, i1_h] += a_local[i_mhci, i_mhco] * dx_local[i_mhco, i1_h]
        db0 = a00 * dx0 + a01 * dx1 + a02 * dx2 + a03 * dx3
        db1 = a10 * dx0 + a11 * dx1 + a12 * dx2 + a13 * dx3
        db2 = a20 * dx0 + a21 * dx1 + a22 * dx2 + a23 * dx3
        db3 = a30 * dx0 + a31 * dx1 + a32 * dx2 + a33 * dx3

        tl.store(DB_PTR + db_base + 0 * H + h_offs, db0.to(tl.bfloat16), mask=mask)
        tl.store(DB_PTR + db_base + 1 * H + h_offs, db1.to(tl.bfloat16), mask=mask)
        tl.store(DB_PTR + db_base + 2 * H + h_offs, db2.to(tl.bfloat16), mask=mask)
        tl.store(DB_PTR + db_base + 3 * H + h_offs, db3.to(tl.bfloat16), mask=mask)

        # dc[m] = Σ_h d[h] * dx[m,h]
        dc0 += tl.sum(d_vec * dx0)
        dc1 += tl.sum(d_vec * dx1)
        dc2 += tl.sum(d_vec * dx2)
        dc3 += tl.sum(d_vec * dx3)

        # dd[h] = Σ_m c[m] * dx[m,h]
        dd_vec = c0 * dx0 + c1 * dx1 + c2 * dx2 + c3 * dx3
        tl.store(DD_PTR + dd_base + h_offs, dd_vec.to(tl.bfloat16), mask=mask)

    # Store da[4,4]
    da_base = pid_n * 16
    tl.store(DA_PTR + da_base + 0, da00)
    tl.store(DA_PTR + da_base + 1, da01)
    tl.store(DA_PTR + da_base + 2, da02)
    tl.store(DA_PTR + da_base + 3, da03)
    tl.store(DA_PTR + da_base + 4, da10)
    tl.store(DA_PTR + da_base + 5, da11)
    tl.store(DA_PTR + da_base + 6, da12)
    tl.store(DA_PTR + da_base + 7, da13)
    tl.store(DA_PTR + da_base + 8, da20)
    tl.store(DA_PTR + da_base + 9, da21)
    tl.store(DA_PTR + da_base + 10, da22)
    tl.store(DA_PTR + da_base + 11, da23)
    tl.store(DA_PTR + da_base + 12, da30)
    tl.store(DA_PTR + da_base + 13, da31)
    tl.store(DA_PTR + da_base + 14, da32)
    tl.store(DA_PTR + da_base + 15, da33)

    # Store dc[4]
    dc_base = pid_n * 4
    tl.store(DC_PTR + dc_base + 0, dc0)
    tl.store(DC_PTR + dc_base + 1, dc1)
    tl.store(DC_PTR + dc_base + 2, dc2)
    tl.store(DC_PTR + dc_base + 3, dc3)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def _pick_h_blk(H: int, default: int = 512) -> int:
    """Pick H_BLK that evenly divides H, capped at default."""
    import math
    return math.gcd(H, default)


def mhc_post_fwd_triton(
    comb_res_mix: torch.Tensor,   # [N, 4, 4] fp32
    residual: torch.Tensor,       # [N, 4, H] bf16
    post_layer_mix: torch.Tensor, # [N, 4]    fp32
    x_input: torch.Tensor,        # [N, H]    bf16
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch the FWD Triton kernel."""
    N = comb_res_mix.shape[0]
    H = residual.shape[2]

    if out is None:
        out = torch.empty_like(residual)

    H_BLK = _pick_h_blk(H)
    grid = (N,)

    _mhc_post_fwd_kernel[grid](
        comb_res_mix, residual, post_layer_mix, x_input, out,
        N, H=H, H_BLK=H_BLK,
    )
    return out


def mhc_post_bwd_triton(
    dx: torch.Tensor,             # [N, 4, H] bf16
    comb_res_mix: torch.Tensor,   # [N, 4, 4] fp32
    residual: torch.Tensor,       # [N, 4, H] bf16
    post_layer_mix: torch.Tensor, # [N, 4]    fp32
    x_input: torch.Tensor,        # [N, H]    bf16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the BWD Triton kernel.

    Returns:
        (da, db, dc, dd) = grads for (comb_res_mix, residual, post_layer_mix, x_input)
    """
    N = dx.shape[0]
    H = dx.shape[2]

    da = torch.empty_like(comb_res_mix)
    db = torch.empty_like(residual)
    dc = torch.empty((N, 4), dtype=torch.float32, device=dx.device)
    dd = torch.empty((N, H), dtype=torch.bfloat16, device=dx.device)

    H_BLK = _pick_h_blk(H)
    grid = (N,)

    _mhc_post_bwd_kernel[grid](
        dx, comb_res_mix, residual, post_layer_mix, x_input,
        da, db, dc, dd,
        N, H=H, H_BLK=H_BLK,
    )
    return da, db, dc, dd


# ---------------------------------------------------------------------------
# Autograd wrapper
# ---------------------------------------------------------------------------


class MhcPostTritonFn(torch.autograd.Function):
    """Autograd wrapper for Triton mhc_post FWD/BWD."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        num_seqs, num_tokens, mhc, hidden = residual.shape

        assert x.dtype == torch.bfloat16
        assert residual.dtype == torch.bfloat16
        assert post_layer_mix.dtype == torch.float32
        assert comb_res_mix.dtype == torch.float32
        assert x.shape == (num_seqs, num_tokens, hidden)
        assert post_layer_mix.shape == (num_seqs, num_tokens, mhc, 1)
        assert comb_res_mix.shape == (num_seqs, num_tokens, mhc, mhc)

        residual_c = residual.contiguous()
        x_c = x.contiguous()
        post_layer_mix_c = post_layer_mix.contiguous()
        comb_res_mix_c = comb_res_mix.contiguous()

        out = mhc_post_fwd_triton(
            comb_res_mix_c.flatten(0, 1),          # [N, mhc, mhc]
            residual_c.flatten(0, 1),              # [N, mhc, H]
            post_layer_mix_c.flatten(0, 1).squeeze(-1),  # [N, mhc]
            x_c.flatten(0, 1),                     # [N, H]
        )
        out = out.view(num_seqs, num_tokens, mhc, hidden)

        ctx.save_for_backward(x_c, residual_c, post_layer_mix_c, comb_res_mix_c)
        ctx.shapes = (num_seqs, num_tokens, mhc, hidden)
        return out

    @staticmethod
    def backward(ctx, d_out):
        x, residual, post_layer_mix, comb_res_mix = ctx.saved_tensors
        num_seqs, num_tokens, mhc, hidden = ctx.shapes
        N = num_seqs * num_tokens

        da, db, dc, dd = mhc_post_bwd_triton(
            d_out.contiguous().view(N, mhc, hidden),
            comb_res_mix.view(N, mhc, mhc),
            residual.view(N, mhc, hidden),
            post_layer_mix.view(N, mhc),
            x.view(N, hidden),
        )

        return (
            dd.view(num_seqs, num_tokens, hidden),
            db.view(num_seqs, num_tokens, mhc, hidden),
            dc.view(num_seqs, num_tokens, mhc, 1),
            da.view(num_seqs, num_tokens, mhc, mhc),
        )


def mhc_post_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """Triton-fused mhc_post (drop-in replacement for mhc_post_ref).

    Args:
        x:              [n0, n1, H]        bf16  (x_input / d)
        residual:       [n0, n1, mhc, H]   bf16  (b)
        post_layer_mix: [n0, n1, mhc, 1]   fp32  (c)
        comb_res_mix:   [n0, n1, mhc, mhc] fp32  (a)

    Returns:
        output:         [n0, n1, mhc, H]   bf16
    """
    return MhcPostTritonFn.apply(x, residual, post_layer_mix, comb_res_mix)
