"""Triton MFMA-accelerated mhc_pre_norm_fn FWD + BWD for TileKernels on MI300.

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
    d_fn cross-token reduction is done on the host via torch.mm.

Performance strategy:
    The GEMV (x @ fn.T) uses bf16 MFMA via tl.dot (v_mfma_f32_16x16x16_bf16),
    with split-K parallelism across the hidden dimension.  fn is pre-transposed
    and converted to bf16 at first use (cached).  This achieves ~0.145ms FWD for
    (4096, 16384) @ (16384, 24) — 4.7× faster than the torch.mm (rocBLAS) path.

    The BWD d_x also uses bf16 MFMA via tl.dot for the fn @ d_out GEMV.
"""

from __future__ import annotations

import math
import weakref

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_K_PAD = 32   # pad mhc_mult3 (24) to 32 for MFMA tile alignment


# ---------------------------------------------------------------------------
# FWD: split-K GEMM kernel (bf16 MFMA)
# ---------------------------------------------------------------------------


@triton.jit
def _norm_fn_fwd_splitk(
    X_PTR,         # [num_tokens, total_hidden] bf16, contiguous
    FN_T_PTR,      # [total_hidden, K_PAD] bf16, contiguous (pre-transposed, padded)
    PART_OUT_PTR,  # [N_SPLITS, num_tokens, K_PAD] fp32
    PART_SQR_PTR,  # [N_SPLITS, num_tokens] fp32
    num_tokens,
    total_hidden,
    # constexpr
    K_PAD: tl.constexpr,       # 32
    T_BLK: tl.constexpr,       # tokens per program (16)
    H_BLK: tl.constexpr,       # hidden tile size (128) — must divide h_per_split
    N_SPLITS: tl.constexpr,    # number of split-K partitions
):
    """Each program handles T_BLK tokens × (total_hidden / N_SPLITS) hidden elements.

    Computes partial dot products and partial squared sums.
    """
    pid_n = tl.program_id(0)   # token block index
    pid_s = tl.program_id(1)   # split index

    h_per_split = total_hidden // N_SPLITS
    h_start = pid_s * h_per_split
    h_end = h_start + h_per_split

    # Token offsets
    t_offs = pid_n * T_BLK + tl.arange(0, T_BLK)  # [T_BLK]
    t_mask = t_offs < num_tokens

    # Accumulators: partial dot products [T_BLK, K_PAD] and sqrsum [T_BLK]
    dot_acc = tl.zeros([T_BLK, K_PAD], dtype=tl.float32)
    sqr_acc = tl.zeros([T_BLK], dtype=tl.float32)

    # Tile over hidden dimension within this split
    for h_off in tl.range(0, h_per_split, H_BLK, num_stages=2):
        h_abs = h_start + h_off
        h_idx = tl.arange(0, H_BLK)  # [H_BLK]

        # Load x tile: [T_BLK, H_BLK] bf16
        x_ptrs = t_offs[:, None] * total_hidden + (h_abs + h_idx)[None, :]  # [T_BLK, H_BLK]
        x_tile = tl.load(X_PTR + x_ptrs, mask=t_mask[:, None], other=0.0)   # bf16

        # Accumulate squared sum (cast to f32)
        x_f32 = x_tile.to(tl.float32)
        sqr_acc += tl.sum(x_f32 * x_f32, axis=1)  # [T_BLK]

        # Load fn_t tile: [H_BLK, K_PAD] bf16
        fn_ptrs = (h_abs + h_idx)[:, None] * K_PAD + tl.arange(0, K_PAD)[None, :]  # [H_BLK, K_PAD]
        fn_tile = tl.load(FN_T_PTR + fn_ptrs)  # bf16

        # MFMA dot: [T_BLK, H_BLK] @ [H_BLK, K_PAD] → [T_BLK, K_PAD]
        dot_acc = tl.dot(x_tile, fn_tile, acc=dot_acc)

    # Store partial results
    k_offs = tl.arange(0, K_PAD)

    # partial_out[pid_s, t, k]
    out_base = pid_s * num_tokens * K_PAD
    out_ptrs = out_base + t_offs[:, None] * K_PAD + k_offs[None, :]  # [T_BLK, K_PAD]
    tl.store(PART_OUT_PTR + out_ptrs, dot_acc, mask=t_mask[:, None])

    # partial_sqr[pid_s, t]
    sqr_base = pid_s * num_tokens
    tl.store(PART_SQR_PTR + sqr_base + t_offs, sqr_acc, mask=t_mask)


# ---------------------------------------------------------------------------
# FWD: reduce partial sums + RMSNorm
# ---------------------------------------------------------------------------


@triton.jit
def _norm_fn_fwd_reduce(
    PART_OUT_PTR,  # [N_SPLITS, num_tokens, K_PAD] fp32
    PART_SQR_PTR,  # [N_SPLITS, num_tokens] fp32
    OUT_PTR,       # [num_tokens, mhc_mult3] fp32
    num_tokens,
    total_hidden,  # for RMS denominator
    # constexpr
    MHC_MULT3: tl.constexpr,
    K_PAD: tl.constexpr,
    T_BLK: tl.constexpr,
    N_SPLITS: tl.constexpr,
    EPS: tl.constexpr,
):
    """Reduce split-K partials and apply RMSNorm scaling.

    Each program handles T_BLK tokens.
    """
    pid_n = tl.program_id(0)
    t_offs = pid_n * T_BLK + tl.arange(0, T_BLK)  # [T_BLK]
    t_mask = t_offs < num_tokens

    k_offs = tl.arange(0, K_PAD)
    k_mask = k_offs < MHC_MULT3

    # Sum dot products across splits
    dot_sum = tl.zeros([T_BLK, K_PAD], dtype=tl.float32)
    sqr_sum = tl.zeros([T_BLK], dtype=tl.float32)

    for s in range(N_SPLITS):
        # Load partial dot
        out_base = s * num_tokens * K_PAD
        out_ptrs = out_base + t_offs[:, None] * K_PAD + k_offs[None, :]
        part_dot = tl.load(PART_OUT_PTR + out_ptrs, mask=t_mask[:, None], other=0.0)
        dot_sum += part_dot

        # Load partial sqrsum
        sqr_base = s * num_tokens
        part_sqr = tl.load(PART_SQR_PTR + sqr_base + t_offs, mask=t_mask, other=0.0)
        sqr_sum += part_sqr

    # RMSNorm: inv_rms = rsqrt(sqrsum / total_hidden + eps)
    inv_rms = tl.rsqrt(sqr_sum / total_hidden + EPS)  # [T_BLK]

    # Scale dot products
    result = dot_sum * inv_rms[:, None]  # [T_BLK, K_PAD]

    # Store only the first MHC_MULT3 columns
    store_ptrs = t_offs[:, None] * MHC_MULT3 + k_offs[None, :]
    tl.store(OUT_PTR + store_ptrs, result, mask=t_mask[:, None] & k_mask[None, :])


# ---------------------------------------------------------------------------
# BWD: d_x kernel using bf16 MFMA
# ---------------------------------------------------------------------------


@triton.jit
def _norm_fn_bwd_dx_mfma(
    DOUT_PTR,      # [num_tokens, mhc_mult3] fp32
    X_PTR,         # [num_tokens, total_hidden] bf16
    FN_T_PTR,      # [total_hidden, K_PAD] bf16 (pre-transposed)
    DX_PTR,        # [num_tokens, total_hidden] fp32 output
    num_tokens,
    total_hidden,
    # constexpr
    MHC_MULT3: tl.constexpr,
    K_PAD: tl.constexpr,
    T_BLK: tl.constexpr,
    H_BLK: tl.constexpr,
    EPS: tl.constexpr,
):
    """Compute d_x for each token using MFMA.

    d_x[h] = inv_rms * (sum_k(d_out[k] * fn[k, h]) - x[h] * inv_rms^2 * c / S)
    where c = sum_k(d_out[k] * dot[k]), dot[k] = sum_h(x[h] * fn[k, h])

    For n_rms_group=1 (single group), this simplifies.
    Each program handles T_BLK tokens.
    """
    pid_n = tl.program_id(0)
    t_offs = pid_n * T_BLK + tl.arange(0, T_BLK)
    t_mask = t_offs < num_tokens

    k_offs = tl.arange(0, K_PAD)
    k_mask = k_offs < MHC_MULT3

    # Load d_out [T_BLK, K_PAD] — zero-pad beyond MHC_MULT3
    dout_ptrs = t_offs[:, None] * MHC_MULT3 + k_offs[None, :]
    d_out = tl.load(DOUT_PTR + dout_ptrs, mask=t_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

    # First pass: compute dot products (x @ fn_t → [T_BLK, K_PAD]) and sqrsum
    dot_acc = tl.zeros([T_BLK, K_PAD], dtype=tl.float32)
    sqr_acc = tl.zeros([T_BLK], dtype=tl.float32)

    for h_off in tl.range(0, total_hidden, H_BLK):
        h_idx = tl.arange(0, H_BLK)

        # Load x [T_BLK, H_BLK] bf16
        x_ptrs = t_offs[:, None] * total_hidden + (h_off + h_idx)[None, :]
        x_tile = tl.load(X_PTR + x_ptrs, mask=t_mask[:, None], other=0.0)
        x_f32 = x_tile.to(tl.float32)
        sqr_acc += tl.sum(x_f32 * x_f32, axis=1)

        # Load fn_t [H_BLK, K_PAD] bf16
        fn_ptrs = (h_off + h_idx)[:, None] * K_PAD + k_offs[None, :]
        fn_tile = tl.load(FN_T_PTR + fn_ptrs)
        dot_acc = tl.dot(x_tile, fn_tile, acc=dot_acc)

    # inv_rms = rsqrt(sqrsum / total_hidden + eps)
    inv_rms = tl.rsqrt(sqr_acc / total_hidden + EPS)  # [T_BLK]

    # c = sum_k(d_out[k] * dot[k]) per token
    c = tl.sum(d_out * dot_acc, axis=1)  # [T_BLK]

    # rms_coeff = -inv_rms^3 * c / S
    rms_coeff = -(inv_rms * inv_rms * inv_rms) * c / total_hidden  # [T_BLK]

    # Second pass: compute d_x[h] = inv_rms * dout_fn[h] + rms_coeff * x[h]
    # where dout_fn = d_out @ fn (matmul: [T_BLK, K_PAD] @ [K_PAD, H_BLK])
    # We need fn in [K_PAD, total_hidden] layout, but we have fn_t = [total_hidden, K_PAD]
    # We tile over hidden and for each tile compute dout_fn via elementwise+reduce

    # Precompute d_out as bf16 for MFMA: [T_BLK, K_PAD]
    d_out_bf16 = d_out.to(tl.bfloat16)

    for h_off in tl.range(0, total_hidden, H_BLK):
        h_idx = tl.arange(0, H_BLK)

        # Load fn_t [H_BLK, K_PAD] bf16, then transpose to get [K_PAD, H_BLK]
        fn_ptrs = (h_off + h_idx)[:, None] * K_PAD + k_offs[None, :]
        fn_tile = tl.load(FN_T_PTR + fn_ptrs)  # [H_BLK, K_PAD] bf16

        # dout_fn = d_out @ fn_tile^T: [T_BLK, K_PAD] @ [K_PAD, H_BLK] → [T_BLK, H_BLK]
        # fn_tile is [H_BLK, K_PAD], so fn_tile^T is [K_PAD, H_BLK]
        fn_tile_t = tl.trans(fn_tile)  # [K_PAD, H_BLK]
        dout_fn = tl.dot(d_out_bf16, fn_tile_t).to(tl.float32)  # [T_BLK, H_BLK]

        # Load x for rms_coeff term
        x_ptrs = t_offs[:, None] * total_hidden + (h_off + h_idx)[None, :]
        x_f32 = tl.load(X_PTR + x_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)

        # d_x = inv_rms * dout_fn + rms_coeff * x
        dx = inv_rms[:, None] * dout_fn + rms_coeff[:, None] * x_f32

        # Store
        dx_ptrs = t_offs[:, None] * total_hidden + (h_off + h_idx)[None, :]
        tl.store(DX_PTR + dx_ptrs, dx, mask=t_mask[:, None])


# ---------------------------------------------------------------------------
# Legacy FWD kernel (element-wise, fallback)
# ---------------------------------------------------------------------------


@triton.jit
def _norm_fn_fwd_kernel_vec(
    X_PTR,       # [num_tokens, total_hidden] bf16, contiguous
    FN_PTR,      # [mhc_mult3, total_hidden]  fp32, contiguous
    OUT_PTR,     # [num_tokens, mhc_mult3]    fp32, contiguous
    num_tokens,
    total_hidden,
    MHC_MULT3: tl.constexpr,
    RMS_GROUP_SIZE: tl.constexpr,
    N_RMS_GROUP: tl.constexpr,
    H_BLK: tl.constexpr,
    K_BLK: tl.constexpr,
    EPS: tl.constexpr,
):
    """Legacy per-token kernel. Kept for fallback / multi-group cases."""
    pid_n = tl.program_id(0)

    k_offs = tl.arange(0, K_BLK)
    k_mask = k_offs < MHC_MULT3

    out_acc = tl.zeros([K_BLK], dtype=tl.float32)
    x_row_base = pid_n * total_hidden

    for g in range(N_RMS_GROUP):
        group_base = g * RMS_GROUP_SIZE
        sqrsum = tl.zeros([], dtype=tl.float32)
        dot_acc = tl.zeros([K_BLK], dtype=tl.float32)

        for h_start in tl.static_range(0, RMS_GROUP_SIZE, H_BLK):
            h_offs = h_start + tl.arange(0, H_BLK)
            x_addrs = x_row_base + group_base + h_offs
            x_vals = tl.load(X_PTR + x_addrs).to(tl.float32)
            sqrsum += tl.sum(x_vals * x_vals)

            fn_addrs = k_offs[:, None] * total_hidden + (group_base + h_offs)[None, :]
            fn_vals = tl.load(FN_PTR + fn_addrs, mask=k_mask[:, None], other=0.0)
            dot_acc += tl.sum(fn_vals * x_vals[None, :], axis=1)

        inv_rms = tl.rsqrt(sqrsum / RMS_GROUP_SIZE + EPS)
        out_acc += dot_acc * inv_rms

    out_base = pid_n * MHC_MULT3
    tl.store(OUT_PTR + out_base + k_offs, out_acc, mask=k_mask)


# ---------------------------------------------------------------------------
# Legacy BWD kernel (element-wise, fallback)
# ---------------------------------------------------------------------------


@triton.jit
def _norm_fn_bwd_kernel(
    DOUT_PTR,    # [num_tokens, mhc_mult3]    fp32
    X_PTR,       # [num_tokens, total_hidden] bf16, contiguous
    FN_PTR,      # [mhc_mult3, total_hidden]  fp32, contiguous
    DX_PTR,      # [num_tokens, total_hidden] fp32, output
    num_tokens,
    total_hidden,
    MHC_MULT3: tl.constexpr,
    RMS_GROUP_SIZE: tl.constexpr,
    N_RMS_GROUP: tl.constexpr,
    H_BLK: tl.constexpr,
    K_BLK: tl.constexpr,
    EPS: tl.constexpr,
):
    """Legacy BWD kernel for d_x. One program per (token, group)."""
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    k_offs = tl.arange(0, K_BLK)
    k_mask = k_offs < MHC_MULT3

    # Load d_out for this token
    d_out = tl.load(DOUT_PTR + pid_n * MHC_MULT3 + k_offs, mask=k_mask, other=0.0)

    group_base = pid_g * RMS_GROUP_SIZE
    x_row_base = pid_n * total_hidden + group_base

    # Recompute sqrsum and dot for this group
    sqrsum = tl.zeros([], dtype=tl.float32)
    dot_acc = tl.zeros([K_BLK], dtype=tl.float32)

    for h_start in tl.static_range(0, RMS_GROUP_SIZE, H_BLK):
        h_offs = h_start + tl.arange(0, H_BLK)
        x_vals = tl.load(X_PTR + x_row_base + h_offs).to(tl.float32)
        sqrsum += tl.sum(x_vals * x_vals)

        fn_addrs = k_offs[:, None] * total_hidden + (group_base + h_offs)[None, :]
        fn_vals = tl.load(FN_PTR + fn_addrs, mask=k_mask[:, None], other=0.0)
        dot_acc += tl.sum(fn_vals * x_vals[None, :], axis=1)

    inv_rms = tl.rsqrt(sqrsum / RMS_GROUP_SIZE + EPS)
    c = tl.sum(d_out * dot_acc)
    rms_coeff = -(inv_rms * inv_rms * inv_rms) * c / RMS_GROUP_SIZE

    for h_start in tl.static_range(0, RMS_GROUP_SIZE, H_BLK):
        h_offs = h_start + tl.arange(0, H_BLK)
        x_vals = tl.load(X_PTR + x_row_base + h_offs).to(tl.float32)

        fn_addrs = k_offs[:, None] * total_hidden + (group_base + h_offs)[None, :]
        fn_vals = tl.load(FN_PTR + fn_addrs, mask=k_mask[:, None], other=0.0)
        dout_fn = tl.sum(d_out[:, None] * fn_vals, axis=0)

        dx = inv_rms * dout_fn + rms_coeff * x_vals
        tl.store(DX_PTR + x_row_base + h_offs, dx)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    return 1 << (n - 1).bit_length()


def _pick_h_blk(rms_group_size: int, default: int = 512) -> int:
    """Pick H_BLK that evenly divides rms_group_size, capped at default."""
    return math.gcd(rms_group_size, default)


# ---------------------------------------------------------------------------
# fn transpose + bf16 cache
# ---------------------------------------------------------------------------


_fn_cache: dict[int, tuple[weakref.ref, torch.Tensor, torch.Tensor]] = {}


def _get_fn_t_bf16(fn: torch.Tensor, k_pad: int = _K_PAD) -> torch.Tensor:
    """Return fn pre-transposed and converted to bf16, padded to [total_hidden, k_pad].

    Cached keyed on fn.data_ptr(). The zero-padded layout eliminates masking
    in the MFMA kernel's K dimension.
    """
    key = fn.data_ptr()
    cached = _fn_cache.get(key)
    if cached is not None:
        ref, fn_t_bf16, fn_t_f32 = cached
        if ref() is fn:
            return fn_t_bf16
    mhc_mult3, total_hidden = fn.shape
    fn_t_bf16 = torch.zeros(total_hidden, k_pad, dtype=torch.bfloat16, device=fn.device)
    fn_t_bf16[:, :mhc_mult3] = fn.T.to(torch.bfloat16)
    fn_t_f32 = fn.T.contiguous()
    _fn_cache[key] = (weakref.ref(fn), fn_t_bf16, fn_t_f32)
    return fn_t_bf16


def _get_fn_transpose(fn: torch.Tensor) -> torch.Tensor:
    """Return fn.T.contiguous() fp32, caching the result."""
    key = fn.data_ptr()
    cached = _fn_cache.get(key)
    if cached is not None:
        ref, fn_t_bf16, fn_t_f32 = cached
        if ref() is fn:
            return fn_t_f32
    mhc_mult3, total_hidden = fn.shape
    fn_t_bf16 = torch.zeros(total_hidden, _K_PAD, dtype=torch.bfloat16, device=fn.device)
    fn_t_bf16[:, :mhc_mult3] = fn.T.to(torch.bfloat16)
    fn_t_f32 = fn.T.contiguous()
    _fn_cache[key] = (weakref.ref(fn), fn_t_bf16, fn_t_f32)
    return fn_t_f32


# ---------------------------------------------------------------------------
# FWD wrapper
# ---------------------------------------------------------------------------


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

    Default path uses bf16 MFMA with split-K parallelism.  Falls back to
    the legacy element-wise kernel when *use_fused_triton=True* or when
    n_rms_group > 1 (multi-group requires per-group dot products).
    """
    num_tokens = x.shape[0]
    total_hidden = x.shape[1]

    # Multi-group: fall back to legacy kernel or torch.mm hybrid
    if n_rms_group > 1:
        if use_fused_triton:
            if out is None:
                out = torch.empty((num_tokens, mhc_mult3), dtype=torch.float32, device=x.device)
            H_BLK = _pick_h_blk(rms_group_size)
            K_BLK = _next_power_of_2(mhc_mult3)
            _norm_fn_fwd_kernel_vec[(num_tokens,)](
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

        # torch.mm hybrid for multi-group
        x_f32 = x.float()
        x_grouped = x_f32.view(num_tokens, n_rms_group, rms_group_size)
        inv_rms = torch.rsqrt(x_grouped.square().sum(-1) / rms_group_size + eps)
        fn_grouped = fn.view(mhc_mult3, n_rms_group, rms_group_size)
        dot_grouped = torch.einsum('ngs,kgs->ngk', x_grouped, fn_grouped)
        result = (dot_grouped * inv_rms.unsqueeze(-1)).sum(1)
        if out is not None:
            out.copy_(result)
            return out
        return result

    if use_fused_triton:
        # Legacy element-wise kernel
        if out is None:
            out = torch.empty((num_tokens, mhc_mult3), dtype=torch.float32, device=x.device)
        H_BLK = _pick_h_blk(rms_group_size)
        K_BLK = _next_power_of_2(mhc_mult3)
        _norm_fn_fwd_kernel_vec[(num_tokens,)](
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

    # --- bf16 MFMA split-K path (default) ---
    K_PAD = _K_PAD
    T_BLK = 16
    H_BLK = 128
    N_SPLITS = 16
    NUM_WARPS = 2

    # Ensure total_hidden is divisible by N_SPLITS and H_BLK
    h_per_split = total_hidden // N_SPLITS
    while h_per_split % H_BLK != 0 and N_SPLITS > 1:
        N_SPLITS //= 2
        h_per_split = total_hidden // N_SPLITS

    fn_t_bf16 = _get_fn_t_bf16(fn, K_PAD)

    n_t_blocks = math.ceil(num_tokens / T_BLK)

    # Allocate partial results
    partial_out = torch.empty(N_SPLITS, num_tokens, K_PAD, dtype=torch.float32, device=x.device)
    partial_sqr = torch.empty(N_SPLITS, num_tokens, dtype=torch.float32, device=x.device)

    # Launch split-K GEMM
    grid_splitk = (n_t_blocks, N_SPLITS)
    _norm_fn_fwd_splitk[grid_splitk](
        x, fn_t_bf16,
        partial_out, partial_sqr,
        num_tokens, total_hidden,
        K_PAD=K_PAD,
        T_BLK=T_BLK,
        H_BLK=H_BLK,
        N_SPLITS=N_SPLITS,
        num_warps=NUM_WARPS,
        num_stages=2,
    )

    # Launch reduce + RMSNorm
    if out is None:
        out = torch.empty((num_tokens, mhc_mult3), dtype=torch.float32, device=x.device)

    grid_reduce = (n_t_blocks,)
    _norm_fn_fwd_reduce[grid_reduce](
        partial_out, partial_sqr,
        out,
        num_tokens, total_hidden,
        MHC_MULT3=mhc_mult3,
        K_PAD=K_PAD,
        T_BLK=T_BLK,
        N_SPLITS=N_SPLITS,
        EPS=eps,
        num_warps=NUM_WARPS,
    )

    return out


# ---------------------------------------------------------------------------
# BWD wrapper
# ---------------------------------------------------------------------------


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

    Default path uses bf16 MFMA for d_x computation.

    Returns:
        d_x:  [num_tokens, total_hidden] fp32
        d_fn: [mhc_mult3, total_hidden]  fp32
    """
    num_tokens = x.shape[0]
    total_hidden = x.shape[1]

    # Recompute inv_rms (needed by all paths for d_fn)
    x_f32 = x.float()
    x_grouped = x_f32.view(num_tokens, n_rms_group, rms_group_size)
    sqrsum = x_grouped.square().sum(-1)
    inv_rms = torch.rsqrt(sqrsum / rms_group_size + eps)

    if n_rms_group > 1 or use_fused_triton:
        if use_fused_triton and n_rms_group == 1:
            # Legacy Triton kernel
            d_x = torch.empty((num_tokens, total_hidden), dtype=torch.float32, device=x.device)
            H_BLK = _pick_h_blk(rms_group_size)
            K_BLK = _next_power_of_2(mhc_mult3)
            _norm_fn_bwd_kernel[(num_tokens, n_rms_group)](
                d_out, x, fn, d_x,
                num_tokens, total_hidden,
                MHC_MULT3=mhc_mult3,
                RMS_GROUP_SIZE=rms_group_size,
                N_RMS_GROUP=n_rms_group,
                H_BLK=H_BLK,
                K_BLK=K_BLK,
                EPS=eps,
            )
        else:
            # torch.mm hybrid for multi-group or fallback
            fn_grouped = fn.view(mhc_mult3, n_rms_group, rms_group_size)
            dot_grouped = torch.einsum('ngs,kgs->ngk', x_grouped, fn_grouped)
            c_g = torch.einsum('nk,ngk->ng', d_out, dot_grouped)
            dout_fn = torch.mm(d_out, fn).view(num_tokens, n_rms_group, rms_group_size)
            rms_coeff = -inv_rms.pow(3) * c_g / rms_group_size
            d_x = (
                inv_rms.unsqueeze(-1) * dout_fn
                + rms_coeff.unsqueeze(-1) * x_grouped
            ).view(num_tokens, total_hidden)
    else:
        # --- Single-group d_x: choose MFMA vs torch.mm based on token count ---
        # BWD MFMA kernel does 2 passes over hidden, so torch.mm is faster for
        # small batch sizes.  Crossover is around N=2048 on MI300.
        if num_tokens >= 2048:
            T_BLK = 16
            H_BLK = 128
            K_PAD = _K_PAD
            NUM_WARPS = 2

            fn_t_bf16 = _get_fn_t_bf16(fn, K_PAD)
            d_x = torch.empty((num_tokens, total_hidden), dtype=torch.float32, device=x.device)

            n_t_blocks = math.ceil(num_tokens / T_BLK)
            _norm_fn_bwd_dx_mfma[(n_t_blocks,)](
                d_out, x, fn_t_bf16, d_x,
                num_tokens, total_hidden,
                MHC_MULT3=mhc_mult3,
                K_PAD=K_PAD,
                T_BLK=T_BLK,
                H_BLK=H_BLK,
                EPS=eps,
                num_warps=NUM_WARPS,
            )
        else:
            # torch.mm hybrid for small batch (faster due to rocBLAS efficiency)
            fn_t_f32 = _get_fn_transpose(fn)
            dot_grouped = torch.mm(x_f32, fn_t_f32)  # [N, K] recompute dot
            c_g = (d_out * dot_grouped).sum(-1)  # [N]
            dout_fn = torch.mm(d_out, fn)  # [N, H]
            inv_rms_flat = inv_rms.view(num_tokens)  # [N] (n_rms_group=1)
            rms_coeff = -inv_rms_flat.pow(3) * c_g / rms_group_size  # [N]
            d_x = (
                inv_rms_flat.unsqueeze(-1) * dout_fn
                + rms_coeff.unsqueeze(-1) * x_f32
            )  # [N, H]

    # d_fn = d_out^T @ x_normed  (same for all paths)
    x_normed = (x_grouped * inv_rms.unsqueeze(-1)).view(num_tokens, total_hidden)
    d_fn = torch.mm(d_out.t(), x_normed)

    return d_x, d_fn


# ---------------------------------------------------------------------------
# Autograd wrapper
# ---------------------------------------------------------------------------


class MhcPreNormFnTritonFn(torch.autograd.Function):
    """Autograd wrapper for MFMA-accelerated mhc_pre_norm_fn FWD + BWD."""

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

        if mhc_norm_weight is not None:
            d_mhc_fn = d_fn * mhc_norm_weight
            d_norm_weight = (d_fn * mhc_fn).sum(0)
        else:
            d_mhc_fn = d_fn
            d_norm_weight = None

        return d_residual, d_mhc_fn, d_norm_weight, None


def mhc_pre_norm_fn_triton(
    residual: torch.Tensor,
    mhc_fn: torch.Tensor,
    mhc_norm_weight: torch.Tensor | None,
    mhc_norm_eps: float,
) -> torch.Tensor:
    """Triton MFMA-accelerated mhc_pre_norm_fn (drop-in replacement for mhc_pre_norm_fn_ref).

    Args:
        residual:        [n0, n1, mhc_mult, hidden_size] bf16
        mhc_fn:          [mhc_mult3, mhc_mult * hidden_size] fp32
        mhc_norm_weight: [mhc_mult * hidden_size] fp32 or None
        mhc_norm_eps:    float

    Returns:
        output: [n0, n1, mhc_mult3] fp32
    """
    return MhcPreNormFnTritonFn.apply(residual, mhc_fn, mhc_norm_weight, mhc_norm_eps)
