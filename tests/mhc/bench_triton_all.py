#!/usr/bin/env python3
"""Benchmark all MHC Triton operators: FWD and BWD with block-size tuning.

Usage:
    python tests/mhc/bench_triton_all.py

Measures wall-clock GPU time using torch.cuda.Event for:
  1. sinkhorn_triton      – sinkhorn_normalize_triton
  2. post_triton           – mhc_post_triton
  3. pre_apply_mix_triton  – mhc_pre_apply_mix_triton
  4. pre_split_mixes_triton– mhc_pre_split_mixes_triton
  5. norm_fn_triton        – mhc_pre_norm_fn_triton

For post_triton and pre_apply_mix_triton, H_BLK is swept over candidate
values to find the best tile size.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Callable

import torch

# ---------------------------------------------------------------------------
# Module loading helpers
# ---------------------------------------------------------------------------

_MODULE_DIR = Path(__file__).resolve().parents[2] / "tile_kernels" / "mhc"


def _load_module(name: str):
    """Load a module from tile_kernels/mhc/ by file name using importlib.util."""
    path = _MODULE_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Pre-load all modules
sinkhorn_mod = _load_module("sinkhorn_triton")
post_mod = _load_module("post_triton")
pre_apply_mix_mod = _load_module("pre_apply_mix_triton")
pre_split_mixes_mod = _load_module("pre_split_mixes_triton")
norm_fn_mod = _load_module("norm_fn_triton")

# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------

WARMUP = 10
BENCH_ITERS = 100


def bench_fn(fn: Callable, *, warmup: int = WARMUP, iters: int = BENCH_ITERS) -> float:
    """Benchmark *fn()* on the current CUDA device.

    Returns median GPU time in milliseconds.
    """
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]  # median


# ===================================================================
# Benchmark definitions
# ===================================================================

N = 4096
MHC = 4
H = 4096
DEVICE = "cuda"


# -------------------------------------------------------------------
# 1. Sinkhorn
# -------------------------------------------------------------------


def bench_sinkhorn():
    """FWD and BWD for sinkhorn_normalize_triton."""
    x = torch.randn(N, MHC, MHC, device=DEVICE, dtype=torch.float32, requires_grad=True)

    sinkhorn_normalize_triton = sinkhorn_mod.sinkhorn_normalize_triton

    def fwd():
        return sinkhorn_normalize_triton(x, repeat=10)

    fwd_ms = bench_fn(fwd)

    # BWD: run fwd then backward
    def fwd_bwd():
        y = sinkhorn_normalize_triton(x, repeat=10)
        y.sum().backward(retain_graph=False)
        if x.grad is not None:
            x.grad = None

    bwd_ms = bench_fn(fwd_bwd)

    return [
        {"op": "sinkhorn", "pass": "FWD", "H_BLK": "-", "ms": f"{fwd_ms:.4f}"},
        {"op": "sinkhorn", "pass": "FWD+BWD", "H_BLK": "-", "ms": f"{bwd_ms:.4f}"},
    ]


# -------------------------------------------------------------------
# 2. post_triton – sweep H_BLK
# -------------------------------------------------------------------


def _post_fwd_with_hblk(h_blk: int):
    """Build a closure that calls the low-level FWD kernel with a specific H_BLK."""
    comb_res_mix = torch.randn(N, MHC, MHC, device=DEVICE, dtype=torch.float32).contiguous()
    residual = torch.randn(N, MHC, H, device=DEVICE, dtype=torch.bfloat16).contiguous()
    post_layer_mix = torch.randn(N, MHC, device=DEVICE, dtype=torch.float32).contiguous()
    x_input = torch.randn(N, H, device=DEVICE, dtype=torch.bfloat16).contiguous()
    out = torch.empty(N, MHC, H, device=DEVICE, dtype=torch.bfloat16)

    kernel = post_mod._mhc_post_fwd_kernel
    grid = (N,)

    def run():
        kernel[grid](
            comb_res_mix, residual, post_layer_mix, x_input, out,
            N, H=H, H_BLK=h_blk,
        )

    return run


def _post_bwd_with_hblk(h_blk: int):
    """Build a closure that calls the low-level BWD kernel with a specific H_BLK."""
    dx = torch.randn(N, MHC, H, device=DEVICE, dtype=torch.bfloat16).contiguous()
    comb_res_mix = torch.randn(N, MHC, MHC, device=DEVICE, dtype=torch.float32).contiguous()
    residual = torch.randn(N, MHC, H, device=DEVICE, dtype=torch.bfloat16).contiguous()
    post_layer_mix = torch.randn(N, MHC, device=DEVICE, dtype=torch.float32).contiguous()
    x_input = torch.randn(N, H, device=DEVICE, dtype=torch.bfloat16).contiguous()
    da = torch.empty(N, MHC, MHC, device=DEVICE, dtype=torch.float32)
    db = torch.empty(N, MHC, H, device=DEVICE, dtype=torch.bfloat16)
    dc = torch.empty(N, MHC, device=DEVICE, dtype=torch.float32)
    dd = torch.empty(N, H, device=DEVICE, dtype=torch.bfloat16)

    kernel = post_mod._mhc_post_bwd_kernel
    grid = (N,)

    def run():
        kernel[grid](
            dx, comb_res_mix, residual, post_layer_mix, x_input,
            da, db, dc, dd,
            N, H=H, H_BLK=h_blk,
        )

    return run


def _h_blk_candidates(H: int) -> list[int]:
    """Generate candidate H_BLK values that evenly divide H (powers of 2)."""
    candidates = []
    v = 1
    while v <= H:
        if H % v == 0:
            candidates.append(v)
        v *= 2
    return candidates


def bench_post():
    """FWD and BWD for mhc_post_triton, sweeping H_BLK."""
    rows: list[dict] = []

    # --- Default autograd path (for comparison) ---
    x = torch.randn(1, N, H, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    residual = torch.randn(1, N, MHC, H, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    plm = torch.randn(1, N, MHC, 1, device=DEVICE, dtype=torch.float32, requires_grad=True)
    crm = torch.randn(1, N, MHC, MHC, device=DEVICE, dtype=torch.float32, requires_grad=True)

    mhc_post_triton = post_mod.mhc_post_triton

    def fwd_default():
        return mhc_post_triton(x, residual, plm, crm)

    fwd_def_ms = bench_fn(fwd_default)
    rows.append({"op": "post", "pass": "FWD(autograd)", "H_BLK": "default", "ms": f"{fwd_def_ms:.4f}"})

    def fwd_bwd_default():
        y = mhc_post_triton(x, residual, plm, crm)
        y.sum().backward(retain_graph=False)
        for t in (x, residual, plm, crm):
            t.grad = None

    fwd_bwd_def_ms = bench_fn(fwd_bwd_default)
    rows.append({"op": "post", "pass": "FWD+BWD(autograd)", "H_BLK": "default", "ms": f"{fwd_bwd_def_ms:.4f}"})

    # --- Sweep H_BLK on the raw kernels ---
    candidates = _h_blk_candidates(H)
    # Only test reasonable block sizes (skip very small and very large)
    candidates = [c for c in candidates if 64 <= c <= H]
    if not candidates:
        candidates = [math.gcd(H, 512)]

    for hb in candidates:
        fwd_fn = _post_fwd_with_hblk(hb)
        ms = bench_fn(fwd_fn)
        rows.append({"op": "post", "pass": "FWD(kernel)", "H_BLK": str(hb), "ms": f"{ms:.4f}"})

    for hb in candidates:
        bwd_fn = _post_bwd_with_hblk(hb)
        ms = bench_fn(bwd_fn)
        rows.append({"op": "post", "pass": "BWD(kernel)", "H_BLK": str(hb), "ms": f"{ms:.4f}"})

    return rows


# -------------------------------------------------------------------
# 3. pre_apply_mix_triton – sweep H_BLK
# -------------------------------------------------------------------


def _pre_apply_mix_fwd_with_hblk(h_blk: int):
    x = torch.randn(N, MHC, H, device=DEVICE, dtype=torch.bfloat16).contiguous()
    mix = torch.randn(N, MHC, device=DEVICE, dtype=torch.float32).contiguous()
    out = torch.empty(N, H, device=DEVICE, dtype=torch.bfloat16)

    kernel = pre_apply_mix_mod._pre_apply_mix_fwd_kernel
    grid = (N,)

    def run():
        kernel[grid](x, mix, out, N, H=H, H_BLK=h_blk)

    return run


def _pre_apply_mix_bwd_with_hblk(h_blk: int):
    o_grad = torch.randn(N, H, device=DEVICE, dtype=torch.bfloat16).contiguous()
    x = torch.randn(N, MHC, H, device=DEVICE, dtype=torch.bfloat16).contiguous()
    mix = torch.randn(N, MHC, device=DEVICE, dtype=torch.float32).contiguous()
    x_grad = torch.zeros(N, MHC, H, device=DEVICE, dtype=torch.bfloat16).contiguous()
    mix_grad = torch.empty(N, MHC, device=DEVICE, dtype=torch.float32)

    kernel = pre_apply_mix_mod._pre_apply_mix_bwd_kernel
    grid = (N,)

    def run():
        kernel[grid](o_grad, x, mix, x_grad, mix_grad, N, H=H, H_BLK=h_blk)

    return run


def bench_pre_apply_mix():
    """FWD and BWD for mhc_pre_apply_mix_triton, sweeping H_BLK."""
    rows: list[dict] = []

    # --- Default autograd path ---
    x = torch.randn(1, N, MHC, H, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    mix = torch.randn(1, N, MHC, 1, device=DEVICE, dtype=torch.float32, requires_grad=True)

    mhc_pre_apply_mix_triton = pre_apply_mix_mod.mhc_pre_apply_mix_triton

    def fwd_default():
        return mhc_pre_apply_mix_triton(x, mix)

    fwd_def_ms = bench_fn(fwd_default)
    rows.append({"op": "pre_apply_mix", "pass": "FWD(autograd)", "H_BLK": "default", "ms": f"{fwd_def_ms:.4f}"})

    def fwd_bwd_default():
        y = mhc_pre_apply_mix_triton(x, mix)
        y.sum().backward(retain_graph=False)
        x.grad = None
        mix.grad = None

    fwd_bwd_def_ms = bench_fn(fwd_bwd_default)
    rows.append({"op": "pre_apply_mix", "pass": "FWD+BWD(autograd)", "H_BLK": "default", "ms": f"{fwd_bwd_def_ms:.4f}"})

    # --- Sweep H_BLK on raw kernels ---
    candidates = _h_blk_candidates(H)
    candidates = [c for c in candidates if 64 <= c <= H]
    if not candidates:
        candidates = [math.gcd(H, 512)]

    for hb in candidates:
        fwd_fn = _pre_apply_mix_fwd_with_hblk(hb)
        ms = bench_fn(fwd_fn)
        rows.append({"op": "pre_apply_mix", "pass": "FWD(kernel)", "H_BLK": str(hb), "ms": f"{ms:.4f}"})

    for hb in candidates:
        bwd_fn = _pre_apply_mix_bwd_with_hblk(hb)
        ms = bench_fn(bwd_fn)
        rows.append({"op": "pre_apply_mix", "pass": "BWD(kernel)", "H_BLK": str(hb), "ms": f"{ms:.4f}"})

    return rows


# -------------------------------------------------------------------
# 4. pre_split_mixes_triton
# -------------------------------------------------------------------


def bench_pre_split_mixes():
    """FWD and BWD for mhc_pre_split_mixes_triton."""
    K = MHC
    KK_TOTAL = (2 + K) * K  # = 24

    inp = torch.randn(1, N, KK_TOTAL, device=DEVICE, dtype=torch.float32, requires_grad=True)
    scale = torch.randn(3, device=DEVICE, dtype=torch.float32, requires_grad=True)
    base = torch.randn(KK_TOTAL, device=DEVICE, dtype=torch.float32, requires_grad=True)
    mhc_post_mult_value = 2.0
    mhc_pre_eps = 1e-6

    mhc_pre_split_mixes_triton = pre_split_mixes_mod.mhc_pre_split_mixes_triton

    def fwd():
        return mhc_pre_split_mixes_triton(inp, scale, base, K, mhc_post_mult_value, mhc_pre_eps)

    fwd_ms = bench_fn(fwd)

    def fwd_bwd():
        pre, post, comb = mhc_pre_split_mixes_triton(inp, scale, base, K, mhc_post_mult_value, mhc_pre_eps)
        loss = pre.sum() + post.sum() + comb.sum()
        loss.backward(retain_graph=False)
        for t in (inp, scale, base):
            t.grad = None

    bwd_ms = bench_fn(fwd_bwd)

    return [
        {"op": "pre_split_mixes", "pass": "FWD", "H_BLK": "-", "ms": f"{fwd_ms:.4f}"},
        {"op": "pre_split_mixes", "pass": "FWD+BWD", "H_BLK": "-", "ms": f"{bwd_ms:.4f}"},
    ]


# -------------------------------------------------------------------
# 5. norm_fn_triton
# -------------------------------------------------------------------


def bench_norm_fn():
    """FWD and BWD for mhc_pre_norm_fn_triton."""
    mhc_mult3 = (2 + MHC) * MHC  # 24
    total_hidden = MHC * H        # 16384
    eps = 1e-6

    residual = torch.randn(1, N, MHC, H, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    fn = torch.randn(mhc_mult3, total_hidden, device=DEVICE, dtype=torch.float32, requires_grad=True)

    mhc_pre_norm_fn_triton = norm_fn_mod.mhc_pre_norm_fn_triton

    def fwd():
        return mhc_pre_norm_fn_triton(residual, fn, None, eps)

    fwd_ms = bench_fn(fwd)

    def fwd_bwd():
        y = mhc_pre_norm_fn_triton(residual, fn, None, eps)
        y.sum().backward(retain_graph=False)
        residual.grad = None
        fn.grad = None

    bwd_ms = bench_fn(fwd_bwd)

    return [
        {"op": "norm_fn", "pass": "FWD", "H_BLK": "-", "ms": f"{fwd_ms:.4f}"},
        {"op": "norm_fn", "pass": "FWD+BWD", "H_BLK": "-", "ms": f"{bwd_ms:.4f}"},
    ]


# ===================================================================
# Table formatter
# ===================================================================


def print_table(rows: list[dict[str, str]]):
    """Print a list of dicts as a formatted ASCII table."""
    if not rows:
        return
    cols = list(rows[0].keys())
    # Compute column widths
    widths = {c: max(len(c), *(len(r[c]) for r in rows)) for c in cols}
    sep = "+-" + "-+-".join("-" * widths[c] for c in cols) + "-+"
    header = "| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |"

    print(sep)
    print(header)
    print(sep)
    for r in rows:
        line = "| " + " | ".join(r[c].ljust(widths[c]) for c in cols) + " |"
        print(line)
    print(sep)


# ===================================================================
# Main
# ===================================================================


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        sys.exit(1)

    torch.cuda.set_device(0)
    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")
    print(f"Parameters: N={N}, MHC={MHC}, H={H}")
    print(f"Warmup={WARMUP}, Bench iterations={BENCH_ITERS}")
    print()

    all_rows: list[dict[str, str]] = []

    benchmarks = [
        ("sinkhorn_triton", bench_sinkhorn),
        ("post_triton", bench_post),
        ("pre_apply_mix_triton", bench_pre_apply_mix),
        ("pre_split_mixes_triton", bench_pre_split_mixes),
        ("norm_fn_triton", bench_norm_fn),
    ]

    for name, bench in benchmarks:
        print(f">>> Benchmarking {name} ...")
        try:
            rows = bench()
            all_rows.extend(rows)
            # Print per-operator mini-table immediately
            print_table(rows)
            print()
        except Exception as exc:
            print(f"  FAILED: {exc}")
            import traceback
            traceback.print_exc()
            print()

    # Final combined table
    print("=" * 72)
    print("Combined results (median GPU ms, lower is better)")
    print("=" * 72)
    print_table(all_rows)


if __name__ == "__main__":
    main()
