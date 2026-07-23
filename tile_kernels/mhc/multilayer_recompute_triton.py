"""Triton-based multilayer recompute for MHC.

Decomposes the fused multi-layer kernel into a Python loop calling the
existing Triton ``pre_apply_mix_fwd_triton`` and ``mhc_post_fwd_triton``
per layer.  Functionally equivalent to the tilelang version in
``multilayer_recompute_kernel.py``.
"""

from __future__ import annotations

import torch

from .post_triton import mhc_post_fwd_triton
from .pre_apply_mix_triton import pre_apply_mix_fwd_triton


def mhc_multilayer_recompute_triton(
    initial_residual: torch.Tensor,
    pre_mix_list: list[torch.Tensor],
    layer_output_list: list[torch.Tensor],
    post_mix_list: list[torch.Tensor],
    comb_mix_list: list[torch.Tensor],
    layer_input_list: list[torch.Tensor],
    residual_list: list[torch.Tensor],
) -> None:
    """Multi-layer residual recompute using Triton sub-kernels.

    For each layer *i* in ``range(num_layers)``:

    1. **pre_apply_mix** –
       ``layer_input[i] = Σ_m pre_mix[i][m] * residual[m, h]``
    2. (model layer forward is external)
    3. **post** (if ``i < num_post``) –
       ``new_residual[m, h] = post_mix[i][m] * layer_output[i][h]
                              + Σ_k comb_mix[i][k, m] * residual[k, h]``

    Args:
        initial_residual: ``(bs, seq, mhc, H)`` bf16 – starting residual.
        pre_mix_list:     length ``L``      – each ``(bs, seq, mhc, 1)`` fp32.
        layer_output_list: length ``L_post`` – each ``(bs, seq, H)`` bf16.
        post_mix_list:    length ``L_post``  – each ``(bs, seq, mhc, 1)`` fp32.
        comb_mix_list:    length ``L_post``  – each ``(bs, seq, mhc, mhc)`` fp32.
        layer_input_list: length ``L``       – each ``(bs, seq, H)`` bf16 (output).
        residual_list:    length ``L_post``  – each ``(bs, seq, mhc, H)`` bf16 (output).
    """
    num_layers = len(pre_mix_list)
    num_post = len(layer_output_list)
    assert num_layers == len(layer_input_list)
    assert num_post == len(post_mix_list) == len(comb_mix_list) == len(residual_list)
    assert num_post == num_layers - 1 or num_post == num_layers, (
        f'post count ({num_post}) must be num_layers-1 or num_layers (num_layers={num_layers})'
    )
    assert num_layers > 0

    mhc_mult = initial_residual.shape[-2]
    hidden = initial_residual.shape[-1]

    # Flatten batch dims for the low-level Triton wrappers: (N, mhc, H)
    N = initial_residual[..., 0, 0].numel()  # product of all dims except last two
    residual = initial_residual.reshape(N, mhc_mult, hidden)  # current residual state

    for i in range(num_layers):
        # --- pre_apply_mix: layer_input = Σ_m pre_mix[m] * residual[m, h] ---
        mix = pre_mix_list[i].reshape(N, mhc_mult)  # (N, mhc) fp32
        li_out = layer_input_list[i].reshape(N, hidden)
        pre_apply_mix_fwd_triton(residual, mix, out=li_out)

        # --- post (if this layer has post data) ---
        if i < num_post:
            comb = comb_mix_list[i].reshape(N, mhc_mult, mhc_mult)  # (N, mhc, mhc) fp32
            post_mix = post_mix_list[i].reshape(N, mhc_mult)  # (N, mhc) fp32 – squeeze trailing 1
            lo = layer_output_list[i].reshape(N, hidden)  # (N, H) bf16
            res_out = residual_list[i].reshape(N, mhc_mult, hidden)

            mhc_post_fwd_triton(comb, residual, post_mix, lo, out=res_out)

            # The tilelang kernel casts new_residual to bf16 then back to fp32
            # between layers. Since res_out is already bf16, reading it back
            # into ``residual`` as the contiguous bf16 tensor achieves the same
            # rounding behaviour.
            residual = res_out
