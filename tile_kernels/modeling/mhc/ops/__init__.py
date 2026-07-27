"""MHC ops with backend dispatch between tilelang and triton.

Set the backend via:
    - Environment variable: MHC_BACKEND=triton (or tilelang)
    - Python API: from tile_kernels.modeling.mhc.ops.backend import set_backend; set_backend('triton')

The default backend is 'tilelang' for backward compatibility.
"""

from .backend import get_backend, set_backend, use_triton

if use_triton():
    # --- sinkhorn: signature compatible ---
    from tile_kernels.mhc.sinkhorn_triton import sinkhorn_normalize_triton as sinkhorn_normalize

    # --- pre_split_mixes: signature compatible ---
    from tile_kernels.mhc.pre_split_mixes_triton import mhc_pre_split_mixes_triton as mhc_pre_split_mixes

    # --- pre_big_fuse: signature compatible ---
    from tile_kernels.mhc.pre_big_fuse_triton import mhc_pre_big_fuse_triton as mhc_pre_big_fuse

    # --- multilayer_recompute: signature compatible ---
    from tile_kernels.mhc.multilayer_recompute_triton import (
        mhc_multilayer_recompute_triton as mhc_multilayer_recompute,
    )

    # --- post: triton version lacks `out` parameter; needs adapters ---
    from tile_kernels.mhc.post_triton import (
        mhc_post_bwd_triton as _mhc_post_bwd_triton_raw,
        mhc_post_fwd_triton as _mhc_post_fwd_triton_raw,
        mhc_post_triton as _mhc_post_triton_raw,
    )

    import torch

    def mhc_post(
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # triton autograd wrapper ignores `out`; result is always freshly allocated
        return _mhc_post_triton_raw(x, residual, post_layer_mix, comb_res_mix)

    def mhc_post_fwd(
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N = comb_res_mix.shape[0] * comb_res_mix.shape[1]
        mhc = residual.shape[-2]
        hidden = residual.shape[-1]
        result = _mhc_post_fwd_triton_raw(
            comb_res_mix.reshape(N, mhc, mhc),
            residual.reshape(N, mhc, hidden),
            post_layer_mix.reshape(N, mhc),
            x.reshape(N, hidden),
            out=out.reshape(N, mhc, hidden) if out is not None else None,
        )
        return result.view_as(residual)

    def mhc_post_bwd(
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        d_o: torch.Tensor,
        fuse_grad_acc: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        N = d_o.shape[0] * d_o.shape[1]
        mhc = d_o.shape[-2]
        hidden = d_o.shape[-1]
        da, db, dc, dd = _mhc_post_bwd_triton_raw(
            d_o.reshape(N, mhc, hidden),
            comb_res_mix.reshape(N, mhc, mhc),
            residual.reshape(N, mhc, hidden),
            post_layer_mix.reshape(N, mhc),
            x.reshape(N, hidden),
        )
        outer_shape = d_o.shape[:-2]
        return (
            dd.view(*outer_shape, hidden),
            db.view(*outer_shape, mhc, hidden),
            dc.view(*outer_shape, mhc, 1),
            da.view(*outer_shape, mhc, mhc),
        )

    # --- pre_apply_mix: triton version lacks `out` parameter ---
    from tile_kernels.mhc.pre_apply_mix_triton import mhc_pre_apply_mix_triton as _mhc_pre_apply_mix_triton_raw

    def mhc_pre_apply_mix(
        x: torch.Tensor,
        mix: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # triton autograd wrapper ignores `out`; result is always freshly allocated
        return _mhc_pre_apply_mix_triton_raw(x, mix)

    # --- norm_fn: triton version lacks `fuse_grad_acc` and `n_splits` ---
    from tile_kernels.mhc.norm_fn_triton import mhc_pre_norm_fn_triton as _mhc_pre_norm_fn_triton_raw

    def mhc_pre_norm_fn(
        residual: torch.Tensor,
        mhc_fn: torch.Tensor,
        mhc_norm_weight: torch.Tensor | None,
        mhc_norm_eps: float,
        fuse_grad_acc: bool = True,
        n_splits: int = 16,
    ) -> torch.Tensor:
        # triton version handles these internally; extra params are silently ignored
        return _mhc_pre_norm_fn_triton_raw(residual, mhc_fn, mhc_norm_weight, mhc_norm_eps)

    # --- expand_to_mhc: pure torch fallback (no triton/tilelang needed) ---
    def expand_to_mhc(hidden, mhc_mult, out=None):
        return hidden.unsqueeze(-2).expand(*hidden.shape[:-1], mhc_mult, hidden.shape[-1]).contiguous()

    # --- head_compute_mix: pure torch fallback ---
    def mhc_head_compute_mix(input_mix, mhc_scale, mhc_base, mhc_pre_eps):
        import torch
        return torch.sigmoid(input_mix * mhc_scale + mhc_base) + mhc_pre_eps

else:
    from .expand import expand_to_mhc
    from .head_compute_mix import mhc_head_compute_mix
    from .multilayer_recompute import mhc_multilayer_recompute
    from .norm_fn import mhc_pre_norm_fn
    from .post import mhc_post, mhc_post_bwd, mhc_post_fwd
    from .pre_apply_mix import mhc_pre_apply_mix
    from .pre_big_fuse import mhc_pre_big_fuse
    from .pre_split_mixes import mhc_pre_split_mixes
    from .sinkhorn import sinkhorn_normalize
