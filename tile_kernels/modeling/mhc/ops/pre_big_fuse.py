import torch

from tile_kernels.modeling.mhc.ops.norm_fn import mhc_pre_norm_fn
from tile_kernels.modeling.mhc.ops.pre_apply_mix import mhc_pre_apply_mix
from tile_kernels.modeling.mhc.ops.pre_split_mixes import mhc_pre_split_mixes
from tile_kernels.modeling.mhc.ops.sinkhorn import sinkhorn_normalize


def mhc_pre_big_fuse(
    residual: torch.Tensor,
    fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mhc_mult = residual.shape[-2]

    mixes = mhc_pre_norm_fn(
        residual, fn, None, rms_eps,
        fuse_grad_acc=False, n_splits=n_splits,
    )

    pre_mix, post_mix, comb_mix = mhc_pre_split_mixes(
        mixes, mhc_scale, mhc_base,
        mhc_mult, mhc_post_mult_value, mhc_pre_eps,
    )

    comb_mix = sinkhorn_normalize(comb_mix, repeat=sinkhorn_repeat, eps=mhc_sinkhorn_eps)

    layer_input = mhc_pre_apply_mix(residual, pre_mix)

    return post_mix, comb_mix, layer_input
