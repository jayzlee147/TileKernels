from tile_kernels.modeling.mhc.ops.post import mhc_post
from tile_kernels.modeling.mhc.ops.pre_apply_mix import mhc_pre_apply_mix


def mhc_multilayer_recompute(
    initial_residual,
    pre_mix_list,
    layer_output_list,
    post_mix_list,
    comb_mix_list,
    layer_input_list,
    residual_list,
):
    num_layers = len(pre_mix_list)
    num_post = len(layer_output_list)

    residual = initial_residual
    for i in range(num_layers):
        mhc_pre_apply_mix(residual, pre_mix_list[i], layer_input_list[i])
        if i < num_post:
            residual = mhc_post(
                layer_output_list[i], residual,
                post_mix_list[i], comb_mix_list[i],
                residual_list[i],
            )
