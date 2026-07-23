"""Tests for the Triton multilayer_recompute implementation."""

import pytest
import torch

from tile_kernels.mhc.multilayer_recompute_triton import mhc_multilayer_recompute_triton
from tile_kernels.mhc.post_triton import mhc_post_triton
from tile_kernels.mhc.pre_apply_mix_triton import mhc_pre_apply_mix_triton


def _generate_test_data(
    bs: int,
    seq: int,
    mhc_mult: int,
    hidden: int,
    num_layers: int,
    num_post: int,
) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
]:
    initial_residual = torch.randn(bs, seq, mhc_mult, hidden, device='cuda', dtype=torch.bfloat16)
    pre_mix_list = [torch.randn(bs, seq, mhc_mult, 1, device='cuda', dtype=torch.float32) for _ in range(num_layers)]
    layer_output_list = [torch.randn(bs, seq, hidden, device='cuda', dtype=torch.bfloat16) for _ in range(num_post)]
    post_mix_list = [torch.randn(bs, seq, mhc_mult, 1, device='cuda', dtype=torch.float32) for _ in range(num_post)]
    comb_mix_list = [torch.randn(bs, seq, mhc_mult, mhc_mult, device='cuda', dtype=torch.float32) for _ in range(num_post)]
    layer_input_list = [torch.empty(bs, seq, hidden, device='cuda', dtype=torch.bfloat16) for _ in range(num_layers)]
    residual_list = [torch.empty(bs, seq, mhc_mult, hidden, device='cuda', dtype=torch.bfloat16) for _ in range(num_post)]
    return initial_residual, pre_mix_list, layer_output_list, post_mix_list, comb_mix_list, layer_input_list, residual_list


def _multilayer_recompute_ref(
    initial_residual: torch.Tensor,
    pre_mix_list: list[torch.Tensor],
    layer_output_list: list[torch.Tensor],
    post_mix_list: list[torch.Tensor],
    comb_mix_list: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Reference implementation: Python loop over layers using the Triton ops."""
    layer_input_refs: list[torch.Tensor] = []
    residual_refs: list[torch.Tensor] = []
    residual = initial_residual
    for i in range(len(pre_mix_list)):
        layer_input = mhc_pre_apply_mix_triton(residual, pre_mix_list[i])
        layer_input_refs.append(layer_input)
        if i < len(layer_output_list):
            residual = mhc_post_triton(layer_output_list[i], residual, post_mix_list[i], comb_mix_list[i])
            residual_refs.append(residual)
    return layer_input_refs, residual_refs


_CORRECTNESS_CASES = [
    (1, 1, 2560),
    (3, 2, 2560),
    (3, 3, 2560),
    (10, 9, 2560),
    (10, 10, 2560),
    (10, 9, 4096),
    (10, 10, 4096),
    (10, 9, 7168),
    (10, 10, 7168),
    (10, 9, 8192),
    (10, 10, 8192),
]


@pytest.mark.parametrize('num_layers,num_post,hidden', _CORRECTNESS_CASES)
def test_mhc_multilayer_recompute_triton_correctness(num_layers: int, num_post: int, hidden: int) -> None:
    """Triton multilayer_recompute must match the reference implementation exactly."""
    torch.manual_seed(0)
    initial_residual, pre_mix_list, layer_output_list, post_mix_list, comb_mix_list, layer_input_list, residual_list = (
        _generate_test_data(1, 8192, 4, hidden, num_layers, num_post)
    )

    layer_input_ref, residual_ref = _multilayer_recompute_ref(
        initial_residual, pre_mix_list, layer_output_list, post_mix_list, comb_mix_list
    )

    mhc_multilayer_recompute_triton(
        initial_residual, pre_mix_list, layer_output_list, post_mix_list, comb_mix_list,
        layer_input_list, residual_list,
    )

    for i in range(num_layers):
        assert torch.equal(layer_input_list[i], layer_input_ref[i]), (
            f'layer_input[{i}] mismatch! max diff = '
            f'{(layer_input_list[i].float() - layer_input_ref[i].float()).abs().max().item()}'
        )
    for i in range(num_post):
        assert torch.equal(residual_list[i], residual_ref[i]), (
            f'residual[{i}] mismatch! max diff = '
            f'{(residual_list[i].float() - residual_ref[i].float()).abs().max().item()}'
        )


@pytest.mark.parametrize('num_layers,num_post', [(1, 1), (2, 1), (2, 2)])
def test_mhc_multilayer_recompute_triton_small(num_layers: int, num_post: int) -> None:
    """Smoke test with small dimensions."""
    torch.manual_seed(42)
    initial_residual, pre_mix_list, layer_output_list, post_mix_list, comb_mix_list, layer_input_list, residual_list = (
        _generate_test_data(1, 16, 4, 512, num_layers, num_post)
    )

    layer_input_ref, residual_ref = _multilayer_recompute_ref(
        initial_residual, pre_mix_list, layer_output_list, post_mix_list, comb_mix_list
    )

    mhc_multilayer_recompute_triton(
        initial_residual, pre_mix_list, layer_output_list, post_mix_list, comb_mix_list,
        layer_input_list, residual_list,
    )

    for i in range(num_layers):
        assert torch.equal(layer_input_list[i], layer_input_ref[i]), (
            f'layer_input[{i}] mismatch! max diff = '
            f'{(layer_input_list[i].float() - layer_input_ref[i].float()).abs().max().item()}'
        )
    for i in range(num_post):
        assert torch.equal(residual_list[i], residual_ref[i]), (
            f'residual[{i}] mismatch! max diff = '
            f'{(residual_list[i].float() - residual_ref[i].float()).abs().max().item()}'
        )


@pytest.mark.parametrize('bs,seq', [(1, 128), (2, 64)])
def test_mhc_multilayer_recompute_triton_batch_shapes(bs: int, seq: int) -> None:
    """Test with various batch / sequence dimensions."""
    torch.manual_seed(7)
    num_layers, num_post, hidden = 3, 2, 2560
    initial_residual, pre_mix_list, layer_output_list, post_mix_list, comb_mix_list, layer_input_list, residual_list = (
        _generate_test_data(bs, seq, 4, hidden, num_layers, num_post)
    )

    layer_input_ref, residual_ref = _multilayer_recompute_ref(
        initial_residual, pre_mix_list, layer_output_list, post_mix_list, comb_mix_list
    )

    mhc_multilayer_recompute_triton(
        initial_residual, pre_mix_list, layer_output_list, post_mix_list, comb_mix_list,
        layer_input_list, residual_list,
    )

    for i in range(num_layers):
        assert torch.equal(layer_input_list[i], layer_input_ref[i]), (
            f'layer_input[{i}] mismatch! max diff = '
            f'{(layer_input_list[i].float() - layer_input_ref[i].float()).abs().max().item()}'
        )
    for i in range(num_post):
        assert torch.equal(residual_list[i], residual_ref[i]), (
            f'residual[{i}] mismatch! max diff = '
            f'{(residual_list[i].float() - residual_ref[i].float()).abs().max().item()}'
        )
