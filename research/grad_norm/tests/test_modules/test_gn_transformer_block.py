from unittest.mock import Mock, patch
import torch.nn as nn
import pytest
from research.grad_norm.modules.gn_transformer_block import GradModifiedTransformerBlock, BlockGradModifPlacement, GradMofiedPreNormBlock
from research.grad_norm.modules.grad_modif_placement import BlockGradModifPlacement


@pytest.fixture
def mock_layer():
    return Mock(spec=nn.Module, name="layer")


@patch("research.grad_norm.modules.gn_transformer_block.GradMofiedPreNormBlock", wraps=GradMofiedPreNormBlock)
def test_single_attention_block(mock_grad_modidied_pre_norm_block, mock_layer):
    gn_placement = BlockGradModifPlacement.create_full()
    block = GradModifiedTransformerBlock(
        dmodel=16,
        layers=[("attention", mock_layer)],
        gn_placement=gn_placement,
        grad_modif_fn=lambda: mock_layer
    )

    assert isinstance(block, GradModifiedTransformerBlock)
    assert len(block.block) == 1

    call_kwargs = mock_grad_modidied_pre_norm_block.call_args.kwargs
    assert call_kwargs['gn_placement'] is gn_placement.attn_mod


@patch("research.grad_norm.modules.gn_transformer_block.GradMofiedPreNormBlock", wraps=GradMofiedPreNormBlock)
def test_single_ff_block(mock_grad_modidied_pre_norm_block, mock_layer):
    gn_placement = BlockGradModifPlacement.create_full()
    block = GradModifiedTransformerBlock(
        dmodel=16,
        layers=[("feedforward", mock_layer)],
        gn_placement=gn_placement,
        grad_modif_fn=lambda: mock_layer
    )

    assert isinstance(block, GradModifiedTransformerBlock)
    assert len(block.block) == 1

    call_kwargs = mock_grad_modidied_pre_norm_block.call_args.kwargs
    assert call_kwargs['gn_placement'] is gn_placement.ff_mod


@patch("research.grad_norm.modules.gn_transformer_block.GradMofiedPreNormBlock", wraps=GradMofiedPreNormBlock)
def test_single_full_block(mock_grad_modidied_pre_norm_block, mock_layer):
    gn_placement = BlockGradModifPlacement.create_full()
    block = GradModifiedTransformerBlock(
        dmodel=16,
        layers=[
            ("attention", mock_layer),
            ("feedforward", mock_layer)
        ],
        gn_placement=gn_placement,
        grad_modif_fn=lambda: mock_layer
    )

    assert isinstance(block, GradModifiedTransformerBlock)
    assert len(block.block) == 2
    
    assert mock_grad_modidied_pre_norm_block.call_count == 2
    call_args_list = mock_grad_modidied_pre_norm_block.call_args_list
    assert call_args_list[0].kwargs['gn_placement'] is gn_placement.attn_mod
    assert call_args_list[1].kwargs['gn_placement'] is gn_placement.ff_mod
