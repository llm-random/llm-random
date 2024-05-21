from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from lizrd.core.llm import Residual
from research.grad_norm.modules.gn_pre_norm_block import GradMofiedPreNormBlock
from research.grad_norm.modules.grad_modif_placement import LayerGradModifPlacement
from research.grad_norm.tests.test_modules.utils import TorchIdModule


@pytest.mark.parametrize("gn_placement", LayerGradModifPlacement.all_placements())
def test_grad_modified_build(gn_placement: LayerGradModifPlacement):
    layer = Mock(spec=nn.Module, name="test_layer")
    grad_modif_layer = Mock(spec=nn.Module, name="test_grad_modif_layer")
    pre_norm = GradMofiedPreNormBlock(
        dmodel=16,
        layer=layer,
        name="test_layer",
        gn_placement=gn_placement,
        gn_layer=lambda: grad_modif_layer,
    )

    residual = pre_norm[0]
    assert isinstance(pre_norm, nn.Sequential)
    assert isinstance(residual, Residual)

    assert hasattr(residual.layer, "post_norm_gn") == gn_placement.post_norm
    if gn_placement.post_norm:
        assert residual.layer.post_norm_gn == grad_modif_layer
    assert hasattr(residual.layer, "post_layer_gn") == gn_placement.post_layer
    if gn_placement.post_layer:
        assert residual.layer.post_layer_gn == grad_modif_layer

    assert (len(pre_norm) == 2) == gn_placement.post_add
    if gn_placement.post_add:
        assert pre_norm[1] == grad_modif_layer


@pytest.mark.parametrize("gn_placement", LayerGradModifPlacement.all_placements())
def test_grad_modified_pre_norm_block_ff(gn_placement: LayerGradModifPlacement):
    layer = Mock(spec=nn.Module, name="test_layer", wraps=TorchIdModule())
    norm_class_factory = Mock()
    norm_class_factory.side_effect = lambda x: Mock(
        spec=nn.Module, name="test_norm_class_factory", wraps=TorchIdModule()
    )

    gn_layer_factory = Mock()
    gn_layer_factory.side_effect = lambda: Mock(spec=nn.Module, name="test_gn_layer_factory", wraps=TorchIdModule())
    pre_norm = GradMofiedPreNormBlock(
        dmodel=16,
        layer=layer,
        norm_class=norm_class_factory,
        name="test_layer",
        gn_placement=gn_placement,
        gn_layer=gn_layer_factory,
    )

    x = torch.rand(1, 16)
    assert torch.allclose(pre_norm.forward(x), x * 2)

    assert layer.call_count == 1
    assert norm_class_factory.call_count == 1
    assert gn_layer_factory.call_count == int(gn_placement.post_norm) + int(gn_placement.post_layer) + int(
        gn_placement.post_add
    )
