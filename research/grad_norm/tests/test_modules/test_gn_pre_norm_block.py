from unittest.mock import Mock
import torch.nn as nn
import pytest

from lizrd.core.llm import Residual
from research.grad_norm.modules.gn_pre_norm_block import GradMofiedPreNormBlock
from research.grad_norm.modules.grad_modif_placement import LayerGradModifPlacement


@pytest.mark.parametrize('gn_placement', LayerGradModifPlacement.all_placements())
def test_grad_modified_build(gn_placement: LayerGradModifPlacement):
    layer = Mock(spec=nn.Module, name="test_layer")
    grad_modif_layer = Mock(spec=nn.Module, name="test_grad_modif_layer")
    pre_norm = GradMofiedPreNormBlock(
        dmodel=16,
        layer=layer,
        name="test_layer",
        gn_placement=gn_placement,
        gn_layer=lambda: grad_modif_layer
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
