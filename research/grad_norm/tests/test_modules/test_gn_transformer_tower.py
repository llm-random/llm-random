import torch.nn as nn
import torch
from unittest.mock import Mock, patch

from research.grad_norm.tests.test_modules.utils import TorchIdModule

from research.grad_norm.modules.gn_transformer_tower import GradModiedTransformerTower
from research.grad_norm.modules.grad_modif_placement import BlockGradModifPlacement


@patch(
    "research.grad_norm.modules.gn_transformer_block.GradMofiedPreNormBlock",
    new=Mock(
        name="GradMofiedPreNormBlock",
        side_effect=lambda *args, **kwargs: Mock(spec=nn.Module, wraps=TorchIdModule()),
    ),
)
def test_transformer_tower_basic_build():
    grad_modif_fn = Mock(name="grad_modif_fn")
    grad_modif_fn.side_effect = lambda: Mock(spec=nn.Module, wraps=TorchIdModule())
    layer_dict = {
        "attention": Mock(
            return_value=Mock(spec=nn.Module, name="attention", wraps=TorchIdModule())
        ),
        "feedforward": Mock(
            return_value=Mock(spec=nn.Module, name="feedforward", wraps=TorchIdModule())
        ),
    }
    gn_tt = GradModiedTransformerTower(
        device=torch.device("cpu"),
        n_blocks=2,
        dmodel=16,
        layer_dict=layer_dict,
        gn_placement=BlockGradModifPlacement.create_full(),
        grad_modif_fn=grad_modif_fn,
    )

    x = torch.rand(1, 16)
    assert torch.allclose(gn_tt(x), x)
