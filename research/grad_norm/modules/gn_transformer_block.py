import torch.nn as nn

from typing import Callable, Optional
from functools import partial
from collections import OrderedDict

from research.grad_norm.modules.gn_pre_norm_block import GradMofiedPreNormBlock
from research.grad_norm.modules.grad_modif_placement import BlockGradModifPlacement


class GradModifiedTransformerBlock(nn.Module):
    def __init__(
        self,
        dmodel,
        layers,
        gn_placement: BlockGradModifPlacement,
        grad_modif_fn: Callable[[], nn.Module],
        norm_class: Optional[nn.Module] = None,
    ):
        super(GradModifiedTransformerBlock, self).__init__()

        residual_layers = []

        for name, layer in layers:
            if name == "attention":
                layer_gn_placement = gn_placement.attn_mod
            elif name == "feedforward":
                layer_gn_placement = gn_placement.ff_mod
            else:
                raise ValueError(
                    "Supprted layer types are 'attention' and 'feedforward'"
                )

            norm_class_kwargs = {}
            if norm_class is not None:
                norm_class_kwargs["norm_class"] = norm_class

            residual_fn = partial(
                GradMofiedPreNormBlock,
                dmodel=dmodel,
                gn_placement=layer_gn_placement,
                gn_layer=grad_modif_fn,
                **norm_class_kwargs,
            )

            residual_layers.append(
                (f"residual_{name}", residual_fn(layer=layer, name=name))
            )

        self.block = nn.Sequential(OrderedDict(residual_layers))

    def forward(self, x):
        return self.block(x)
