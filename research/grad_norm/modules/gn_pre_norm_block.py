import torch.nn as nn

from collections import OrderedDict
from lizrd.core.llm import Residual
from typing import Callable, Optional
from research.grad_norm.modules.grad_modif_placement import LayerGradModifPlacement


def GradMofiedPreNormBlock(
    dmodel,
    layer,
    name,
    gn_placement: LayerGradModifPlacement,
    norm_class=nn.LayerNorm,
    gn_layer: Optional[Callable[[], nn.Module]] = None,
):
    inside_residual_blocks = [
        ("pre_norm", norm_class(dmodel)),
        ("post_norm_gn", gn_layer()) if gn_placement.post_norm else None,
        (f"{name}", layer),
        ("post_layer_gn", gn_layer()) if gn_placement.post_layer else None,
    ]

    on_residual_blocks = [
        (
            "residual",
            Residual(
                nn.Sequential(
                    OrderedDict([b for b in inside_residual_blocks if b is not None])
                )
            ),
        ),
        ("post_add_gn", gn_layer()) if gn_placement.post_add else None,
    ]

    return nn.Sequential(OrderedDict([b for b in on_residual_blocks if b is not None]))
