from collections import OrderedDict
from typing import Callable, Tuple

import torch.nn as nn

from lizrd.core.llm import Residual
from research.grad_norm.modules.grad_modif_placement import LayerGradModifPlacement


def get_gn_or_log_layer(
    log_name: str, gn_layer: Callable[[], nn.Module], log_layer: Callable[[], nn.Module], use_gn: bool
) -> Tuple[str, nn.Module]:
    return (log_name, gn_layer()) if use_gn else (log_name, log_layer())


def GradMofiedPreNormBlock(
    dmodel,
    layer,
    name,
    gn_layer: Callable[[], nn.Module],
    log_layer: Callable[[], nn.Module],
    gn_placement: LayerGradModifPlacement,
    norm_class=nn.LayerNorm,
):
    inside_residual_blocks = [
        ("pre_norm", norm_class(dmodel)),
        get_gn_or_log_layer("post_norm_gn", gn_layer, log_layer, gn_placement.post_norm),
        (f"{name}", layer),
        get_gn_or_log_layer("post_layer_gn", gn_layer, log_layer, gn_placement.post_layer),
    ]

    on_residual_blocks = [
        (
            "residual",
            Residual(nn.Sequential(OrderedDict([b for b in inside_residual_blocks if b is not None]))),
        ),
        get_gn_or_log_layer("post_add_gn", gn_layer, log_layer, gn_placement.post_add),
    ]

    return nn.Sequential(OrderedDict([b for b in on_residual_blocks if b is not None]))
