import collections
from functools import partial
from typing import Tuple, DefaultDict, List, Any

import plotly_express as px
import torch

from lizrd.core import nn
from lizrd.core.misc import EinMix
from lizrd.support.ash import Check
from lizrd.support.logging import get_current_logger
from research.reinitialization.core.linears import prepare_tensor_for_logging


def get_parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_mean_and_std(ff_layer):
    assert isinstance(ff_layer, Check)
    layer = ff_layer.layer[0]
    if isinstance(layer, EinMix):
        weight = lambda x: x.layer.weight
    elif isinstance(layer, nn.Linear):
        weight = lambda x: x.weight
    else:
        raise NotImplementedError
    weight_tensor = weight(layer)
    return weight_tensor.mean().item(), weight_tensor.std().item()


def register_activation_hooks(
    model: nn.Module,
) -> Tuple[DefaultDict[List, torch.Tensor], List[Any]]:
    """Registers forward hooks in specified layers.
    Parameters
    ----------
    model:
        PyTorch model
    Returns
    -------
    activations_dict:
        dict of lists containing activations of specified layers in
        ``layers_to_save``.
    """
    activations_dict = collections.defaultdict(list)

    handles = []
    for name, module in model.named_modules():
        if "logging" in name:
            handle = module.register_forward_hook(
                partial(save_activations, activations_dict, name)
            )
            handles.append(handle)
    return activations_dict, handles


def save_activations(
    activations: DefaultDict, name: str, module: nn.Module, inp, out: torch.Tensor
) -> None:
    """PyTorch Forward hook to save outputs at each forward
    pass. Mutates specified dict objects with each fwd pass.
    """
    activations[name] = out.detach().cpu()


def log_tensor(tensor, name, series, step):
    logger = get_current_logger()
    fig = px.histogram(prepare_tensor_for_logging(tensor, with_replacement=False))
    logger.report_plotly(
        figure=fig,
        title=name,
        series=series,
        iteration=step,
    )


def log_scalar(*, value, name, series, step):
    logger = get_current_logger()
    logger.report_scalar(
        title=name,
        series=series,
        value=value,
        iteration=step,
    )
