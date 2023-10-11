import collections
from functools import partial
from typing import DefaultDict

import numpy as np
import plotly_express as px
import torch

from lizrd.core import nn
from lizrd.core.misc import EinMix
from lizrd.support.ash import Check
from lizrd.support.logging import get_current_logger


def get_parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_mean_and_std(ff_layer):
    assert isinstance(ff_layer, Check)
    if isinstance(ff_layer, EinMix):
        weight = lambda x: x.layer.weight
    elif isinstance(ff_layer, nn.Linear):
        weight = lambda x: x.weight
    else:
        raise NotImplementedError
    weight_tensor = weight(ff_layer)
    return weight_tensor.mean().item(), weight_tensor.std().item()


def register_activation_hooks(
    model: nn.Module,
):
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

    def _save_activations(
        activations: DefaultDict, name: str, module: nn.Module, inp, out: torch.Tensor
    ) -> None:
        """PyTorch Forward hook to save outputs at each forward
        pass. Mutates specified dict objects with each fwd pass.
        """
        activations[name] = out.detach().cpu()

    activations_dict = collections.defaultdict(list)

    handles = []
    for name, module in model.named_modules():
        if "logging" in name:
            handle = module.register_forward_hook(
                partial(_save_activations, activations_dict, name)
            )
            handles.append(handle)
    return activations_dict, handles


def prepare_tensor_for_logging(
    x: torch.Tensor, sample_size=2500, with_replacement=False
):
    """Prepare tensor or tensors for logging by sampling it to a maximum of `sample_size` elements.
    Default sample size = 2500 is selected because (experimentally) this works with ClearML plotting
    """
    num_elems = x.numel()
    x = x.detach().view(-1).cpu().numpy()

    if num_elems <= sample_size:
        return x.tolist()

    random_indices = np.random.choice(num_elems, sample_size, replace=with_replacement)
    ret_val = x[random_indices].tolist()
    return ret_val


# TODO: move the following to logging in the future
def log_tensor_distribution(*, tensor, name, series, step):
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
