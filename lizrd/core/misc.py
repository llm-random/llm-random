from contextlib import contextmanager
from functools import wraps
import time
from typing import Union
from plotly import express as px
from einops.layers.torch import EinMix as OGEinMix
import opt_einsum
import torch
from torch.utils.checkpoint import checkpoint

import torch.nn as nn
from lizrd.core.initialization import get_init_weight


class Noop(nn.Module):
    def __init__(self):
        super(Noop, self).__init__()

    def forward(self, x):
        return x


class ParameterLayer(nn.Module):
    def __init__(self, tensor):
        super(ParameterLayer, self).__init__()
        self.parameter = nn.Parameter(tensor)

    def forward(self, x):
        del x
        return self.parameter


def einsum(subscript, *operands, use_opt_einsum=False, **kwargs):
    if use_opt_einsum:
        return opt_einsum.contract(subscript, *operands, **kwargs)
    else:
        return torch.einsum(subscript, *operands, **kwargs)


class EinMix(nn.Module):
    def __init__(self, signature, weight_shape=None, bias_shape=None, **kwargs):
        super(EinMix, self).__init__()
        self.change_anything = False
        if "..." in signature:
            self.change_anything = True
            self.og_signature = signature
            signature = signature.replace("...", "squeezed")
        self.layer = OGEinMix(
            signature, weight_shape=weight_shape, bias_shape=bias_shape, **kwargs
        )
        if self.layer.bias is not None:
            self.layer.bias.data *= 0.0

    def forward(self, x):
        if not self.change_anything:
            return self.layer(x)
        # else
        beginning, end = self.og_signature.split("->")
        beginning = beginning.split()
        end = end.split()
        assert beginning[0] == end[0] == "..."
        # TODO(jaszczur): fix this hack below, properly
        contracted_dims = (
            len(x.shape) - len(beginning) + 1 + (1 if "(" in "".join(beginning) else 0)
        )
        ellipsis_shape = list(x.shape[:contracted_dims])
        newx = torch.reshape(x, [-1] + list(x.shape[contracted_dims:]))
        output = self.layer(newx)
        newoutput = torch.reshape(output, ellipsis_shape + list(output.shape[1:]))
        return newoutput


def DenseEinMix(dinp, dout):
    return EinMix(
        "... dinp -> ... dout",
        weight_shape="dinp dout",
        bias_shape="dout",
        dinp=dinp,
        dout=dout,
    )


class Linear(nn.Linear):
    def __init__(self, *args, init_type, init_scale, **kwargs):
        if "bias" not in kwargs:
            kwargs["bias"] = False
        super().__init__(*args, **kwargs)
        self.weight.data = get_init_weight(
            shape=self.weight.shape,
            fan_in=self.in_features,
            init_type=init_type,
            scale=init_scale,
            dtype=self.weight.dtype,
        )


def check_layer_funs(*layer_funs):
    for layer_fun in layer_funs:
        if isinstance(layer_fun, nn.Module):
            raise TypeError(
                "Expected layer function/lambda, got nn.Module: {}".format(
                    type(layer_fun)
                )
            )


class StopGradient(nn.Module):
    def __init__(self):
        super(StopGradient, self).__init__()

    def forward(self, x):
        return stop_gradient(x)


def stop_gradient(x):
    return x.detach()


class StopValuePassGradient(nn.Module):
    def __init__(self):
        super(StopValuePassGradient, self).__init__()

    def forward(self, x):
        return x - x.detach()


class Aggregate(nn.Module):
    def __init__(self, function, *layers):
        super(Aggregate, self).__init__()
        self.function = function
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        result = None
        for layer in self.layers:
            if result is None:
                result = layer(x)
            else:
                result = self.function(result, layer(x))
        return result


class Chungus(nn.Module):
    """
    https://i.ytimg.com/vi/inD-WWvtTW0/maxresdefault.jpg
    """

    def __init__(self, module, n_chungs):
        super(Chungus, self).__init__()
        self.module = module
        self.n_chungs = n_chungs

    def custom(self):
        def custom_forward(*inputs):
            output = self.module(inputs[0])
            return output

        return custom_forward

    def forward(self, x):
        output = []
        chunged_inputs = torch.chunk(x, self.n_chungs, dim=0)
        for chunged_input in chunged_inputs:
            partial_output = checkpoint(
                self.custom(),
                chunged_input,
            )
            output.append(partial_output)
        return torch.cat(output, dim=0)


def Sum(*layers):
    return Aggregate((lambda x, y: x + y), *layers)


def GradientLike(value_layer, gradient_layer):
    return Sum(
        StopGradient(value_layer),
        StopValuePassGradient(gradient_layer),
    )


def get_default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def default(x, d):
    return x if x is not None else d


def print_available_gpus():
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print("Found {} GPU(s)".format(count))
        for i in range(count):
            print("GPU {}: {}".format(i, torch.cuda.get_device_name(i)))


def are_state_dicts_the_same(
    model_state_dict_1: dict, model_state_dict_2: dict
) -> bool:
    if len(model_state_dict_1) != len(model_state_dict_2):
        print(f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}")
        return False

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(model_state_dict_1.keys())).startswith("module"):
        model_state_dict_1 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_1.items()
        }

    if next(iter(model_state_dict_2.keys())).startswith("module"):
        model_state_dict_2 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_2.items()
        }

    for (k_1, v_1), (k_2, v_2) in zip(
        model_state_dict_1.items(), model_state_dict_2.items()
    ):
        if k_1 != k_2:
            print(f"Key mismatch: {k_1} vs {k_2}")
            return False
        # convert both to the same CUDA device
        if str(v_1.device) != "cuda:0":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda:0":
            v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        if not torch.allclose(v_1, v_2):
            print(f"Tensor mismatch: {v_1} vs {v_2}")
            return False
    return True


def get_neuron_magnitudes(
    lin1_weight: torch.Tensor, lin2_weight: torch.Tensor
) -> torch.Tensor:
    weights1 = torch.sqrt(einsum("f m -> f", lin1_weight**2))
    weights2 = torch.sqrt(einsum("m f -> f", lin2_weight**2))

    return (weights1 * weights2).flatten()


def get_split_neuron_magnitudes(
    lin1_weight: torch.Tensor, lin2_weight: torch.Tensor
) -> torch.Tensor:
    """
    Returns the magnitude of the matrix formed by the concatenation of the two weight matrices.
    """
    weights1 = torch.sqrt(einsum("f m -> f", lin1_weight**2))
    weights2 = torch.sqrt(einsum("m f -> f", lin2_weight**2))

    # return concatenation
    return torch.cat((weights1**2, weights2**2), dim=0).flatten()


def get_mixed_neuron_magnitudes(
    lin1_weight: torch.Tensor, lin2_weight: torch.Tensor
) -> torch.Tensor:
    """
    Returns magnitudes of "nerons" formed by random combinations of rows/cols
    """
    weights1 = torch.sqrt(einsum("f m -> f", lin1_weight**2))
    weights2 = torch.sqrt(einsum("m f -> f", lin2_weight**2))

    weights1 = weights1.flatten()
    weights2 = weights2.flatten()
    weights1 = weights1.flip(0)
    return weights1 * weights2


def get_dmodel_magnitudes(
    lin1_weight: torch.Tensor, lin2_weight: torch.Tensor
) -> torch.Tensor:
    """
    Aggregate by dmodel instead of dff
    """
    weights1 = torch.sqrt(einsum("f m -> m", lin1_weight**2))
    weights2 = torch.sqrt(einsum("m f -> m", lin2_weight**2))

    return (weights1 * weights2).flatten()


def resolve_activation_name(activation: str) -> nn.Module:
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "silu":
        return nn.SiLU()
    elif activation == "softmax_last":
        return nn.Softmax(dim=-1)
    elif activation == "softmax":
        return nn.Softmax()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unrecognized activation: {activation}")


def propagate_forward_pass_cache(module: nn.Module, forward_pass_cache=None):
    """
    This function propagates the cache from the module to all its children.
    """
    if forward_pass_cache is None:
        forward_pass_cache = dict()
    module.forward_pass_cache = forward_pass_cache
    for child in module.children():
        propagate_forward_pass_cache(child, forward_pass_cache)


class MeasuringLayer(nn.Module):
    def __init__(self, layer, name, parent):
        super().__init__()
        self.l = layer
        self.name = name
        self.parent = [parent]

    def forward(self, *args, **kwargs):
        with measure_time(self.parent[0], self.name):
            return self.l(*args, **kwargs)


class LoggingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # info about position in model
        self.layer_type: Union[str, None] = None
        self.block_number: Union[int, None] = None

        # whether to log
        self.logging_switch = False

        # caches for logging and propagation
        self.logging_cache = {}
        self.forward_pass_cache: Union[dict, None] = None

    def clean_up_after_logging(self):
        assert self.logging_switch
        self.logging_switch = False
        self.logging_cache = {}

    def prepare_for_logging(self):
        self.logging_switch = True

    def update_cache_for_logging(self, key, value):
        if self.logging_switch:
            if isinstance(value, dict):
                if key in self.logging_cache:
                    self.logging_cache[key].update(value)
                else:
                    self.logging_cache[key] = value
            elif isinstance(value, torch.Tensor):
                if value.dtype == torch.bfloat16:
                    value = value.float()
                self.logging_cache[key] = value.clone().detach().cpu()
            elif isinstance(value, float) or isinstance(value, int):
                self.logging_cache[key] = value
            else:
                raise NotImplementedError

    def _combine_to_dict_key(self, key, layer_type, block_number):
        return f"block_{block_number}_{layer_type}_{key}"

    def update_forward_pass_cache(self, key, value):
        combined_key = self._combine_to_dict_key(
            key, self.layer_type, self.block_number
        )
        self.forward_pass_cache[combined_key] = value

    def get_from_forward_pass_cache(self, key, block_number, layer_type):
        combined_key = self._combine_to_dict_key(key, layer_type, block_number)
        return self.forward_pass_cache[combined_key]

    def log(self, verbosity_level):
        if verbosity_level == 0:
            return self.log_time()
        elif verbosity_level == 1:
            return self.log_light()
        elif verbosity_level == 2:
            return self.log_heavy()
        else:
            raise Exception("Invalid verbosity level")

    def log_light(self):
        return {}

    def log_heavy(self):
        return {}

    def log_time(self):
        log = {}
        if "time" in self.logging_cache:
            instr_names = list(self.logging_cache["time"].keys())
            instr_times = list(self.logging_cache["time"].values())
            times_fig = px.bar(x=instr_names, y=instr_times)
            log["time"] = times_fig
        return log

    def measure(self, module, name, exists=True):
        if not exists:
            return nn.Identity()
        return MeasuringLayer(module, name, self)


@contextmanager
def measure_time(layer: LoggingLayer, instruction_name: str):
    """
    This simple context manager is used to measure the time of a block of code.
    Args:
        layer: The LoggingLayer object that will be used to cache the time.
        instruction_name: The name of the instruction that is being measured.
    """
    if layer.logging_switch:
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start = time.time()
    yield
    if layer.logging_switch:
        if torch.cuda.is_available():
            end.record()
            torch.cuda.synchronize()
            layer.update_cache_for_logging(
                "time", {instruction_name: start.elapsed_time(end)}
            )
        else:
            end = time.time()
            layer.update_cache_for_logging("time", {instruction_name: end - start})


def time_measured(name):
    def _decorator(func):
        @wraps(func)
        def _decorator_wrapper(self, *args, **kwargs):
            with measure_time(self, name):
                return func(self, *args, **kwargs)

        return _decorator_wrapper

    return _decorator
