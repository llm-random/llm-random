import torch
import torch.nn
from torch.utils.checkpoint import checkpoint

from torch import nn
from lizrd.core.initialization import get_init_weight


class Linear(torch.nn.Linear):
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


class Aggregate(nn.Module):
    def __init__(self, function, *layers):
        super(Aggregate, self).__init__()
        self.function = function
        self.layers = torch.nn.ModuleList(layers)

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


class Checkpoint(nn.Module):
    def __init__(self, module):
        super(Checkpoint, self).__init__()
        self.module = module

    def custom(self):
        def custom_forward(*inputs):
            output = self.module(inputs[0])
            return output

        return custom_forward

    def forward(self, x):
        return checkpoint(self.custom(), x)


def Sum(*layers):
    return Aggregate((lambda x, y: x + y), *layers)


def print_available_gpus():
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print("Found {} GPU(s)".format(count))
        for i in range(count):
            print("GPU {}: {}".format(i, torch.cuda.get_device_name(i)))


def resolve_activation_name(activation: str) -> torch.nn.Module:
    if activation == "relu":
        return torch.nn.ReLU()
    elif activation == "gelu":
        return torch.nn.GELU()
    elif activation == "silu":
        return torch.nn.SiLU()
    elif activation == "softmax":
        return torch.nn.Softmax()
    else:
        raise ValueError(f"Unrecognized activation: {activation}")


def propagate_forward_pass_cache(module: torch.nn.Module, forward_pass_cache=None):
    """
    This function propagates the cache from the module to all its children.
    """
    if forward_pass_cache is None:
        forward_pass_cache = dict()
    module.forward_pass_cache = forward_pass_cache
    for child in module.children():
        propagate_forward_pass_cache(child, forward_pass_cache)


def decode_bias_string(bias):
    assert bias in ["both", "first", "second", "none"]
    if bias == "both":
        bias_first = bias_second = True
    elif bias == "first":
        bias_first = True
        bias_second = False
    elif bias == "second":
        bias_first = False
        bias_second = True
    else:
        bias_first = bias_second = False
    return bias_first, bias_second
