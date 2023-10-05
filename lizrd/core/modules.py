import torch
from einops.layers.torch import EinMix as OGEinMix
from torch.utils.checkpoint import checkpoint

from lizrd.core import nn
from lizrd.core.init import get_init_weight
from lizrd.core.misc import stop_gradient
from lizrd.support import ash


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


@ash.check("... inp -> ... out")
def DenseEinMix(dinp, dout):
    return EinMix(
        "... dinp -> ... dout",
        weight_shape="dinp dout",
        bias_shape="dout",
        dinp=dinp,
        dout=dout,
    )


@ash.check("... inp -> ... out")
class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        if "bias" not in kwargs:
            kwargs["bias"] = False
        super().__init__(*args, **kwargs)
        self.weight.data = get_init_weight(
            self.weight.shape, self.in_features, dtype=self.weight.dtype
        )


@ash.check("... -> ...")
class StopGradient(nn.Module):
    def __init__(self):
        super(StopGradient, self).__init__()

    def forward(self, x):
        return stop_gradient(x)


@ash.check("... -> ...")
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


def GradientLike(value_layer, gradient_layer):
    return Sum(
        StopGradient(value_layer),
        StopValuePassGradient(gradient_layer),
    )
