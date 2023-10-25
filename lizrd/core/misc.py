from einops.layers.torch import EinMix as OGEinMix
import opt_einsum
import torch
import torch.nn
from torch.utils.checkpoint import checkpoint

from lizrd.core import nn
from lizrd.core.initialization import get_init_weight
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


@ash.check("... -> ...")
class StopGradient(nn.Module):
    def __init__(self):
        super(StopGradient, self).__init__()

    def forward(self, x):
        return stop_gradient(x)


def stop_gradient(x):
    return x.detach()


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


def resolve_activation_name(activation: str) -> torch.nn.Module:
    if activation == "relu":
        return nn.ReLU()
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
