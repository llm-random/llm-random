import random
import string

import torch
from einops.layers.torch import EinMix as OGEinMix
import opt_einsum
from lizrd.core import nn
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


def get_init_weight(shape, fan_in, fan_out=None, gain=1.0, dtype=torch.float32):
    if fan_out is not None:
        raise ValueError("fan_out unsupported")
    range_ = gain * (3 / fan_in) ** 0.5
    return torch.zeros(shape, dtype=dtype).uniform_(-range_, range_)


def get_init_bias(shape, fan_in=None, fan_out=None, dtype=torch.float32):
    if fan_in is not None:
        raise ValueError("fan_in unsupported")
    if fan_out is not None:
        raise ValueError("fan_out unsupported")
    return torch.zeros(shape, dtype=dtype)


def einsum(subscript, *operands, use_opt_einsum=False, **kwargs):
    if use_opt_einsum:
        return opt_einsum.contract(subscript, *operands, **kwargs)
    else:
        return torch.einsum(subscript, *operands, **kwargs)


class EinMix(nn.Module):
    def __init__(self, signature, weight_shape=None, bias_shape=None, **kwargs):
        super(EinMix, self).__init__()
        self.change_anything = False
        if '...' in signature:
            self.change_anything = True
            self.og_signature = signature
            signature = signature.replace('...', 'squeezed')
        self.layer = OGEinMix(signature, weight_shape=weight_shape, bias_shape=bias_shape, **kwargs)
        if self.layer.bias is not None:
            self.layer.bias.data *= 0.0

    def forward(self, x):
        if not self.change_anything:
            return self.layer(x)
        # else
        beginning, end = self.og_signature.split('->')
        beginning = beginning.split()
        end = end.split()
        assert beginning[0] == end[0] == '...'
        # TODO(jaszczur): fix this hack below, properly
        contracted_dims = len(x.shape)-len(beginning)+1 + (1 if '(' in ''.join(beginning) else 0)
        ellipsis_shape = list(x.shape[:contracted_dims])
        newx = torch.reshape(x, [-1] + list(x.shape[contracted_dims:]))
        output = self.layer(newx)
        newoutput = torch.reshape(output, ellipsis_shape + list(output.shape[1:]))
        return newoutput


@ash.check('... inp -> ... out')
def DenseEinMix(dinp, dout):
    return EinMix('... dinp -> ... dout',
                  weight_shape='dinp dout', bias_shape='dout',
                  dinp=dinp, dout=dout)
                  

@ash.check('... inp -> ... out')
class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # This is to make sure values after the layer keep the variance
        self.weight.data *= 3 ** 0.5
        self.bias.data *= 0.0


def check_layer_funs(*layer_funs):
    for layer_fun in layer_funs:
        if isinstance(layer_fun, nn.Module):
            raise TypeError('Expected layer function/lambda, got nn.Module: {}'
                            .format(type(layer_fun)))


@ash.check('... -> ...')
class StopGradient(nn.Module):
    def __init__(self):
        super(StopGradient, self).__init__()
        pass

    def forward(self, x):
        return stop_gradient(x)


def stop_gradient(x):
    return x.detach()


@ash.check('... -> ...')
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


def Sum(*layers):
    return Aggregate((lambda x, y: x+y), *layers)


def GradientLike(value_layer, gradient_layer):
    return Sum(
        StopGradient(value_layer),
        StopValuePassGradient(gradient_layer),
    )


def print_available_gpus():
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print('Found {} GPU(s)'.format(count))
        for i in range(count):
            print('GPU {}: {}'.format(i, torch.cuda.get_device_name(i)))

def generate_random_string(length: int) -> str:
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))

def are_state_dicts_the_same(model_state_dict_1: dict, model_state_dict_2: dict) -> bool:
    if len(model_state_dict_1) != len(model_state_dict_2):
        print(
            f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}"
        )
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

    for ((k_1, v_1), (k_2, v_2)) in zip(
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

