import torch
from einops.layers.torch import EinMix as OGEinMix
import opt_einsum
from torch import nn

import ash


class GeneralizedReLU(nn.Module):
    def __init__(self, ndim, bias=False):
        super(GeneralizedReLU, self).__init__()
        assert bias is False  # we assume bias was already added
        self.ndim = ndim
        self.alpha = nn.Parameter(torch.zeros(ndim))
        self.beta = nn.Parameter(torch.ones(ndim))
        self.a = nn.Parameter(torch.zeros(ndim))
        self.b = nn.Parameter(torch.zeros(ndim))

    def forward(self, x):
        above_zero = self.beta * x + self.b
        below_zero = self.alpha * x + self.a
        result = torch.where(x > 0., above_zero, below_zero)
        return result


# class ParameterLayer(nn.Module):
#     def __init__(self, tensor):
#         super(ParameterLayer, self).__init__()
#         self.parameter = nn.Parameter(tensor)
#
#     def forward(self, x):
#         del x
#         return self.parameter


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
def Dense(dinp, dout):
    return EinMix('... dinp -> ... dout',
                  weight_shape='dinp dout', bias_shape='dout',
                  dinp=dinp, dout=dout)


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