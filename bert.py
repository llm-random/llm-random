import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import time
from time import sleep
from opt_einsum import contract
import opt_einsum
from einops.layers.torch import Rearrange, Reduce
#https://einops.rocks/3-einmix-layer/

from einops.layers.torch import EinMix as OGEinMix

import gc
from typing import List


class EinMix(nn.Module):
    def __init__(self, signature, **kwargs):
        super(EinMix, self).__init__()
        self.change_anything = False
        if '...' in signature:
            self.change_anything = True
            self.og_signature = signature
            signature = signature.replace('...', 'squeezed')
        self.layer = OGEinMix(signature, **kwargs)

    def forward(self, x):
        if not self.change_anything:
            return self.layer(x)
        # else
        beginning, end = self.og_signature.split('->')
        beginning = beginning.split()
        end = end.split()
        assert beginning[0] == end[0] == '...'
        contracted_dims = len(x.shape)-len(beginning)+1
        ellipsis_shape = list(x.shape[:contracted_dims])
        newx = torch.reshape(x, [-1] + list(x.shape[contracted_dims:]))
        output = self.layer(newx)
        newoutput = torch.reshape(output, ellipsis_shape + list(output.shape[1:]))
        return newoutput


def Dense(dinp, dout):
    return EinMix('... dinp -> ... dout',
                  weight_shape='dinp dout', bias_shape='dout',
                  dinp=dinp, dout=dout)


def FeedForward(dmodel, dff):
    return nn.Sequential(
        EinMix('... dm -> ... dff',
               weight_shape='dm dff', bias_shape='dff',
               dm=dmodel, dff=dff),
        nn.ReLU(),
        EinMix('... dff -> ... dm',
               weight_shape='dff dm', bias_shape='dm',
               dm=dmodel, dff=dff)
    )


class Residual(nn.Module):
    def __init__(self, layer):
        super(Residual, self).__init__()
        self.layer = layer

    def forward(self, x):
        out = self.layer(x)
        return out + x


class Attention(nn.Module):
    def __init__(self, dmodel, heads, dhead=None):
        super(Attention, self).__init__()
        if dhead is None:
            assert dmodel % heads == 0
            dhead = dmodel // heads
        layer_fun = lambda: EinMix('... dm -> ... heads dhead',
                                   weight_shape='dm heads dhead', bias_shape='heads dhead',
                                   dm=dmodel, heads=heads, dhead=dhead)
        self.Q = layer_fun()
        self.K = layer_fun()
        self.V = layer_fun()

        # self.A = Reduce('... seqlen1 heads dhead, ... seqlen2 heads dhead -> ... heads seqlen1 seqlen2')

        self.D = EinMix('... heads dhead -> ... dm',
                   weight_shape='heads dhead dm', bias_shape='dm',
                   dm=dmodel, heads=heads, dhead=dhead)

    def forward(self, x):
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        a = torch.einsum('... l h d, ... L h d -> ... h l L',
                         q, k)
        a = torch.softmax(a, dim=-1)
        prefinal = torch.einsum('... h l L, ... L h d -> ... l h d', a, v)
        output = self.D(prefinal)
        return output


def ResidualBlock(layer):
    return Residual(nn.Sequential(
        nn.LayerNorm(),
        layer,
        nn.LayerNorm(),
    ))


def EncoderBlock(*layers):
    residual_layers = []
    for layer in layers:
        residual_layers.append(ResidualBlock(layer))
    return nn.Sequential(*residual_layers)


def EncoderTower(n_blocks, *layer_funs):
    encoder_blocks = []
    for i_block in range(n_blocks):
        layers = [layer_fun() for layer_fun in layer_funs]
        encoder_blocks.append(EncoderBlock(*layers))
    return nn.Sequential(*encoder_blocks)