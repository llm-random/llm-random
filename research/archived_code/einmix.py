import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import time
from time import sleep
from opt_einsum import contract
import opt_einsum
from einops.layers.torch import Rearrange, Reduce

# https://einops.rocks/3-einmix-layer/

import gc
from typing import List

from einops.layers.torch import EinMix as Mix
import torch
from lizrd.core import nn
from einops.layers.torch import Rearrange, Reduce


def DenseFF(dmodel, dff):
    return nn.Sequential(
        Mix("b d -> b f", weight_shape="d f", bias_shape="f", d=dmodel, f=dff),
        nn.ReLU(),
        Mix("b f -> b d", weight_shape="d f", bias_shape="d", d=dmodel, f=dff),
    )


class Residual(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.inner_layer = layer

    def forward(self, x):
        res = self.inner_layer(x)
        return x + res


def SparseFFController(dmodel, dff, lowrank, sparsity):
    if sparsity == 0:
        sparsity = 1
    assert dff % sparsity == 0
    Nexperts = sparsity
    Nsets = dff // sparsity
    return nn.Sequential(
        Mix("b dm -> b lowrank", weight_shape="dm lowrank", dm=dmodel, lowrank=lowrank),
        Mix(
            "b lowrank -> b Ns Ne",
            weight_shape="lowrank Ns Ne",
            bias_shape="Ns Ne",
            lowrank=lowrank,
            Ns=Nsets,
            Ne=Nexperts,
        ),
        nn.Softmax(dim=-1),
    )


class Multiply(nn.Module):
    def __init__(self, layer1, layer2):
        super().__init__()
        self.layer1 = layer1
        self.layer2 = layer2

    def forward(self, x):
        r1 = self.layer1(x)
        r2 = self.layer2(x)
        return r1 * r2


def SparseFF(dmodel, dff, lowrank, sparsity):
    if sparsity == 0:
        sparsity = 1
    assert dff % sparsity == 0
    Nexperts = sparsity
    Nsets = dff // sparsity
    return nn.Sequential(
        Multiply(
            SparseFFController(dmodel, dff, lowrank, sparsity),
            Mix(
                "b dm -> b Ns Ne",
                weight_shape="dm Ns Ne",
                bias_shape="Ns Ne",
                dm=dmodel,
                Ns=Nsets,
                Ne=Nexperts,
            ),
        ),
        nn.ReLU(),
        Mix(
            "b Ns Ne -> b dm",
            weight_shape="Ns Ne dm",
            bias_shape="dm",
            dm=dmodel,
            Ns=Nsets,
            Ne=Nexperts,
        ),
    )


def testmodel(batch, dmodel, dff):
    batch = 128
    dmodel = 1024
    dff = 4096
    lowrank = 64
    sparsity = 32
    model = Residual(SparseFF(dmodel, dff, lowrank, sparsity))
    # model = SparseFFController(dmodel, dff, lowrank, sparsity)
    sample = torch.Tensor(np.random.random((batch, dmodel)))
    model.train()
    result = str(model(sample).shape)
    print("worked", result)


if __name__ == "__main__":
    testmodel(128, 1024, 4096)
