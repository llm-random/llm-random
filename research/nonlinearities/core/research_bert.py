from collections import OrderedDict

import numpy as np
import torch

import lizrd.core.nn as nn
from lizrd.core.bert import LowRank
from lizrd.core.misc import EinMix, Linear
from lizrd.support import ash


@ash.check("... d -> ... d")
def FeedForwardBottleneck(dmodel, exp_rate, bottleneck_chop_ratio=None):
    """
    :param dmodel: dimension of the model
    :param exp_rate: M/N, where N is the dimension of the model and M is the number of neurons in the middle layer (before ReLU)
    :param bottleneck_chop_ratio: only relevant for FeedForwardInceptionNeck, carries out all the logic described next,
            and then divides the size of the bottleneck layer by the bottleneck_chop_ratio

    assumes that the number of parameters should be the same as for a FeedForward layer of the same input/output dimensions;
    with the bottleneck layer size being B, and in/out being N and M respectively, the resulting equation is:
        B(N+M) = NM

    we mainly want to compare ourselves with the original hyperparameters, so let's assume, that the M on th RHS is M = 4N
        B(N+M) = 4N^2
        B = 4N^2 / (N+M)

    now, with the expansion rate being a = M/N, we get
        B = 4N^2/(a+1)N = 4N/(a+1)
    to sum up, the above choice of the bottleneck size B guarantees that the number of parameters is the same as
    in a FeedForward layer of sizes d_in,d_out = N,4N
    """
    N = dmodel
    B = 4 * N / (exp_rate + 1)
    if bottleneck_chop_ratio is not None:
        B = int(B * bottleneck_chop_ratio)
    else:
        B = int(B)
    M = exp_rate * N
    block = nn.Sequential(
        OrderedDict(
            [
                ("logging_lowrank1", LowRank(N, M, B)),
                ("relu", nn.ReLU(inplace=True)),
                ("logging_lowrank2", LowRank(M, N, B)),
            ]
        )
    )
    return block


@ash.check("... d -> ... d")
def FeedForwardBottleneckFORCED(dmodel, dff, bottleneck_size):
    """
    Similar to FeedForwardBottleneck, but the bottleneck size is forced to be bottleneck_size
    """
    N = dmodel
    B = bottleneck_size
    M = dff
    block = nn.Sequential(
        OrderedDict(
            [
                ("logging_bottleneck1", Linear(N, B)),
                ("logging_bottleneck2", Linear(B, M)),
                ("relu", nn.ReLU(inplace=True)),
                ("logging_bottleneck3", Linear(M, B)),
                ("logging_bottleneck4", Linear(B, N)),
            ]
        )
    )
    return block


@ash.check("... d -> ... d")
def FeedForwardMultineck(
    dmodel, exp_rate, n_heads, parameter_sharing_mode: str = "none"
):
    """
    :param dmodel: dimension of the model
    :param exp_rate: exp_rate: M/N, where M is the size of the "expanded" layer (before ReLU)
    :param n_heads: number of independent bottlenecks, later aggregated
    :param parameter_sharing_mode: one of "none", "neck_and_ff", "input_and_neck"
    An iteration on FeedForwardBottleneck, where there are multiple bottlenecks, EACH writes to the output stream independently, like multiheadattention
    Assumes that the number of parameters should be the same as for a FeedForward layer of the same input/output dimensions;
    with the bottleneck layer size being B and the number of heads H, and in/out being N and M respectively, the resulting equation is:
        HB(N+M) = NM

    we mainly want to compare ourselves with the original hyperparameters, so let's assume, that the M on th RHS is M = 4N
        HB(N+M) = 4N^2
        HB = 4N^2 / (N+M)

    now, with the expansion rate being a = M/N, we get
        B = 4N^2/(a+1)NH = 4N/H(a+1)
    to sum up, the above choice of the bottleneck size B guarantees that the number of parameters is the same as
    in a dense FeedForward layer of sizes d_in,d_out = N,4N
    """
    assert (
        4 * dmodel % n_heads == 0
    ), f"4*dmodel = {4 * dmodel} should be divisible by n_heads={n_heads}"

    assert parameter_sharing_mode in [
        "none",
        "neck_and_ff",
        "input_and_neck",
    ], f"parameter_sharing_mode={parameter_sharing_mode} is not supported, should be one of 'none', 'neck_and_ff', 'input_and_neck'"

    if parameter_sharing_mode == "input_and_neck":
        raise NotImplementedError(
            "parameter_sharing_mode='input_and_neck' is not implemented yet"
        )

    weight_shapes = {
        "multineck_1": "nheads dmodel dhead",
        "expand": "nheads dhead dff",
        "contract": "nheads dff dhead",
        "multineck_2": "nheads dhead dmodel",
    }

    bias_shapes = {
        "multineck_1": "nheads dhead",
        "expand": "nheads dff",
        "contract": "nheads dhead",
        "multineck_2": "dmodel",
    }

    if parameter_sharing_mode == "neck_and_ff":
        weight_shapes["expand"] = "dhead dff"
        weight_shapes["contract"] = "dff dhead"

        bias_shapes["expand"] = "dff"
        bias_shapes["contract"] = "dhead"

    assert None not in [weight_shapes, bias_shapes]

    N = dmodel
    M = exp_rate * N
    B = int(4 * N / (n_heads * (exp_rate + 1)))

    multineck_1 = EinMix(
        "batch seqlen dmodel -> batch seqlen nheads dhead",
        weight_shape=weight_shapes["multineck_1"],
        bias_shape=bias_shapes["multineck_1"],
        dmodel=N,
        nheads=n_heads,
        dhead=B,
    )
    expand = EinMix(
        "batch seqlen nheads dhead -> batch seqlen nheads dff",
        weight_shape=weight_shapes["expand"],
        bias_shape=bias_shapes["expand"],
        dff=M,
        nheads=n_heads,
        dhead=B,
    )
    contract = EinMix(
        "batch seqlen nheads dff -> batch seqlen nheads dhead",
        weight_shape=weight_shapes["contract"],
        bias_shape=bias_shapes["contract"],
        dff=M,
        nheads=n_heads,
        dhead=B,
    )
    multineck_2 = EinMix(
        "batch seqlen nheads dhead -> batch seqlen dmodel",
        weight_shape=weight_shapes["multineck_2"],
        bias_shape=bias_shapes["multineck_2"],
        dmodel=N,
        nheads=n_heads,
        dhead=B,
    )

    block = nn.Sequential(
        OrderedDict(
            [
                ("logging_multineck_1", multineck_1),
                ("logging_expand", expand),
                ("relu", nn.ReLU(inplace=True)),
                ("logging_contract", contract),
                ("logging_multineck_2", multineck_2),
            ]
        )
    )

    return block


@ash.check("... d -> ... d")
class FeedForwardInceptionNeck(nn.Module):
    """
    iteration on FeedForwardMultiNeck, where the heads are not the same size, but are defined by the head_sizes list
    (as fractions of the resulting B dimension)
    :return:
    """

    def __init__(self, dmodel, exp_rate, head_sizes):
        super(FeedForwardInceptionNeck, self).__init__()
        self.head_sizes = head_sizes
        self.n_heads = len(head_sizes)
        self.bottleneck_heads = nn.ModuleList(
            [
                FeedForwardBottleneck(dmodel, exp_rate, bottleneck_chop_ratio=head_size)
                for head_size in head_sizes
            ]
        )

    def forward(self, x):
        x = torch.stack([head(x) for head in self.bottleneck_heads], dim=-1)
        x = torch.einsum("... h -> ...", x)
        x = x / np.sqrt(self.n_heads)
        return x


@ash.check("... d -> ... d")
def FeedForwardChoppedNeck(dmodel, n_chunks):
    """
    init params: dmodel, n_chunks
    Divides both the input vector and the ff layer into n_heads chunks, and restricts the dense layer to operate
    on each chunk pair independently, disallowing inter-pair communication.
    An abvious drawback of this approach is that the different dimensions of the chunks are not able to communicate with each other in the FF layer.
    An iteration on this idea should address that restriction.
    :param dmodel: dimension of the model
    :param n_chunks: number of chunks to divide the input vector into
    To calculate the ff_layer size, as in ForwardBottleneck, we formulate the following equation:
        n_chunks * (N/n_chunks * M/n_chunks) = N*4N
    solving for M, we get
        M = 4*N*n_chunks
    In short, chopping the input vector into n_chunks chunks results in the ff layer being 4 n_chunks times larger
    Hence every chunk in the ff layer is of the size 4*dmodel
    """
    assert (
        dmodel % n_chunks == 0
    ), f"dmodel={dmodel} should be divisible by n_chunks={n_chunks}"
    chunk_size = dmodel // n_chunks
    ff_chunk_size = 4 * dmodel

    return nn.Sequential(
        EinMix(
            "batch seqlen (n_chunks chunk_size)-> batch seqlen n_chunks dff_chunk_size",
            weight_shape="n_chunks chunk_size dff_chunk_size",
            bias_shape="(n_chunks dff_chunk_size)",
            n_chunks=n_chunks,
            chunk_size=chunk_size,
            dff_chunk_size=ff_chunk_size,
        ),
        nn.ReLU(inplace=True),
        EinMix(
            "batch seqlen n_chunks dff_chunk_size-> batch seqlen (n_chunks chunk_size)",
            weight_shape="n_chunks dff_chunk_size chunk_size",
            bias_shape="(n_chunks chunk_size)",
            n_chunks=n_chunks,
            chunk_size=chunk_size,
            dff_chunk_size=ff_chunk_size,
        ),
    )
