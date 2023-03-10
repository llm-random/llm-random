from collections import OrderedDict

import lizrd.core.nn as nn
from lizrd.core.bert import LowRank, Residual
from lizrd.core.misc import EinMix
from lizrd.support import ash
from research.nonlinearities.temporary_code.helper_layers import (
    Sum_norm,
    MultineckShufflePermute,
)


@ash.check("... d -> ... d")
def FeedForwardMultineckFORCED(
    dmodel, dhead, n_heads, dff, parameter_sharing_mode: str = "none"
):
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

    multineck_1 = EinMix(
        "batch seqlen dmodel -> batch seqlen nheads dhead",
        weight_shape=weight_shapes["multineck_1"],
        bias_shape=bias_shapes["multineck_1"],
        dmodel=dmodel,
        nheads=n_heads,
        dhead=dhead,
    )
    expand = EinMix(
        "batch seqlen nheads dhead -> batch seqlen nheads dff",
        weight_shape=weight_shapes["expand"],
        bias_shape=bias_shapes["expand"],
        dff=dff,
        nheads=n_heads,
        dhead=dhead,
    )
    contract = EinMix(
        "batch seqlen nheads dff -> batch seqlen nheads dhead",
        weight_shape=weight_shapes["contract"],
        bias_shape=bias_shapes["contract"],
        dff=dff,
        nheads=n_heads,
        dhead=dhead,
    )
    multineck_2 = EinMix(
        "batch seqlen nheads dhead -> batch seqlen dmodel",
        weight_shape=weight_shapes["multineck_2"],
        bias_shape=bias_shapes["multineck_2"],
        dmodel=dmodel,
        nheads=n_heads,
        dhead=dhead,
    )

    return nn.Sequential(
        OrderedDict(
            [
                ("logging_pre_expand", multineck_1),
                ("logging_expand", expand),
                ("relu", nn.ReLU(inplace=True)),
                ("logging_contract", contract),
                ("logging_post_contract", multineck_2),
            ]
        )
    )


@ash.check("... d -> ... d")
def FeedForwardMultibias(dmodel, dff, n_bias_copies):
    """
    the simplest way to increase nonlinearities: initialise a few sets of biases and try all of them simultaneously, then average the results
    """

    f1 = EinMix(
        "batch seqlen dmodel -> batch seqlen dff n_biases",
        weight_shape="dmodel dff",
        bias_shape="n_biases dff",
        dmodel=dmodel,
        dff=dff,
        n_biases=n_bias_copies,
    )
    f2 = nn.Linear(dff, dmodel)
    return nn.Sequential(
        OrderedDict(
            [
                ("logging_ff_split_to_multiple_biases", f1),
                ("relu", nn.ReLU(inplace=True)),
                ("logging_ff_aggregate_biases", Sum_norm(n_bias_copies=n_bias_copies)),
                ("logging_ff_output", f2),
            ]
        )
    )


def MultineckShuffle(dmodel, dhead, n_heads, dff):
    split = EinMix(
        "batch seqlen dmodel -> batch seqlen nheads dhead",
        weight_shape="nheads dhead dmodel",
        bias_shape="",
        dmodel=dmodel,
        nheads=n_heads,
        dhead=dhead,
    )
    expand = EinMix(
        "batch seqlen nheads dhead -> batch seqlen nheads dff",
        weight_shape="nheads dhead dff",
        bias_shape="nheads dff",
        dff=dff,
        nheads=n_heads,
        dhead=dhead,
    )
    contract = EinMix(
        "batch seqlen nheads dff -> batch seqlen nheads dhead",
        weight_shape="nheads dff dhead",
        bias_shape="nheads dhead",
        dff=dff,
        nheads=n_heads,
        dhead=dhead,
    )
    aggregate = EinMix(
        "batch seqlen nheads dhead -> batch seqlen dmodel",
        weight_shape="nheads dhead dmodel",
        bias_shape="dmodel",
        dmodel=dmodel,
        nheads=n_heads,
        dhead=dhead,
    )
    return nn.Sequential(
        OrderedDict(
            [
                ("logging_ff_split", split),
                ("logging_ff_expand", expand),
                ("ff_relu", nn.ReLU(inplace=True)),
                ("logging_ff_shuffle", MultineckShufflePermute(n_heads)),
                ("logging_ff_contract", contract),
                ("logging_ff_aggregate", aggregate),
            ]
        )
    )


@ash.check("... d -> ... d")
def FeedForwardBottleneckFORCED(dmodel, dbottle, dff):
    """
    Modification of a standard feed-forward transformer layer done by replacing each dense layer with two consecutive dense layers with no activation between them
    :param dmodel:
    :param dbottle:
    :param dff:
    :return:
    """
    block = nn.Sequential(
        OrderedDict(
            [
                ("logging_lowrank1", LowRank(dmodel, dff, dbottle)),
                ("relu", nn.ReLU(inplace=True)),
                ("logging_lowrank2", LowRank(dff, dmodel, dbottle)),
            ]
        )
    )
    return block


@ash.check("... d -> ... d")
def OverparametrisedFeedForward(dmodel, dff):
    """
    Modification of a standard feed-forward transformer layer done by replacing each dense layer with two consecutive dense layers with no activation between them
    :param dmodel: dimension of the model
    :param dff: dimension of the feedforward layer
    """
    block = nn.Sequential(
        OrderedDict(
            [
                ("logging_ff1", nn.Linear(dmodel, dff)),
                ("logging_ff2", nn.Linear(dff, dff)),
                ("relu", nn.ReLU(inplace=True)),
                ("logging_ff3", nn.Linear(dff, dff)),
                ("logging_ff4", nn.Linear(dff, dmodel)),
            ]
        )
    )
    return block


def OverparametrisedFeedForwardResidual(dmodel, dff):
    """
    Modification of a standard feed-forward transformer layer done by replacing each dense layer with two consecutive dense layers with no activation between them
    :param dmodel: dimension of the model
    :param dff: dimension of the feedforward layer
    """
    block = nn.Sequential(
        OrderedDict(
            [
                ("logging_ff1", nn.Linear(dmodel, dff)),
                ("logging_ff2", Residual(nn.Linear(dff, dff))),
                ("relu", nn.ReLU(inplace=True)),
                ("logging_ff3", Residual(nn.Linear(dff, dff))),
                ("logging_ff4", nn.Linear(dff, dmodel)),
            ]
        )
    )
    return block


def OverparametrisedFeedForwardNormed(dmodel, dff):
    """
    Modification of a standard feed-forward transformer layer done by replacing each dense layer with two consecutive dense layers with no activation between them
    :param dmodel: dimension of the model
    :param dff: dimension of the feedforward layer
    """
    block = nn.Sequential(
        OrderedDict(
            [
                ("logging_ff1", nn.Linear(dmodel, dff)),
                ("norm1", nn.LayerNorm(dff)),
                ("logging_ff2", nn.Linear(dff, dff)),
                ("relu", nn.ReLU(inplace=True)),
                ("logging_ff3", nn.Linear(dff, dff)),
                ("norm2", nn.LayerNorm(dff)),
                ("logging_ff4", nn.Linear(dff, dmodel)),
            ]
        )
    )
    return block


def OverparametrisedFeedForwardResidualNormed(dmodel, dff):
    """
    Modification of a standard feed-forward transformer layer done by replacing each dense layer with two consecutive dense layers with no activation between them
    :param dmodel: dimension of the model
    :param dff: dimension of the feedforward layer
    """
    block = nn.Sequential(
        OrderedDict(
            [
                ("logging_ff1", nn.Linear(dmodel, dff)),
                ("norm1", nn.LayerNorm(dff)),
                ("logging_ff2", Residual(nn.Linear(dff, dff))),
                ("relu", nn.ReLU(inplace=True)),
                ("logging_ff3", Residual(nn.Linear(dff, dff))),
                ("norm2", nn.LayerNorm(dff)),
                ("logging_ff4", nn.Linear(dff, dmodel)),
            ]
        )
    )
    return block


def RescaledMultineckFeedForward(dmodel, dhead, n_heads, dff, aggregate_per_head=False):
    """
    Like MultineckFeedForward, but with a rescaling layer (one parameter per head) before the expansion layer
    """

    multineck_1 = EinMix(
        "batch seqlen dmodel -> batch seqlen nheads dhead",
        weight_shape="nheads dmodel dhead",
        bias_shape="nheads dhead",
        dmodel=dmodel,
        nheads=n_heads,
        dhead=dhead,
    )
    expand = EinMix(
        "batch seqlen nheads dhead -> batch seqlen nheads dff",
        weight_shape="nheads dhead dff",
        bias_shape="nheads dff",
        dff=dff,
        nheads=n_heads,
        dhead=dhead,
    )
    rescale = EinMix(
        "batch seqlen nheads dff -> batch seqlen nheads dff",
        weight_shape="nheads",
        bias_shape="nheads",
        dff=dff,
        nheads=n_heads,
    )
    aggregate = EinMix(
        "batch seqlen nheads dff -> batch seqlen dmodel",
        weight_shape="nheads dff dmodel" if aggregate_per_head else "dff dmodel",
        bias_shape="dmodel",
        dmodel=dmodel,
        nheads=n_heads,
        dff=dff,
    )

    return nn.Sequential(multineck_1, expand, nn.ReLU(inplace=True), rescale, aggregate)


@ash.check("... d -> ... d")
def FeedForwardChoppedNeckFORCED(dmodel, n_chunks):
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
    ff_chunk_size = 4 * dmodel // n_chunks

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


@ash.check("... d -> ... d")
def FeedForwardMultilinear(dmodel, dff, nheads):
    """
    Sanity check for the EinMix layer, used to compare normal nn.Linear with EinMix
    """
    expand = EinMix(
        "batch seqlen dmodel -> batch seqlen nheads dff",
        weight_shape="dmodel dff nheads",
        bias_shape="nheads dff",
        dmodel=dmodel,
        nheads=nheads,
        dff=dff,
    )

    contract = EinMix(
        "batch seqlen nheads dff -> batch seqlen dmodel",
        weight_shape="nheads dff dmodel",
        bias_shape="dmodel",
        dmodel=dmodel,
        nheads=nheads,
        dff=dff,
    )

    block = nn.Sequential(
        OrderedDict(
            [
                ("logging_pre_relu", expand),
                ("relu", nn.ReLU(inplace=True)),
                ("logging_post_relu", contract),
            ]
        )
    )
    return block


@ash.check("... d -> ... d")
def LinearEinmix(dmodel, dff):
    """
    Sanity check for the EinMix layer, used to compare normal nn.Linear with EinMix
    """
    expand = EinMix(
        "batch seqlen dmodel -> batch seqlen dff",
        weight_shape="dmodel dff",
        bias_shape="dff",
        dmodel=dmodel,
        dff=dff,
    )

    contract = EinMix(
        "batch seqlen dff -> batch seqlen dmodel",
        weight_shape="dff dmodel",
        bias_shape="dmodel",
        dmodel=dmodel,
        dff=dff,
    )

    block = nn.Sequential(
        OrderedDict(
            [
                ("logging_pre_relu", expand),
                ("relu", nn.ReLU(inplace=True)),
                ("logging_post_relu", contract),
            ]
        )
    )
    return block
