import torch

import lizrd.core.nn as nn
from lizrd.core.misc import EinMix
from lizrd.support import ash


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
        multineck_1, expand, nn.ReLU(inplace=True), contract, multineck_2
    )


@ash.check("... d -> ... d")
class FeedForwardMultibias(nn.Module):
    def __init__(self, dmodel, dff, n_bias_copies):
        super().__init__()
        self.dmodel = dmodel
        self.dff = dff
        self.f1 = EinMix(
            "batch seqlean dmodel -> batch seqlen dff n_biases",
            weight_shape="dmodel dff",
            bias_shape="n_biases dff",
            dmodel=dmodel,
            dff=dff,
            n_biases=n_bias_copies,
        )
        self.f2 = nn.Linear(dff, dmodel)

    def forward(self, x):
        x = self.f1(x)
        x = x.sum(dim=-1) / (self.n_bias_copies) ** (1 / 2)
        x = self.f2(x)
        return x


@ash.check("... d -> ... d")
class FeedForwardMultibias(nn.Module):
    """
    the simplest way to increase nonlinearities: initialise a few sets of biases and try all of them simultaneously, then average the results
    """

    def __init__(self, dmodel, dff, n_bias_copies):
        super().__init__()
        self.dmodel = dmodel
        self.dff = dff
        self.f1 = EinMix(
            "batch seqlean dmodel -> batch seqlen dff n_biases",
            weight_shape="dmodel dff",
            bias_shape="n_biases dff",
            dmodel=dmodel,
            dff=dff,
            n_biases=n_bias_copies,
        )
        self.f2 = nn.Linear(dff, dmodel)

    def forward(self, x):
        x = self.f1(x)
        x = nn.ReLU(x)
        x = x.sum(dim=-1) / (self.n_bias_copies) ** (1 / 2)
        x = self.f2(x)
        return x


@ash.check("... d -> ... d")
class FeedForwardMultibiasMax(nn.Module):
    """
    the simplest way to increase nonlinearities: initialise a few sets of biases and try all of them simultaneously, then average the results
    """

    def __init__(self, dmodel, dff, n_bias_copies):
        super().__init__()
        self.dmodel = dmodel
        self.dff = dff
        self.f1 = EinMix(
            "batch seqlean dmodel -> batch seqlen dff n_biases",
            weight_shape="dmodel dff",
            bias_shape="n_biases dff",
            dmodel=dmodel,
            dff=dff,
            n_biases=n_bias_copies,
        )
        self.f2 = nn.Linear(dff, dmodel)

    def forward(self, x):
        x = self.f1(x)
        x = torch.max(x, dim=-1)
        x = self.f2(x)
        return x


class MultineckShuffle(nn.Module):
    def __init__(self, dmodel, dhead, n_heads, dff):
        super().__init__()
        self.dmodel = dmodel
        self.dhead = dhead
        self.n_heads = n_heads
        self.dff = dff
        self.split = EinMix(
            "batch seqlen dmodel -> batch seqlen nheads dhead",
            weight_shape="nheads dhead dmodel",
            bias_shape="",
            dmodel=dmodel,
            nheads=n_heads,
            dhead=dhead,
        )
        self.expand = EinMix(
            "batch seqlen nheads dhead -> batch seqlen nheads dff",
            weight_shape="nheads dhead dff",
            bias_shape="nheads dff",
            dff=dff,
            nheads=n_heads,
            dhead=dhead,
        )
        self.contract = EinMix(
            "batch seqlen nheads dff -> batch seqlen nheads dhead",
            weight_shape="nheads dff dhead",
            bias_shape="nheads dhead",
            dff=dff,
            nheads=n_heads,
            dhead=dhead,
        )
        self.aggregate = EinMix(
            "batch seqlen nheads dhead -> batch seqlen dmodel",
            weight_shape="nheads dhead dmodel",
            bias_shape="dmodel",
            dmodel=dmodel,
            nheads=n_heads,
            dhead=dhead,
        )

    def forward(self, x):
        pre = nn.Sequential(self.split, self.expand, nn.ReLU(inplace=True))
        post = nn.Sequential(self.contract, self.aggregate)
        x = pre(x)
        x = torch.permute(x, (0, 1, 3, 2)).resize(
            (x.shape[0], x.shape[1], self.n_heads, -1)
        )
        x = post(x)
        return x
