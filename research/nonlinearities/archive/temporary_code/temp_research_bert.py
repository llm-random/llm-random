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