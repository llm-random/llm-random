from collections import OrderedDict
from typing import Literal, Callable, Optional
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from lizrd.core import misc
from lizrd.core.misc import default, Aggregate
from lizrd.core.initialization import get_init_weight, ValidInitType
from lizrd.core.misc import Linear, LoggingLayer
from lizrd.support.logging import make_histogram
import plotly.express as px


def decode_bias_string(bias):
    assert bias in ["both", "first", "second", "none"]
    if bias == "both":
        bias_first = bias_second = True
    elif bias == "first":
        bias_first = True
        bias_second = False
    elif bias == "second":
        bias_first = False
        bias_second = True
    else:
        bias_first = bias_second = False
    return bias_first, bias_second


class SwiGLUFeedForward(LoggingLayer):
    def __init__(
        self,
        dmodel,
        dff,
        init_type: ValidInitType,
        init_scale: float,
    ):
        super().__init__()
        self.w1_gate = Linear(
            dmodel, dff * 2, init_type=init_type, init_scale=init_scale, bias=False
        )
        self.w2 = Linear(
            dff, dmodel, init_type=init_type, init_scale=init_scale, bias=False
        )

    def forward(self, x):
        pre_activation, gate = torch.chunk(self.w1_gate(x), 2, dim=-1)
        activation = nn.functional.silu(pre_activation)
        return self.w2(activation * gate)


class GeneralizedRelu(LoggingLayer):
    def __init__(
        self,
        dmodel,
        dff,
        polynomial_config: str,
        init_type: ValidInitType,
        init_scale: float,
    ):
        # init: random / given
        # learnable: yes / no
        # 0/nl,1/nl <- relu
        # 0/nl,0/nl,1/nl <- relu2
        # 1/nl <- step function
        super().__init__()
        self.w1 = Linear(
            dmodel, dff, init_type=init_type, init_scale=init_scale, bias=False
        )
        self.w2 = Linear(
            dff, dmodel, init_type=init_type, init_scale=init_scale, bias=False
        )
        coefficients = []
        for single_config in polynomial_config.split(","):
            init_val, learnable = single_config.split("/")
            if init_val == "?":
                init_val = torch.randn(dff, dtype=torch.float) * init_scale
            else:
                init_val = float(init_val) * torch.ones(dff, dtype=torch.float)

            if learnable == "l":
                learnable = True
            elif learnable == "nl":
                learnable = False
            else:
                raise ValueError(f"Unknown learnable type: {learnable}")
            coefficients.append(torch.nn.Parameter(init_val, requires_grad=learnable))
        self.coefficients = torch.nn.ParameterList(coefficients)
        self.polynomial_config = polynomial_config

    def forward(self, x):
        hidden = self.w1(x)
        gt0 = hidden >= 0
        output = torch.zeros_like(hidden)
        for i, coeff in enumerate(self.coefficients):
            self.update_cache_for_logging(f"coeff_{i}", coeff)

            coeff = coeff.unsqueeze(0).unsqueeze(0)

            if i == 0:
                output += coeff * gt0
            else:
                output += coeff * (hidden**i) * gt0

        return self.w2(output)

    def log_heavy(self):
        # log histograms of the coefficients
        histograms = {}
        for i, coeff in enumerate(self.logging_cache.values()):
            histograms[f"coeff_{i}"] = px.histogram(
                x=coeff.numpy(),
                title=f"coeff_{i}",
            )

        return histograms


def FeedForward(
    dmodel,
    dff,
    init_type: ValidInitType,
    init_scale: float,
    bias: Literal["both", "first", "second", "none"] = "both",
):
    bias_first, bias_second = decode_bias_string(bias)

    return nn.Sequential(
        OrderedDict(
            [
                (
                    "logging_ff_pre_relu",
                    Linear(
                        dmodel,
                        dff,
                        bias=bias_first,
                        init_type=init_type,
                        init_scale=init_scale,
                    ),
                ),
                ("relu", nn.ReLU()),
                (
                    "logging_ff_post_relu",
                    Linear(
                        dff,
                        dmodel,
                        bias=bias_second,
                        init_type=init_type,
                        init_scale=init_scale,
                    ),
                ),
            ]
        )
    )
