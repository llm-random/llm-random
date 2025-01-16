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


def ProjectedFeedForward( #dev
    dmodel,
    dff,
    projected_dmodel,
    projected_dff,
    init_type: ValidInitType,
    init_scale: float,
    bias: Literal["both", "first", "second", "none"] = "both",
):
    """
    P1 = torch.rand(xs, xb)
    W = torch.rand(xb, yb)
    P2 = torch.rand(yb, ys)
    P1@W@P2 = (xs, ys)

    :param _type_ dmodel: _description_ #xb
    :param _type_ dff: _description_ #yb
    :param _type_ projected_dmodel: _description_ #xs
    :param _type_ projected_dff: _description_ #ys
    :param ValidInitType init_type: _description_
    :param float init_scale: _description_
    :param Literal[&quot;both&quot;, &quot;first&quot;, &quot;second&quot;, &quot;none&quot;] bias: _description_, defaults to "both"
    :return _type_: _description_
    """

    bias_first, bias_second = decode_bias_string(bias)
    return nn.Sequential(
        OrderedDict(
            [

                (
                    "logging_ff_pre_relu_p11",
                    Linear(
                        dmodel, #xs
                        projected_dmodel, #xb
                        bias=bias_first,
                        init_type=init_type,
                        init_scale=init_scale,
                    ),
                ),
                (
                    "logging_ff_pre_relu",
                    Linear(
                        projected_dmodel, #xb
                        projected_dff, #yb
                        bias=bias_first,
                        init_type=init_type,
                        init_scale=init_scale,
                    ),
                ),
                (
                    "logging_ff_pre_relu_p12",
                    Linear(
                        projected_dff, #yb
                        dff, #ys
                        bias=bias_first,
                        init_type=init_type,
                        init_scale=init_scale,
                    ),
                ),
                ("relu", nn.ReLU()),
                (
                    "logging_ff_post_relu_p21",
                    Linear(
                        dff, #ys
                        projected_dff, #yb
                        bias=bias_second,
                        init_type=init_type,
                        init_scale=init_scale,
                    ),
                ),
                (
                    "logging_ff_post_relu",
                    Linear(
                        projected_dff, #yb
                        projected_dmodel, #xb
                        bias=bias_second,
                        init_type=init_type,
                        init_scale=init_scale,
                    ),
                ),
                (
                    "logging_ff_post_relu_p22",
                    Linear(
                        projected_dmodel, #xb
                        dmodel, #xs
                        bias=bias_second,
                        init_type=init_type,
                        init_scale=init_scale,
                    ),
                ),
            ]
        )
    )


def attention_mechanism(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dhead: int,
    causal: bool,
    flash: bool,
):
    if flash:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
        ):
            output = F.scaled_dot_product_attention(
                query=query.contiguous(),
                key=key.contiguous(),
                value=value.contiguous(),
                attn_mask=None,
                is_causal=causal,
            )
    else:
        # implementation without flash assumes other dim order
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        a = torch.einsum("... l h d, ... L h d -> ... h l L", query, key)
        a = a * (1 / dhead**0.5)
        if causal:
            a.masked_fill_(
                torch.tril(torch.ones_like(a)) == 0, float("-inf")
            )  # mask out future tokens
        a = torch.softmax(a, dim=-1)
        output = torch.einsum("... h l L, ... L h d -> ... l h d", a, value)
        output = output.transpose(1, 2)

    return output


class AttentionMechanism(nn.Module):
    def __init__(self, use_flash_attention: bool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_flash_attention = use_flash_attention

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dhead: int,
        causal: bool,
        *args,
        **kwargs,
    ):
        return attention_mechanism(
            query=query,
            key=key,
            value=value,
            dhead=dhead,
            causal=causal,
            flash=self.use_flash_attention,
        )
    

class ProjectedAttention(LoggingLayer):
    def __init__(
        self,
        dmodel, # xs
        projected_dmodel, # xb
        heads,
        causal,
        init_type: str,
        init_scale: float,
        dhead=None,
        flash=False,
    ):
        """
            P1 = torch.rand(xs, xb)
            W = torch.rand(xb, yb)  
            P2 = torch.rand(yb, ys)
            P1@W@P2 = (xs, ys)
        """
        super(ProjectedAttention, self).__init__()
        assert dhead is None
        if dhead is None:
            assert projected_dmodel % heads == 0
            assert dmodel % heads == 0
            projected_dhead = projected_dmodel // heads
            dhead = dmodel // heads

        self.heads = heads
        self.dhead = dhead
        self.causal = causal
        self.flash = flash

        self.input_projection = nn.Sequential(
            Linear(
                dmodel, # xs
                projected_dmodel, # xb
                bias=False,
                init_type=init_type,
                init_scale=init_scale,
            ),
            Linear(
                projected_dmodel, # xb
                3 * heads * projected_dhead, # yb
                bias=False,
                init_type=init_type,
                init_scale=init_scale,
            ),
            Linear(
                3 * heads * projected_dhead, #yb
                3 * heads * dhead, #ys
                bias=False,
                init_type=init_type,
                init_scale=init_scale,
            ),
        )
            
        self.output_projection = nn.Sequential(
            Linear(
                heads * dhead, # xs
                heads * projected_dhead, # xb
                bias=False,
                init_type=init_type,
                init_scale=init_scale,
            ),
            Linear(
                heads * projected_dhead, # xb
                projected_dmodel, # yb
                bias=False,
                init_type=init_type,
                init_scale=init_scale,
            ),
            Linear(
                projected_dmodel, # yb
                dmodel, # ys
                bias=False,
                init_type=init_type,
                init_scale=init_scale,
            ),
            
        )

        self.attention_mechanism = AttentionMechanism(use_flash_attention=flash)

    def forward(self, x):
        projected = self.input_projection(x)

        batch, seq_len = x.shape[:-1]
        projected = projected.view(
            batch, seq_len, self.heads, 3 * self.dhead
        ).transpose(1, 2)
        q, k, v = torch.chunk(projected, chunks=3, dim=-1)

        attention_output = self.attention_mechanism(
            query=q, key=k, value=v, dhead=self.dhead, causal=self.causal
        )

        output = self.output_projection(attention_output.transpose(1, 2).flatten(-2))

        return output