from collections import OrderedDict
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from lizrd.core.initialization import ValidInitType
from lizrd.core.llm import Residual, RoPE
from lizrd.core.misc import Linear, LoggingLayer
from lizrd.core.initialization import get_init_weight, ValidInitType


def ProjectedTokenEmbedding(
    vocab_size,
    embedding_dim,
    projected_embedding_dim,
    init_type: ValidInitType,
    init_scale: float,
):
    weight = get_init_weight(
        shape=(vocab_size, projected_embedding_dim),
        fan_in=1,  # fan_in=1 is also default in pytorch
        init_type=init_type,
        scale=init_scale,
    )

    return nn.Sequential(
        OrderedDict([
                (
                    "embedding",
                    nn.Embedding(vocab_size, projected_embedding_dim, _weight=weight)
                ),
                (
                    "embedding_p",
                    Linear(
                        projected_embedding_dim, #yb
                        embedding_dim, #ys
                        bias=False,
                        init_type=init_type,
                        init_scale=init_scale,
                    ),
                )
            ])
    )

class ProjectedTokenEmbeddingRes(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        projected_embedding_dim,
        init_type: ValidInitType,
        init_scale: float,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        weight = get_init_weight(
            shape=(vocab_size, projected_embedding_dim),
            fan_in=1,  # fan_in=1 is also default in pytorch
            init_type=init_type,
            scale=init_scale,
        )
        self.embedding = nn.Sequential(
            OrderedDict([
                    (
                        "embedding",
                        nn.Embedding(vocab_size, projected_embedding_dim, _weight=weight)
                    ),
                    (
                        "embedding_p",
                        Linear(
                            projected_embedding_dim, #yb
                            embedding_dim, #ys
                            bias=False,
                            init_type=init_type,
                            init_scale=init_scale,
                        ),
                    )
                ])
        )
        weight_res = get_init_weight(
            shape=(vocab_size, embedding_dim),
            fan_in=None,  # fan_in=1 is also default in pytorch
            init_type="zeros",
            scale=None,
        )
        self.embedding_res = nn.Embedding(vocab_size, embedding_dim, _weight=weight_res)

    def forward(self, x):
        h1 = self.embedding(x)
        h2 = self.embedding_res(x)
        return h1 + h2


class ProjectedPositionalEmbedding(nn.Module):
    def __init__(
        self,
        max_length,
        embedding_dim,
        projected_embedding_dim,
        init_type: ValidInitType,
        init_scale: float, 
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        weight = get_init_weight(
            shape=(max_length, projected_embedding_dim),
            fan_in=1,
            init_type=init_type,
            scale=init_scale,
        )
        self.projected_layer = nn.Sequential(
            OrderedDict([
                    (
                        "pe_layer",
                        nn.Embedding(max_length, projected_embedding_dim, _weight=weight),
                    ),
                    (
                        "pe_layer_p",
                        Linear(
                            projected_embedding_dim, #yb
                            embedding_dim, #ys
                            bias=False,
                            init_type=init_type,
                            init_scale=init_scale,
                        ),
                    )
                ])
        )

    def forward(self, x):
        positions = torch.arange(0, x.shape[-1], device=x.device)
        positions = positions * torch.ones_like(x)
        embeddings = self.projected_layer(positions)
        return embeddings
    

class ProjectedPositionalEmbeddingRes(nn.Module):
    def __init__(
        self,
        max_length,
        embedding_dim,
        projected_embedding_dim,
        init_type: ValidInitType,
        init_scale: float,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        weight = get_init_weight(
            shape=(max_length, projected_embedding_dim),
            fan_in=1,
            init_type=init_type,
            scale=init_scale,
        )
        self.projected_layer = nn.Sequential(
            OrderedDict([
                    (
                        "pe_layer",
                        nn.Embedding(max_length, projected_embedding_dim, _weight=weight),
                    ),
                    (
                        "pe_layer_p",
                        Linear(
                            projected_embedding_dim, #yb
                            embedding_dim, #ys
                            bias=False,
                            init_type=init_type,
                            init_scale=init_scale,
                        ),
                    )
                ])
        )
        weight_res = get_init_weight(
            shape=(max_length, embedding_dim),
            fan_in=None,
            init_type="zeros",
            scale=None,
        )
        self.projected_layer_res = nn.Embedding(max_length, embedding_dim, _weight=weight_res)

    def forward(self, x):
        positions = torch.arange(0, x.shape[-1], device=x.device)
        positions = positions * torch.ones_like(x)
        embeddings = self.projected_layer(positions) + self.projected_layer_res(positions)
        return embeddings


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
    bias: Literal["both", "first", "second", "none"] = "none",
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
        self.projected_dhead = projected_dhead
        
        self.input_projection_q = nn.Sequential(
            OrderedDict([
                ("input_projection",
                Linear(
                    dmodel, # xs
                    heads * projected_dhead, # xb
                    bias=False,
                    init_type=init_type,
                    init_scale=init_scale,
                )),
                ("projected_weight",
                Linear(
                    projected_dmodel, # xb
                    heads * projected_dhead, # yb
                    bias=False,
                    init_type=init_type,
                    init_scale=init_scale,
                )),
                ("output_projection",
                Linear(
                    projected_dmodel, # xb
                    dmodel, # xs
                    bias=False,
                    init_type=init_type,
                    init_scale=init_scale,
                ))
            ])
        )

        self.input_projection_k = nn.Sequential(
            OrderedDict([
                ("input_projection",
                Linear(
                    dmodel, # xs
                    heads * projected_dhead, # xb
                    bias=False,
                    init_type=init_type,
                    init_scale=init_scale,
                )),
                ("projected_weight",
                Linear(
                    projected_dmodel, # xb
                    heads * projected_dhead, # yb
                    bias=False,
                    init_type=init_type,
                    init_scale=init_scale,
                )),
                ("output_projection",
                Linear(
                    projected_dmodel, # xb
                    dmodel, # xs
                    bias=False,
                    init_type=init_type,
                    init_scale=init_scale,
                ))
            ])
        )

        self.input_projection_v = nn.Sequential(
            OrderedDict([
                ("input_projection",
                Linear(
                    dmodel, # xs
                    heads * projected_dhead, # xb
                    bias=False,
                    init_type=init_type,
                    init_scale=init_scale,
                )),
                ("projected_weight",
                Linear(
                    projected_dmodel, # xb
                    heads * projected_dhead, # yb
                    bias=False,
                    init_type=init_type,
                    init_scale=init_scale,
                )),
                ("output_projection",
                Linear(
                    projected_dmodel, # xb
                    dmodel, # xs
                    bias=False,
                    init_type=init_type,
                    init_scale=init_scale,
                ))
            ])
        )

        self.output_projection = nn.Sequential(
            OrderedDict([
                ("output_projection_p21",
                Linear(
                    heads * dhead, # xs
                    heads * projected_dhead, # xb
                    bias=False,
                    init_type=init_type,
                    init_scale=init_scale,
                )),
                ("output_projection",
                Linear(
                    heads * projected_dhead, # xb
                    projected_dmodel, # yb
                    bias=False,
                    init_type=init_type,
                    init_scale=init_scale,
                )),
                ("output_projection_p22",
                Linear(
                    projected_dmodel, # yb
                    dmodel, # ys
                    bias=False,
                    init_type=init_type,
                    init_scale=init_scale,
                )),
            ])
        )

        self.attention_mechanism = AttentionMechanism(use_flash_attention=flash)

    def forward(self, x):
        q = self.input_projection_q(x)
        k = self.input_projection_k(x)
        v = self.input_projection_v(x)

        projected = torch.concat((q,k,v), dim=-1)

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

def PreNormNoBiasBlock(dmodel, layer, name, norm_class=nn.LayerNorm):
    return Residual(
        nn.Sequential(
            OrderedDict(
                [
                    ("pre_norm", norm_class(dmodel, bias=False)),
                    (f"{name}", layer),
                ]
            )
        )
    )


class ProjectedAttentionRes(LoggingLayer):
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
        super(ProjectedAttentionRes, self).__init__()
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
        self.projected_dhead = projected_dhead
        
        self.input_projection_q = nn.Sequential(
            OrderedDict([
                ("input_projection",
                Linear(
                    dmodel, # xs
                    heads * projected_dhead, # xb
                    bias=False,
                    init_type=init_type,
                    init_scale=init_scale,
                )),
                ("projected_weight",
                Linear(
                    projected_dmodel, # xb
                    heads * projected_dhead, # yb
                    bias=False,
                    init_type=init_type,
                    init_scale=init_scale,
                )),
                ("output_projection",
                Linear(
                    projected_dmodel, # xb
                    dmodel, # xs
                    bias=False,
                    init_type=init_type,
                    init_scale=init_scale,
                ))
            ])
        )
        
        self.input_projection_k = nn.Sequential(
            OrderedDict([
                ("input_projection",
                Linear(
                    dmodel, # xs
                    heads * projected_dhead, # xb
                    bias=False,
                    init_type=init_type,
                    init_scale=init_scale,
                )),
                ("projected_weight",
                Linear(
                    projected_dmodel, # xb
                    heads * projected_dhead, # yb
                    bias=False,
                    init_type=init_type,
                    init_scale=init_scale,
                )),
                ("output_projection",
                Linear(
                    projected_dmodel, # xb
                    dmodel, # xs
                    bias=False,
                    init_type=init_type,
                    init_scale=init_scale,
                ))
            ])
        )

        self.input_projection_v = nn.Sequential(
            OrderedDict([
                ("input_projection",
                Linear(
                    dmodel, # xs
                    heads * projected_dhead, # xb
                    bias=False,
                    init_type=init_type,
                    init_scale=init_scale,
                )),
                ("projected_weight",
                Linear(
                    projected_dmodel, # xb
                    heads * projected_dhead, # yb
                    bias=False,
                    init_type=init_type,
                    init_scale=init_scale,
                )),
                ("output_projection",
                Linear(
                    projected_dmodel, # xb
                    dmodel, # xs
                    bias=False,
                    init_type=init_type,
                    init_scale=init_scale,
                ))
            ])
        )
        self.input_projection_q_res = Linear(
            dmodel, # xs
            heads * dhead, # ys
            bias=False,
            init_type="zeros",
            init_scale=None,
        )
        self.input_projection_k_res = Linear(
            dmodel, # xs
            heads * dhead, # ys
            bias=False,
            init_type="zeros",
            init_scale=None,
        )
        self.input_projection_v_res = Linear(
            dmodel, # xs
            heads * dhead, # ys
            bias=False,
            init_type="zeros",
            init_scale=None,
        )

        self.output_projection = nn.Sequential(
            OrderedDict([
                ("output_projection_p21",
                Linear(
                    heads * dhead, # xs
                    heads * projected_dhead, # xb
                    bias=False,
                    init_type=init_type,
                    init_scale=init_scale,
                )),
                ("output_projection",
                Linear(
                    heads * projected_dhead, # xb
                    projected_dmodel, # yb
                    bias=False,
                    init_type=init_type,
                    init_scale=init_scale,
                )),
                ("output_projection_p22",
                Linear(
                    projected_dmodel, # yb
                    dmodel, # ys
                    bias=False,
                    init_type=init_type,
                    init_scale=init_scale,
                )),
            ])
        )
        self.output_projection_res = Linear(
            heads * dhead, # xs
            dmodel, # ys
            bias=False,
            init_type="zeros",
            init_scale=None,
        )

        self.attention_mechanism = AttentionMechanism(use_flash_attention=flash)

    def forward(self, x):
        q = self.input_projection_q(x) + self.input_projection_q_res(x)
        k = self.input_projection_k(x) + self.input_projection_k_res(x)
        v = self.input_projection_v(x) + self.input_projection_v_res(x)


        projected = torch.concat((q,k,v), dim=-1)

        batch, seq_len = x.shape[:-1]
        projected = projected.view(
            batch, seq_len, self.heads, 3 * self.dhead
        ).transpose(1, 2)
        q, k, v = torch.chunk(projected, chunks=3, dim=-1)

        attention_output = self.attention_mechanism(
            query=q, key=k, value=v, dhead=self.dhead, causal=self.causal
        )

        to_output = attention_output.transpose(1, 2).flatten(-2)
        output = self.output_projection(to_output) + self.output_projection_res(to_output)

        return output
    

class ClassProejectedFeedForwardRes(nn.Module):
    def __init__(
        self,
        dmodel,
        dff,
        projected_dmodel,
        projected_dff,
        init_type: ValidInitType,
        init_scale: float,
        bias_first,
        bias_second, 
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.ff_in = nn.Sequential(
            OrderedDict([
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
                )
            ])
        )
        self.act_fun = nn.ReLU()
        self.ff_out = nn.Sequential(
            OrderedDict(
                [
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

        self.ff_in_res = Linear(
            dmodel, # xs
            dff, # ys
            bias=False,
            init_type="zeros",
            init_scale=None,
        )

        self.ff_out_res = Linear(
            dff, # xs
            dmodel, # ys
            bias=False,
            init_type="zeros",
            init_scale=None,
        )

    
    def forward(self, x):
        h = self.ff_in(x) + self.ff_in_res(x)
        h = self.act_fun(h)
        h = self.ff_out(h) + self.ff_out_res(h)
        return h



def ProjectedFeedForwardRes( #dev
    dmodel,
    dff,
    projected_dmodel,
    projected_dff,
    init_type: ValidInitType,
    init_scale: float,
    bias: Literal["both", "first", "second", "none"] = "none",
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
    return ClassProejectedFeedForwardRes(dmodel, dff, projected_dmodel, projected_dff, init_type, init_scale,bias_first, bias_second)


class PredictionHeadRes(nn.Module):
    def __init__(self, projected_dmodel, vocab_size, dm, init_type, init_scale, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = torch.nn.Sequential(
            OrderedDict([
                (
                    "head_p",
                    Linear(
                        dm, #xs
                        projected_dmodel, #xb
                        bias=False,
                        init_type=init_type,
                        init_scale=init_scale,
                    ),
                ),
                (
                    "head",
                    Linear( 
                        projected_dmodel, 
                        vocab_size, 
                        init_type=init_type, 
                        init_scale=init_scale
                    ),
                )
            ])
        )

        self.head_res = Linear(
            dm, # xs
            vocab_size, # ys
            bias=False,
            init_type="zeros",
            init_scale=None,
        )
    
    def forward(self, x):
        return self.head(x) + self.head_res(x)


