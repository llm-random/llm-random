from collections import OrderedDict
from typing import Literal, Callable, Optional
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from lizrd.core import misc
from lizrd.core.misc import default, Aggregate
from lizrd.core.initialization import get_init_weight
from lizrd.core.misc import Linear
from lizrd.support import ash
from research.conditional.utils.layer_manager import LoggingLayer


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


@ash.check("... d -> ... d")
class SwiGLUFeedForward(LoggingLayer):
    def __init__(
        self,
        dmodel,
        dff,
        init_type: Literal["kaiming_uniform", "truncated_normal"],
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


@ash.check("... d -> ... d")
def FeedForward(
    dmodel,
    dff,
    init_type: Literal["kaiming_uniform", "truncated_normal"],
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


class EveryOtherLayer:
    def __init__(
        self, layer1_fn: Callable[[], nn.Module], layer2_fn: Callable[[], nn.Module]
    ):
        """
        This class is used to alternate between two layers.
        It is useful for Mixture of Experts,
        where every other layer is a regular linear layer.
        """
        self.layer1_fn = layer1_fn
        self.layer2_fn = layer2_fn
        self.counter = 0

    def __call__(self):
        if self.counter % 2 == 0:
            layer = self.layer1_fn()
        else:
            layer = self.layer2_fn()
        self.counter += 1
        return layer


@ash.check("... -> ... ")
class Residual(LoggingLayer):
    def __init__(self, layer):
        super(Residual, self).__init__()
        self.layer = layer

    def forward(self, x):
        out = self.layer(x)
        self.update_cache_for_logging("update", out)
        self.update_cache_for_logging("residual_stream", x)
        return out + x

    def log_heavy(self):
        updates = self.logging_cache["update"]
        residual_stream = self.logging_cache["residual_stream"]

        update_norms = torch.norm(updates, dim=-1)
        residual_norms = torch.norm(residual_stream, dim=-1)

        update_norms_mean = torch.mean(update_norms)
        update_norms_std = torch.std(update_norms)
        residual_norms_mean = torch.mean(residual_norms)
        residual_norms_std = torch.std(residual_norms)

        update_to_residual_ratio = update_norms / residual_norms
        update_to_residual_ratio_mean = torch.mean(update_to_residual_ratio)
        update_to_residual_ratio_std = torch.std(update_to_residual_ratio)

        return {
            "update_norms/mean": update_norms_mean,
            "update_norms/std": update_norms_std,
            "residual_norms/mean": residual_norms_mean,
            "residual_norms/std": residual_norms_std,
            "update_to_residual_ratio/mean": update_to_residual_ratio_mean,
            "update_to_residual_ratio/std": update_to_residual_ratio_std,
        }


@ash.check("... -> ... ")
class Parallel(nn.Module):
    def __init__(self, *layers):
        super(Parallel, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        return sum(layer(x) for layer in self.layers)


@ash.check("... dinp -> ... a b")
class SplitLastAxis(nn.Module):
    def __init__(self, a, b):
        super(SplitLastAxis, self).__init__()
        self.a = a
        self.b = b

    def forward(self, x):
        a, b = self.a, self.b
        assert x.shape[-1] == a * b
        result = x.view(x.shape[:-1] + (a, b))
        assert result.shape[-2:] == (a, b)
        # print("wtf", x.shape, result.shape)
        return result


@ash.check("... a b -> ... dout")
class MergeLastAxis(nn.Module):
    def forward(self, x):
        result = x.reshape(x.shape[:-2] + (-1,))
        # print('wtf', x.shape, result.shape)
        return result


@ash.check("... a b -> ... b a")
class Transpose(nn.Module):
    def forward(self, x):
        # return einops.rearrange(x, '... a b -> ... b a')
        return torch.transpose(x, -1, -2)


@ash.check("... dinp -> ... dout")
def LowRank(dinput, doutput, dlowrank):
    return nn.Sequential(
        Linear(dinput, dlowrank, bias=False),
        Linear(dlowrank, doutput),
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
                query=query,
                key=key,
                value=value,
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


@ash.check("... d -> ... d")
class Attention(LoggingLayer):
    def __init__(
        self,
        dmodel,
        heads,
        causal,
        init_type: str,
        init_scale: float,
        dhead=None,
        flash=False,
    ):
        super(Attention, self).__init__()
        if dhead is None:
            assert dmodel % heads == 0
            dhead = dmodel // heads

        self.heads = heads
        self.dhead = dhead
        self.causal = causal
        self.flash = flash

        self.input_projection = Linear(
            dmodel,
            3 * heads * dhead,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.output_projection = Linear(
            heads * dhead,
            dmodel,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
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


class RoPE(nn.Module):
    # features are paired x_i, x_{i + d_head/2}
    def __init__(self, dhead, length):
        super().__init__()
        self.dhead = dhead
        self.length = length
        angle_exponents = torch.arange(0, dhead, 2) / dhead
        angles = torch.pow(1 / 10000, angle_exponents).reshape(1, -1)
        angle_per_token = angles * torch.arange(0, length).reshape(-1, 1)
        self.register_buffer("sin", torch.sin(angle_per_token).repeat(1, 2))
        self.register_buffer("cos", torch.cos(angle_per_token).repeat(1, 2))

    def forward(self, x):
        [y1, y2] = torch.chunk(x, chunks=2, dim=-1)
        x_rotated = torch.cat([-y2, y1], dim=-1)
        return x * self.cos + x_rotated * self.sin


@ash.check("... d -> ... d")
class AttentionRoPE(LoggingLayer):
    def __init__(
        self,
        dmodel,
        heads,
        causal,
        length,
        init_type: str,
        init_scale: float,
        dhead=None,
        flash=False,
    ):
        super(AttentionRoPE, self).__init__()
        if dhead is None:
            assert dmodel % heads == 0
            dhead = dmodel // heads

        self.heads = heads
        self.dhead = dhead
        self.causal = causal
        self.flash = flash

        self.input_projection = Linear(
            dmodel,
            3 * heads * dhead,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.output_projection = Linear(
            heads * dhead,
            dmodel,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.rope = RoPE(dhead, length=length)
        self.attention_mechanism = AttentionMechanism(use_flash_attention=flash)

    def forward(self, x):
        projected = self.input_projection(x)

        batch, seq_len = x.shape[:-1]
        projected = projected.view(
            batch, seq_len, self.heads, 3 * self.dhead
        ).transpose(1, 2)
        q, k, v = torch.chunk(projected, chunks=3, dim=-1)
        q = self.rope(q)
        k = self.rope(k)

        attention_output = self.attention_mechanism(
            query=q, key=k, value=v, dhead=self.dhead, causal=self.causal
        )

        output = self.output_projection(attention_output.transpose(1, 2).flatten(-2))

        return output


@ash.check("... d -> ... d")
class Attention(LoggingLayer):
    def __init__(
        self,
        dmodel,
        heads,
        causal,
        init_type: str,
        init_scale: float,
        dhead=None,
        flash=False,
    ):
        super(Attention, self).__init__()
        if dhead is None:
            assert dmodel % heads == 0
            dhead = dmodel // heads

        self.heads = heads
        self.dhead = dhead
        self.causal = causal
        self.flash = flash

        self.input_projection = Linear(
            dmodel,
            3 * heads * dhead,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.output_projection = Linear(
            heads * dhead,
            dmodel,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
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


class RMSNorm(nn.Module):
    def __init__(self, dmodel, eps=1e-5):
        super().__init__()
        self.eps = eps

        self.g = nn.Parameter(torch.ones(dmodel))
        self.b = nn.Parameter(torch.zeros(dmodel))

    def forward(self, x):
        norm = torch.mean(x**2, dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.g + self.b


class ReZero(nn.Module):
    def __init__(self, fn, init=0.0):
        super().__init__()
        self.rezero_g = nn.Parameter(torch.tensor(init))
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.rezero_g


def RezeroBlock(dmodel, layer, name):
    return Residual(ReZero(layer))


def PostNormBlock(dmodel, layer, name, norm_class=nn.LayerNorm):
    return nn.Sequential(
        OrderedDict(
            [
                (f"{name}", Residual(layer)),
                ("post_norm", norm_class(dmodel)),
            ]
        )
    )


def ParallelPreNormBlock(dmodel, layer, name, norm_class=nn.LayerNorm):
    assert isinstance(layer, Parallel)
    layer.layers = nn.ModuleList(
        *[
            torch.nn.Sequential(
                OrderedDict(
                    [
                        ("pre_norm", norm_class(dmodel)),
                        (f"{type(module)}", module),
                    ]
                )
            )
            for module in layer.layers
        ]
    )
    return Residual(layer)


def PreNormBlock(dmodel, layer, name, norm_class=nn.LayerNorm):
    return Residual(
        nn.Sequential(
            OrderedDict(
                [
                    ("pre_norm", norm_class(dmodel)),
                    (f"{name}", layer),
                ]
            )
        )
    )


@ash.check("... d -> ... d")
class TransformerBlock(nn.Module):
    def __init__(self, dmodel, layers, residual_fn):
        super(TransformerBlock, self).__init__()

        residual_fn = default(residual_fn, partial(PreNormBlock, dmodel=dmodel))
        residual_layers = [
            (f"residual_{name}", residual_fn(layer=layer, name=name))
            for name, layer in layers
        ]
        self.block = nn.Sequential(OrderedDict(residual_layers))

    def forward(self, x):
        return self.block(x)


@ash.check("... d -> ... d")
class TransformerTower(nn.Module):
    def __init__(
        self,
        n_blocks,
        dmodel,
        layer_dict,
        device: torch.device = None,
        model_fragmentation: Optional[list[int]] = None,
        residual_fn: Optional[Callable] = None,
    ):
        super().__init__()
        misc.check_layer_funs(*layer_dict.values())
        self.blocks = []
        self.model_fragmentation = (
            [] if model_fragmentation is None else model_fragmentation
        )
        self.device = device

        for i_block in range(n_blocks):
            layers_info = [
                (name, layer_fun()) for name, layer_fun in layer_dict.items()
            ]

            for name, layer in layers_info:
                layer.layer_type = name
                layer.block_number = i_block

            _, current_device = self.get_current_device(i_block)
            block = TransformerBlock(
                dmodel,
                layers_info,
                residual_fn,
            )
            if current_device != torch.device("cpu"):
                block = block.to(current_device)

            name_and_block = (
                f"block_{i_block}",
                block,
            )
            self.blocks.append(name_and_block)
        self.blocks = nn.Sequential(OrderedDict(self.blocks))

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            should_transfer, current_device = self.get_current_device(i)
            if should_transfer:
                x = x.to(current_device)
            x = block(x)
        return x

    def get_current_device(self, block_num):
        if self.model_fragmentation is None or self.device == torch.device("cpu"):
            return False, self.device

        for i, split_num in enumerate(self.model_fragmentation):
            if split_num > block_num:
                return block_num in self.model_fragmentation, torch.device(f"cuda:{i}")

        return block_num in self.model_fragmentation, torch.device(
            f"cuda:{len(self.model_fragmentation)}"
        )


@ash.check("... -> ... d")
def TokenEmbedding(
    vocab_size,
    embedding_dim,
    init_type: Literal["kaiming_uniform", "truncated_normal"],
    init_scale: float,
):
    weight = get_init_weight(
        shape=(vocab_size, embedding_dim),
        fan_in=1,  # fan_in=1 is also default in pytorch
        init_type=init_type,
        scale=init_scale,
    )
    return nn.Embedding(vocab_size, embedding_dim, _weight=weight)


@ash.check("... -> ... d")
class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        max_length,
        embedding_dim,
        init_type: Literal["kaiming_uniform", "truncated_normal"],
        init_scale: float,
    ):
        super(PositionalEmbedding, self).__init__()
        self.layer = nn.Embedding(max_length, embedding_dim)
        default_weight = self.layer.weight.data
        self.layer.weight.data = get_init_weight(
            shape=default_weight.shape,
            fan_in=1,
            init_type=init_type,
            scale=init_scale,
            dtype=default_weight.dtype,
        )
        # TODO(jaszczur): add initialization as positional encoding

    def forward(self, x):
        positions = torch.arange(0, x.shape[-1], device=x.device)
        positions = positions * torch.ones_like(x)
        embeddings = self.layer(positions)
        return embeddings


class EmbeddingLayer(Aggregate):
    def __init__(self, *layers):
        super(EmbeddingLayer, self).__init__((lambda x, y: x + y), *layers)


class PredictionHead(Linear):
    def __init__(
        self, embedding_dim, output_size, init_type, init_scale, multiplier=1.0
    ):
        super(PredictionHead, self).__init__(
            embedding_dim, output_size, init_type=init_type, init_scale=init_scale
        )
        self.register_buffer("multiplier", torch.tensor(multiplier))

    def forward(self, x):
        return self.multiplier * super(PredictionHead, self).forward(x)


@ash.check("... -> ... out")
class LLM(nn.Module):
    def __init__(self, embedding_layer, encoder_tower, head):
        super(LLM, self).__init__()
        self.embedding_layer = embedding_layer
        self.encoder = encoder_tower
        self.head = head

    def forward(self, *args, **kwargs):
        x = self.embedding_layer(*args, **kwargs)
        x = self.encoder(x)
        x = self.head(x)
        return x
