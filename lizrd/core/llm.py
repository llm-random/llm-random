from collections import OrderedDict
from typing import Literal, Callable, Optional
from functools import partial

import torch

import lizrd.core.nn as nn
from lizrd.core import misc
from lizrd.core.misc import Checkpoint, default
from lizrd.support import ash


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
def FeedForward(
    dmodel,
    dff,
    bias: Literal["both", "first", "second", "none"] = "both",
):
    bias_first, bias_second = decode_bias_string(bias)

    return nn.Sequential(
        OrderedDict(
            [
                ("logging_ff_pre_relu", misc.Linear(dmodel, dff, bias=bias_first)),
                ("relu", nn.ReLU(inplace=True)),
                (
                    "logging_ff_post_relu",
                    misc.Linear(dff, dmodel, bias=bias_second),
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
class Residual(nn.Module):
    def __init__(self, layer):
        super(Residual, self).__init__()
        self.layer = layer

    def forward(self, x):
        out = self.layer(x)
        return out + x


@ash.check("... -> ... ")
class Parallel(nn.Module):
    def __init__(self, *layers):
        super(Parallel, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        return x + sum(layer(x) for layer in self.layers)


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
        misc.Linear(dinput, dlowrank, bias=False), misc.Linear(dlowrank, doutput)
    )


@ash.check("... d -> ... d")
class Attention(nn.Module):
    def __init__(self, dmodel, heads, dhead=None):
        super(Attention, self).__init__()
        if dhead is None:
            assert dmodel % heads == 0
            dhead = dmodel // heads

        self.heads = heads
        self.dhead = dhead
        self.dmodel = dmodel

        key_query_value_gen = lambda: misc.EinMix(
            "... dmodel -> ... heads dhead",
            weight_shape="dmodel heads dhead",
            bias_shape="heads dhead",
            dmodel=dmodel,
            heads=heads,
            dhead=dhead,
        )

        self.Q = key_query_value_gen()
        self.K = key_query_value_gen()
        self.V = key_query_value_gen()

        combine_gen = lambda: misc.EinMix(
            "... heads dhead -> ... dmodel",
            weight_shape="heads dhead dmodel",
            bias_shape="dmodel",
            dmodel=dmodel,
            heads=heads,
            dhead=dhead,
        )
        self.D = combine_gen()

    def forward(self, x):
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        a = torch.einsum("... l h d, ... L h d -> ... h l L", q, k)
        a = a * (1 / self.dhead**0.5)
        a = torch.softmax(a, dim=-1)
        prefinal = torch.einsum("... h l L, ... L h d -> ... l h d", a, v)
        output = self.D(prefinal)
        return output


@ash.check("... d -> ... d")
class CausalAttention(nn.Module):
    def __init__(self, dmodel, heads, dhead=None):
        super(CausalAttention, self).__init__()
        if dhead is None:
            assert dmodel % heads == 0
            dhead = dmodel // heads

        self.heads = heads
        self.dhead = dhead
        self.dmodel = dmodel

        key_query_value_gen = lambda: misc.EinMix(
            "... dmodel -> ... heads dhead",
            weight_shape="dmodel heads dhead",
            bias_shape="heads dhead",
            dmodel=dmodel,
            heads=heads,
            dhead=dhead,
        )

        self.Q = key_query_value_gen()
        self.K = key_query_value_gen()
        self.V = key_query_value_gen()

        combine_gen = lambda: misc.EinMix(
            "... heads dhead -> ... dmodel",
            weight_shape="heads dhead dmodel",
            bias_shape="dmodel",
            dmodel=dmodel,
            heads=heads,
            dhead=dhead,
        )
        self.D = combine_gen()

    def forward(self, x):
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        a = torch.einsum("... l h d, ... L h d -> ... h l L", q, k)
        a = a * (1 / self.dhead**0.5)
        a.masked_fill_(
            torch.tril(torch.ones_like(a)) == 0, float("-inf")
        )  # mask out future tokens
        a = torch.softmax(a, dim=-1)
        prefinal = torch.einsum("... h l L, ... L h d -> ... l h d", a, v)
        output = self.D(prefinal)
        return output


class ReZero(nn.Module):
    def __init__(self, fn, init=0.0):
        super().__init__()
        self.rezero_g = nn.Parameter(torch.tensor(init))
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.rezero_g


def RezeroBlock(dmodel, layer, name):
    return Residual(ReZero(layer))


def PostNormBlock(dmodel, layer, name):
    return nn.Sequential(
        OrderedDict(
            [
                (f"{name}", Residual(layer)),
                ("post_norm", nn.LayerNorm(dmodel)),
            ]
        )
    )


def PreNormBlock(dmodel, layer, name):
    return Residual(
        nn.Sequential(
            OrderedDict(
                [
                    ("pre_norm", nn.LayerNorm(dmodel)),
                    (f"{name}", layer),
                ]
            )
        )
    )


@ash.check("... d -> ... d")
def TransformerBlock(dmodel, layers, gradient_checkpointing, residual_fn):
    residual_fn = default(residual_fn, partial(PreNormBlock, dmodel=dmodel))
    residual_layers = [residual_fn(layer=layer, name=name) for name, layer in layers]
    if gradient_checkpointing:
        residual_layers = [Checkpoint(layer) for layer in residual_layers]
    return nn.Sequential(*residual_layers)


@ash.check("... d -> ... d")
class TransformerTower(nn.Module):
    def __init__(
        self,
        n_blocks,
        dmodel,
        layer_dict,
        gradient_checkpointing: bool = False,
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
            _, current_device = self.get_current_device(i_block)
            name_and_block = (
                f"block_{i_block}",
                TransformerBlock(
                    dmodel, layers_info, gradient_checkpointing, residual_fn
                ).to(current_device),
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
def TokenEmbedding(vocab_size, embedding_dim):
    return nn.Embedding(vocab_size, embedding_dim)


@ash.check("... -> ... d")
class PositionalEmbedding(nn.Module):
    def __init__(self, max_length, embedding_dim):
        super(PositionalEmbedding, self).__init__()
        self.layer = nn.Embedding(max_length, embedding_dim)
        # TODO(jaszczur): add initialization as positional encoding

    def forward(self, x):
        positions = torch.arange(0, x.shape[-1], device=x.device)
        positions = positions * torch.ones_like(x)
        embeddings = self.layer(positions)
        return embeddings


@ash.check("... -> ... d")
def EmbeddingLayer(*layers):
    return misc.Sum(*layers)


@ash.check("... inp -> ... out")
def PredictionHead(embedding_dim, output_size):
    return nn.Linear(embedding_dim, output_size)


@ash.check("... -> ... out")
class LLM(nn.Module):
    def __init__(self, embedding_layer, encoder_tower, head):
        super(LLM, self).__init__()

        self.encoder = nn.Sequential(
            OrderedDict(
                [
                    ("embedding_layer", embedding_layer),
                    ("encoder", encoder_tower),
                ]
            )
        )
        self.full_model = nn.Sequential(
            OrderedDict(
                [
                    ("embedding_layer", embedding_layer),
                    ("encoder", encoder_tower),
                    ("head", head),
                ]
            )
        )
        self.head = head

    def forward(self, *args, **kwargs):
        return self.full_model.forward(*args, **kwargs)
