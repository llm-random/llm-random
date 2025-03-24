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


def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


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


class TransformerTower(nn.Module):
    def __init__(
        self,
        n_blocks,
        dmodel,
        layer_or_block_definition,
        device: torch.device = None,
        model_fragmentation: Optional[list[int]] = None,
        residual_fn: Optional[Callable] = None,
    ):
        super().__init__()
        if type(layer_or_block_definition) is dict:
            misc.check_layer_funs(*layer_or_block_definition.values())
        elif type(layer_or_block_definition) is list:
            for layer in layer_or_block_definition:
                misc.check_layer_funs(*layer)
            assert len(layer_or_block_definition) == n_blocks
        else:
            raise ValueError("layer_definition must be dict or list")

        self.blocks = []
        self.model_fragmentation = (
            [] if model_fragmentation is None else model_fragmentation
        )
        self.device = device

        if type(layer_or_block_definition) is dict:
            block_definitions = [layer_or_block_definition] * n_blocks
        else:
            block_definitions = layer_or_block_definition

        for i_block, layer_dict in enumerate(block_definitions):
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


def TokenEmbedding(
    vocab_size,
    embedding_dim,
    init_type: ValidInitType,
    init_scale: float,
):
    weight = get_init_weight(
        shape=(vocab_size, embedding_dim),
        fan_in=1,  # fan_in=1 is also default in pytorch
        init_type=init_type,
        scale=init_scale,
    )
    return nn.Embedding(vocab_size, embedding_dim, _weight=weight)


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        max_length,
        embedding_dim,
        init_type: ValidInitType,
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
    def __init__(self, embedding_dim, output_size, init_type, init_scale):
        super(PredictionHead, self).__init__(
            # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. (vide Karpathy)
            next_multiple_of_n(embedding_dim, n=128),
            output_size,
            init_type=init_type,
            init_scale=init_scale,
        )


class LLM(nn.Module):
    def __init__(self, embedding_layer, encoder_tower, final_norm, head):
        super(LLM, self).__init__()
        self.embedding_layer = embedding_layer
        self.encoder = encoder_tower
        self.head = head
        self.final_layernorm = final_norm

    def forward(self, *args, **kwargs):
        x = self.embedding_layer(*args, **kwargs)
        x = self.encoder(x)
        if self.final_layernorm:
            x = self.final_layernorm(x)

        x = self.head(x)
        return x
