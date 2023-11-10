from collections import OrderedDict
from typing import Literal, Callable, Optional
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from lizrd.core import misc
from lizrd.core.misc import default, Aggregate
from lizrd.core.initialization import get_init_weight
from lizrd.core.misc import Checkpoint, Linear
from lizrd.core.distributed import wrap_in_fsdp
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
                ("relu", nn.ReLU(inplace=True)),
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
        attn_in_high_precision: bool = False,
        rank: Optional[int] = None,
        param_precision: Optional[torch.dtype] = None,
        offload_params: bool = False,
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
        attention_mechanism = AttentionMechanism(use_flash_attention=flash)
        if attn_in_high_precision:
            self.attention_mechanism = wrap_in_fsdp(
                module=attention_mechanism,
                rank=rank,
                param_precision=torch.float32,
                offload_params=offload_params,
                cast_inputs=True,
                output_cast_dtype=param_precision,
            )
        self.attention_mechanism = attention_mechanism

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


class TransformerBlock(nn.Sequential):
    def __init__(
        self,
        dmodel,
        layers,
        gradient_checkpointing,
        residual_fn,
        fsdp_wrap_attn_and_ff=False,
        rank=None,
        fsdp_param_precision=torch.float32,
        fsdp_cpu_offloading=False,
    ):
        residual_fn = default(residual_fn, partial(PreNormBlock, dmodel=dmodel))

        residual_layers = []
        for name, layer in layers:
            module = residual_fn(layer=layer, name=name)
            if fsdp_wrap_attn_and_ff:
                wrap_in_fsdp(
                    rank=rank,
                    module=residual_fn(layer=layer, name=name),
                    param_precision=fsdp_param_precision,
                    offload_params=fsdp_cpu_offloading,
                )
            residual_layers.append(module)

        if gradient_checkpointing:
            residual_layers = [Checkpoint(layer) for layer in residual_layers]
        super(TransformerBlock, self).__init__(*residual_layers)


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
        rank=None,
        fsdp_wrap_whole_transformer_blocks=False,
        fsdp_wrap_attn_and_ff=False,
        fsdp_param_precision=torch.float32,
        fsdp_cpu_offloading=False,
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
                gradient_checkpointing,
                residual_fn,
                fsdp_wrap_attn_and_ff=fsdp_wrap_attn_and_ff,
                rank=rank,
                fsdp_param_precision=fsdp_param_precision,
                fsdp_cpu_offloading=fsdp_cpu_offloading,
            )
            if current_device != torch.device("cpu"):
                block = block.to(current_device)

            if fsdp_wrap_whole_transformer_blocks:
                block = wrap_in_fsdp(
                    module=block,
                    rank=rank,
                    param_precision=fsdp_param_precision,
                    offload_params=fsdp_cpu_offloading,
                )
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
    def __init__(self, embedding_dim, output_size, init_type, init_scale):
        super(PredictionHead, self).__init__(
            embedding_dim, output_size, init_type=init_type, init_scale=init_scale
        )


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
