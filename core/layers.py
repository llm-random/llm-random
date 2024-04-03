from collections import OrderedDict
from contextlib import contextmanager
from functools import wraps
import time
from typing import Literal, Union
import torch
from torch import nn
import torch.nn.functional as F
from lizrd.core.initialization import get_init_weight
from lizrd.core.misc import check_layer_funs


######## MeasuringLayer LAYER COPY ########


class MeasuringLayer(nn.Module):
    def __init__(self, layer, name, parent):
        super().__init__()
        self.l = layer
        self.name = name
        self.parent = [parent]

    def forward(self, *args, **kwargs):
        with measure_time(self.parent[0], self.name):
            return self.l(*args, **kwargs)


def time_measured(name):
    def _decorator(func):
        @wraps(func)
        def _decorator_wrapper(self, *args, **kwargs):
            with measure_time(self, name):
                return func(self, *args, **kwargs)

        return _decorator_wrapper

    return _decorator


######## END MeasuringLayer LAYER COPY ########


class CachingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # info about position in model
        self.layer_type: Union[str, None] = None
        self.block_number: Union[int, None] = None

        # caches for logging and propagation
        self.logging_cache = {}
        self.forward_pass_cache: Union[dict, None] = None

    def clean_up_after_logging(self):
        assert self.logging_switch
        self.logging_switch = False
        self.logging_cache = {}

    def prepare_for_logging(self):
        self.logging_switch = True

    def update_cache_for_logging(self, key, value):
        if isinstance(value, dict):
            if key in self.logging_cache:
                self.logging_cache[key].update(value)
            else:
                self.logging_cache[key] = value
        elif isinstance(value, torch.Tensor):
            self.logging_cache[key] = value.clone().detach().cpu()
        elif isinstance(value, float) or isinstance(value, int):
            self.logging_cache[key] = value
        else:
            raise NotImplementedError

    def _combine_to_dict_key(self, key, layer_type, block_number):
        return f"block_{block_number}_{layer_type}_{key}"

    def update_forward_pass_cache(self, key, value):
        combined_key = self._combine_to_dict_key(
            key, self.layer_type, self.block_number
        )
        self.forward_pass_cache[combined_key] = value

    def get_from_forward_pass_cache(self, key, block_number, layer_type):
        combined_key = self._combine_to_dict_key(key, layer_type, block_number)
        return self.forward_pass_cache[combined_key]

    def log(self, verbosity_level):
        if verbosity_level == 0:
            return self.log_time()
        elif verbosity_level == 1:
            return self.log_light()
        elif verbosity_level == 2:
            return self.log_heavy()
        else:
            raise Exception("Invalid verbosity level")

    def log_light(self):
        return {}

    def log_heavy(self):
        return {}

    def log_time(self):
        log = {}
        if "time" in self.logging_cache:
            instr_names = list(self.logging_cache["time"].keys())
            instr_times = list(self.logging_cache["time"].values())
            times_fig = px.bar(x=instr_names, y=instr_times)
            log["time"] = times_fig
        return log

    def measure(self, module, name, exists=True):
        if not exists:
            return nn.Identity()
        return MeasuringLayer(module, name, self)


######## measure_time LAYER COPY  (almost - CachingLayer) ########
@contextmanager
def measure_time(layer: CachingLayer, instruction_name: str):
    """
    This simple context manager is used to measure the time of a block of code.
    Args:
        layer: The LoggingLayer object that will be used to cache the time.
        instruction_name: The name of the instruction that is being measured.
    """
    if layer.logging_switch:
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start = time.time()
    yield
    if layer.logging_switch:
        if torch.cuda.is_available():
            end.record()
            torch.cuda.synchronize()
            layer.update_cache_for_logging(
                "time", {instruction_name: start.elapsed_time(end)}
            )
        else:
            end = time.time()
            layer.update_cache_for_logging("time", {instruction_name: end - start})


######## CORE LAYERS COPY (almost - CachingLayer) ########


class Linear(nn.Linear):
    def __init__(self, *args, init_type, init_scale, **kwargs):
        if "bias" not in kwargs:
            kwargs["bias"] = False
        super().__init__(*args, **kwargs)
        self.weight.data = get_init_weight(
            shape=self.weight.shape,
            fan_in=self.in_features,
            init_type=init_type,
            scale=init_scale,
            dtype=self.weight.dtype,
        )


class Residual(CachingLayer):
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


class Attention(CachingLayer):
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


class Aggregate(nn.Module):
    def __init__(self, function, *layers):
        super(Aggregate, self).__init__()
        self.function = function
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        result = None
        for layer in self.layers:
            if result is None:
                result = layer(x)
            else:
                result = self.function(result, layer(x))
        return result


class EmbeddingLayer(Aggregate):
    def __init__(self, *layers):
        super(EmbeddingLayer, self).__init__((lambda x, y: x + y), *layers)


def FeedForward(
    dmodel,
    dff,
    init_type: Literal["kaiming_uniform", "truncated_normal"],
    init_scale: float,
    bias: Literal["both", "first", "second", "none"] = "both",
):
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


class PredictionHead(Linear):
    def __init__(self, embedding_dim, output_size, init_type, init_scale):
        super(PredictionHead, self).__init__(
            embedding_dim, output_size, init_type=init_type, init_scale=init_scale
        )


class TransformerBlock(nn.Module):
    def __init__(self, layers, partial_norm_block):
        super(TransformerBlock, self).__init__()

        residual_layers = [
            (f"residual_{name}", partial_norm_block(layer=layer, name=name))
            for name, layer in layers
        ]
        self.block = nn.Sequential(OrderedDict(residual_layers))

    def forward(self, x):
        return self.block(x)


def TransformerTower(n_blocks, block_definition, norm_block_function):
    check_layer_funs(*block_definition.values())
    encoder_blocks = []
    for i_block in range(n_blocks):
        block_layers = [
            (name, layer_fun()) for name, layer_fun in block_definition.items()
        ]
        name_and_block = (
            f"block_{i_block}",
            TransformerBlock(block_layers, norm_block_function),
        )
        encoder_blocks.append(name_and_block)

    return nn.Sequential(OrderedDict(encoder_blocks))


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
