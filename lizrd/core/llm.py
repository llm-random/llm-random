import random
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.models.utils import Collator, pad_and_concat
from tqdm.auto import tqdm

from lizrd.core import misc
from lizrd.core.initialization import ValidInitType, get_init_weight
from lizrd.core.misc import Aggregate, Linear, LoggingLayer, default


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


class Parallel(nn.Module):
    def __init__(self, *layers):
        super(Parallel, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        return sum(layer(x) for layer in self.layers)


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


class MergeLastAxis(nn.Module):
    def forward(self, x):
        result = x.reshape(x.shape[:-2] + (-1,))
        # print('wtf', x.shape, result.shape)
        return result


class Transpose(nn.Module):
    def forward(self, x):
        # return einops.rearrange(x, '... a b -> ... b a')
        return torch.transpose(x, -1, -2)


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
            embedding_dim, output_size, init_type=init_type, init_scale=init_scale
        )


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


class HarnessLM(TemplateLM):
    def __init__(self, model: LLM, batch_size: int, tokenizer, max_length: int, device):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def eot_token_id(self):
        return self.tokenizer.eot_id

    def generate_until(self, x: list[Instance]):
        raise NotImplementedError

    def loglikelihood_rolling(self, x: list[Instance]):
        raise NotImplementedError

    def tok_encode(self, s: str) -> List[int]:
        return [self.eot_token_id()] + self.tokenizer.text_to_ids(s)

    def _select_cont_toks(
        self, logits: torch.Tensor, contlen: int = None, inplen: int = None
    ) -> torch.Tensor:
        assert (
            contlen and inplen
        ), "Must pass input len and cont. len to select scored logits for causal LM"
        # discard right-padding.
        # also discard the input/context tokens. we'll only score continuations.
        logits = logits[inplen - contlen : inplen]

        return logits

    # _loglikelihood_tokens,
    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: int = None,
    ) -> List[Tuple[float, bool]]:
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        def _lookup_one_token_cont(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key to group and lookup one-token continuations"""
            # Use with group_by="contexts" (optional)"
            # allows for the creation of a lookup, so we can reuse logits in case of one-token continuations.
            # speeds up some multiple-choice tasks proportionally to the number of choices.
            # groups requests by context+continuation[:-1] and infer on one request/group.
            return req[-2] + req[-1][:-1]

        n_padding_requests = 0
        if len(requests) % self.batch_size != 0:
            print("Warning: batch size is not a multiple of the number of requests")
            n_padding_requests = self.batch_size - (len(requests) % self.batch_size)
            for _ in range(n_padding_requests):
                requests.append(random.choice(requests))

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            # group_by="contexts",
            group_by=None,
            group_fn=_lookup_one_token_cont,
        )

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        n_reordered_requests = len(re_ord)
        batch_size = self.batch_size
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto"
            and n_reordered_requests > 0
            and not override_bs
            else None
        )

        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )
        sum_of_chunks = 0
        for chunk in chunks:
            print(f"{sum_of_chunks} / {len(requests)}")
            sum_of_chunks += len(chunk)
            inps = []
            cont_toks_list = []
            inplens = []

            conts = []
            encoder_attns = []

            padding_len_inp = None
            padding_len_cont = None
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                    device=self.device,
                )
                (inplen,) = inp.shape

                # build encoder attn masks
                encoder_attns.append(torch.ones_like(inp))

                cont = torch.tensor(
                    (continuation_enc)[-self.max_length :],
                    # TODO: left-shift these?
                    # TODO: our code assumes we never end up truncating conts for either model type
                    dtype=torch.long,
                    device=self.device,
                )
                (contlen,) = cont.shape

                conts.append(cont)

                padding_len_cont = (
                    max(padding_len_cont, contlen)
                    if padding_len_cont is not None
                    else contlen
                )

                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )

                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            # create encoder attn mask and batched conts, if seq2seq
            call_kwargs = {}
            batched_inps = pad_and_concat(
                padding_len_inp, inps, padding_side="right"
            )  # [batch, padding_len_inp]

            # multi_logits = F.log_softmax(
            #     self._model_call(batched_inps, **call_kwargs), dim=-1
            # )  # [batch, padding_length (inp or cont), vocab]
            multi_logits = F.log_softmax(
                self.model(batched_inps, **call_kwargs), dim=-1
            )  # [batch, padding_length (inp or cont), vocab]

            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = inplen + (logits.shape[0] - padding_len_inp)
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)

                # check for one-token continuation cache hits.
                # noop in case group_by != "contexts" or no cache hit and returns the
                # original args. Otherwise, expands the logits batch dimension and yields each
                # batch along with matching continuation tokens and prompt strings.
                # logits -> [1, seq, vocab]
                for request_str, cont_toks, logits in re_ord.get_cache(
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):
                    cont_toks = torch.tensor(
                        cont_toks, dtype=torch.long, device=self.device
                    ).unsqueeze(
                        0
                    )  # [1, seq]
                    max_equal = (greedy_tokens == cont_toks).all()

                    # Obtain log-probs at the corresponding continuation token indices
                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                        -1
                    )  # [1, seq]

                    # Answer: (log prob, is-exact-match)
                    answer = (float(logits.sum()), bool(max_equal))

                    res.append(answer)

                    self.cache_hook.add_partial("loglikelihood", request_str, answer)
                    pbar.update(1)

        pbar.close()

        res = re_ord.get_original(res)
        if n_padding_requests > 0:
            res = res[:-n_padding_requests]
        return res
