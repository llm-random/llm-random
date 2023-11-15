from collections import OrderedDict
from typing import Callable, Literal, Optional
from lizrd.core import llm
import lizrd.core.nn as nn
from research.blanks.utils import (
    get_first_blanks_in_series,
    shift_left,
    shift_right,
    make_blanks_attention_mask,
)
import torch.nn.functional as F
from research.blanks.utils import get_first_blanks_in_series, shift_left, shift_right

import torch


def get_model(
    max_length: int,
    vocab_size: int,
    ff_layer_fun: Callable[[], torch.nn.Module],
    attention_layer_fun: Callable[[], torch.nn.Module],
    dm: int,
    n_blocks: int,
    device: torch.device,
    init_type,
    init_scale,
    gradient_checkpointing: bool = False,
    model_fragmentation: Optional[list[int]] = None,
    residual_fn: Callable[[], torch.nn.Module] = None,
    n_blanks: int = 0,
    blank_id: int = 0,
    blanks_residual: bool = False,
    blanks_add_embedding: bool = False,
    blanks_learnable_weights: bool = False,
    blank_initial_weight: float = 1.0,
    blanks_straight_through: bool = False,
):
    if model_fragmentation is None or device == torch.device("cpu"):
        first_gpu = device
        last_gpu = device
    else:
        first_gpu = torch.device("cuda:0")
        last_gpu = torch.device(f"cuda:{len(model_fragmentation)}")

    if n_blanks > 0 and blanks_add_embedding:
        embedding_layer = llm.EmbeddingLayer(
            llm.PositionalEmbedding(
                max_length, dm, init_type=init_type, init_scale=init_scale
            ).to(first_gpu),
            BlankEmbedding(
                vocab_size, dm, blank_token_id=blank_id, n_blanks=n_blanks
            ).to(first_gpu),
        )
    else:
        embedding_layer = llm.EmbeddingLayer(
            llm.PositionalEmbedding(
                max_length, dm, init_type=init_type, init_scale=init_scale
            ).to(first_gpu),
            llm.TokenEmbedding(
                vocab_size, dm, init_type=init_type, init_scale=init_scale
            ).to(first_gpu),
        )

    layer_dict = {"attention": attention_layer_fun, "feedforward": ff_layer_fun}
    # Python officially preserves dict order since 3.7, so we pass the layer dict
    encoder_tower = llm.TransformerTower(
        n_blocks,
        dm,
        layer_dict,
        gradient_checkpointing,
        device,
        model_fragmentation=model_fragmentation,
        residual_fn=residual_fn,
    )

    if n_blanks > 0 and blanks_residual:
        head = BlankDiffPredictionHead(
            dm,
            vocab_size,
            init_type=init_type,
            init_scale=init_scale,
            blank_token_id=blank_id,
            n_blanks=n_blanks,
            learnable_weights=blanks_learnable_weights,
            initial_blank_weight=blank_initial_weight,
            use_straight_through=blanks_straight_through,
        ).to(last_gpu)
    else:
        head = llm.PredictionHead(
            dm, vocab_size, init_type=init_type, init_scale=init_scale
        ).to(last_gpu)

    if n_blanks > 0:
        model = BlankLLM(embedding_layer, encoder_tower, head)
    else:
        model = llm.LLM(embedding_layer, encoder_tower, head)

    return model


def get_ff_layer(args):
    if args.ff_mode == "vanilla":
        return lambda: llm.FeedForward(
            args.dmodel, args.dff, init_type=args.init_type, init_scale=args.init_scale
        )
    else:
        raise NotImplementedError(f"ff_mode {args.ff_mode} not implemented")


def get_attention_layer(args):
    attention_layer_fun = lambda: llm.Attention(
        dmodel=args.dmodel,
        heads=args.n_att_heads,
        causal=True,
        dhead=args.dhead,
        flash=True,
        init_type=args.init_type,
        init_scale=args.init_scale,
    )

    return attention_layer_fun


class BlankDiffPredictionHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        output_size: int,
        init_type: str,
        init_scale: float,
        blank_token_id: int,
        n_blanks: int,
        learnable_weights: bool,
        initial_blank_weight: float,
        use_straight_through: bool = False,
    ):
        super(BlankDiffPredictionHead, self).__init__()
        self.linear = nn.Linear(
            embedding_dim,
            output_size,
            init_type=init_type,
            init_scale=init_scale,
            bias=False,
        )
        self.blank_token_id = blank_token_id
        self.n_blanks = n_blanks

        self.learnable_weights = learnable_weights
        self.preblank_weight = nn.Parameter(torch.tensor(1.0))
        self.blank_weight = nn.Parameter(torch.tensor(initial_blank_weight))
        self.use_straight_through = use_straight_through

    def forward(self, encoder_output: torch.Tensor, model_input: torch.Tensor):
        is_blank = model_input.eq(self.blank_token_id)
        is_first_blank = get_first_blanks_in_series(is_blank)
        is_preblank = shift_left(is_first_blank)
        preblank_encoder_output = encoder_output * is_preblank.unsqueeze(-1)
        if self.learnable_weights:
            is_not_blank = ~is_blank
            assert is_not_blank.dtype == torch.bool
            if self.use_straight_through:
                encoder_output = (
                    (encoder_output * is_not_blank.unsqueeze(-1))
                    + (
                        encoder_output.detach()
                        * is_blank.unsqueeze(-1)
                        * abs(self.blank_weight)
                    )
                    + (
                        encoder_output * is_blank.unsqueeze(-1)
                        - encoder_output.detach() * is_blank.unsqueeze(-1)
                    )
                )
            else:
                encoder_output = (encoder_output * is_not_blank.unsqueeze(-1)) + (
                    encoder_output * is_blank.unsqueeze(-1) * abs(self.blank_weight)
                )

            for _ in range(self.n_blanks):
                preblank_encoder_output = shift_right(preblank_encoder_output)
                encoder_output.add_(preblank_encoder_output * abs(self.preblank_weight))
        else:
            for _ in range(self.n_blanks):
                preblank_encoder_output = shift_right(preblank_encoder_output)
                encoder_output.add_(preblank_encoder_output)

        return self.linear(encoder_output)


class BlankSeparateHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        output_size: int,
        blank_token_id: int,
        n_blanks: int,
        learnable_weights: bool,
        initial_blank_weight: float,
        use_straight_through: bool = False,
    ):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, output_size, bias=False)
        self.blank_token_id = blank_token_id
        self.n_blanks = n_blanks

        self.learnable_weights = learnable_weights
        self.preblank_weight = nn.Parameter(torch.tensor(1.0))
        self.blank_weight = nn.Parameter(torch.tensor(initial_blank_weight))
        self.use_straight_through = use_straight_through

    ...


class BlankLLM(nn.Module):
    def __init__(self, embedding_layer, encoder_tower, head):
        super().__init__()

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

        self.attention_manager = BlankAttentionManager(self)

    def forward(self, *args, **kwargs):
        self.attention_manager.set_mask(make_blanks_attention_mask(args[0]))
        if isinstance(self.head, BlankDiffPredictionHead):
            encoder_output = self.encoder.forward(*args, **kwargs)
            return self.head(encoder_output, *args, **kwargs)
        else:
            return self.full_model.forward(*args, **kwargs)


class BlankEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        blank_token_id: int,
        n_blanks: int,
        init_type: Literal["kaiming_uniform", "truncated_normal"],
        init_scale: float,
    ):
        super(BlankEmbedding, self).__init__()
        self.embedding = llm.TokenEmbedding(
            vocab_size,
            embedding_dim,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.blank_token_id = blank_token_id
        self.n_blanks = n_blanks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding_output = self.embedding(x)
        is_blank = x.eq(self.blank_token_id)
        is_first_blank = get_first_blanks_in_series(is_blank)
        is_preblank = shift_left(is_first_blank)
        preblank_embedding_output = embedding_output * is_preblank.unsqueeze(-1)
        for _ in range(self.n_blanks):
            preblank_embedding_output = shift_right(preblank_embedding_output)
            embedding_output.add_(preblank_embedding_output)
        return embedding_output


class BlankAttentionManager:
    def __init__(
        self,
        model,
    ):
        self._layers = []
        self._register_layers(model)

    def _register_layers(self, model):
        for name, layer in model.named_modules():
            if name.endswith("attention"):
                self._layers.append(layer)

    def set_mask(self, mask):
        for layer in self._layers:
            if isinstance(layer, BlankAttention):
                layer.set_mask(mask)
            else:
                raise ValueError(
                    f"Layer {layer} is not a BlankAttention layer (something is not yes)"
                )


class BlankAttention(nn.Module):
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
        super(BlankAttention, self).__init__()
        if dhead is None:
            assert dmodel % heads == 0
            dhead = dmodel // heads

        self.heads = heads
        self.dhead = dhead
        self.causal = causal
        self.flash = flash

        self.input_projection = nn.Linear(
            dmodel,
            3 * heads * dhead,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.output_projection = nn.Linear(
            heads * dhead,
            dmodel,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, x):
        projected = self.input_projection(x)

        batch, seq_len = x.shape[:-1]
        projected = projected.view(
            batch, seq_len, self.heads, 3 * self.dhead
        ).transpose(1, 2)
        q, k, v = torch.chunk(projected, chunks=3, dim=-1)

        attention_output = custom_mask_attention_mechanism(
            query=q,
            key=k,
            value=v,
            dhead=self.dhead,
            mask=self.mask,
            flash=self.flash,
        )

        output = self.output_projection(attention_output.transpose(1, 2).flatten(-2))

        return output


def custom_mask_attention_mechanism(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dhead: int,
    mask: torch.Tensor,
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
                attn_mask=mask == 0,
            )
    else:
        # implementation without flash assumes other dim order
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        a = torch.einsum("... l h d, ... L h d -> ... h l L", query, key)
        a = a * (1 / dhead**0.5)
        a.masked_fill_(mask, float("-inf"))
        a = torch.softmax(a, dim=-1)
        output = torch.einsum("... h l L, ... L h d -> ... l h d", a, value)
        output = output.transpose(1, 2)

    return output
