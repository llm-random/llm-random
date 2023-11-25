from collections import OrderedDict
from typing import Callable, Literal, Optional, List
from lizrd.core import llm
from research.blanks.utils import (
    get_first_blanks_in_series,
    get_is_blank,
    shift_left,
    shift_right,
    make_blanks_fixed_positions,
)
from lizrd.core.initialization import get_init_weight


import torch

from lizrd.core import llm
import lizrd.core.misc as misc
import torch.nn.functional as F


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
    blank_ids: torch.Tensor,
    blanks_use_multiple_embeddings: bool,
    gradient_checkpointing: bool = False,
    model_fragmentation: Optional[list[int]] = None,
    residual_fn: Callable[[], torch.nn.Module] = None,
    n_blanks: int = 0,
    blanks_residual: bool = False,
    blanks_add_embedding: bool = False,
    blanks_learnable_weights: bool = False,
    blank_initial_weight: float = 1.0,
    blanks_straight_through: bool = False,
    blanks_use_custom_positional_embedding: bool = False,
):
    if model_fragmentation is None or device == torch.device("cpu"):
        first_gpu = device
        last_gpu = device
    else:
        first_gpu = torch.device("cuda:0")
        last_gpu = torch.device(f"cuda:{len(model_fragmentation)}")

    if blanks_use_custom_positional_embedding:
        positional_embedding = BlankPositionalEmbedding(
            max_length,
            dm,
            init_type=init_type,
            init_scale=init_scale,
            blank_tokens_ids=blank_ids,
            n_blanks=n_blanks,
        ).to(first_gpu)
    else:
        positional_embedding = llm.PositionalEmbedding(
            max_length, dm, init_type=init_type, init_scale=init_scale
        ).to(first_gpu)
    if n_blanks > 0 and blanks_add_embedding:
        embedding_layer = llm.EmbeddingLayer(
            positional_embedding,
            BlankEmbedding(
                vocab_size,
                dm,
                blank_tokens_ids=blank_ids,
                n_blanks=n_blanks,
                init_type=init_type,
                init_scale=init_scale,
            ).to(first_gpu),
        )
    else:
        embedding_layer = llm.EmbeddingLayer(
            positional_embedding,
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
            blank_tokens_ids=blank_ids,
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
    if args.blanks_use_custom_attention and args.n_blanks > 0:
        attention_layer_fun = lambda: BlankAttention(
            dmodel=args.dmodel,
            heads=args.n_att_heads,
            dhead=args.dhead,
            flash=False,
            init_type=args.init_type,
            init_scale=args.init_scale,
        )
    else:
        attention_layer_fun = lambda: llm.Attention(
            dmodel=args.dmodel,
            heads=args.n_att_heads,
            dhead=args.dhead,
            flash=True,
            causal=True,
            init_type=args.init_type,
            init_scale=args.init_scale,
        )
    return attention_layer_fun


class BlankDiffPredictionHead(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        output_size: int,
        init_type: str,
        init_scale: float,
        blank_tokens_ids: torch.Tensor,
        n_blanks: int,
        learnable_weights: bool,
        initial_blank_weight: float,
        use_straight_through: bool = False,
    ):
        super(BlankDiffPredictionHead, self).__init__()
        self.linear = misc.Linear(
            embedding_dim,
            output_size,
            init_type=init_type,
            init_scale=init_scale,
            bias=False,
        )
        self.blank_tokens_ids = blank_tokens_ids
        self.n_blanks = n_blanks

        self.learnable_weights = learnable_weights
        self.preblank_weight = torch.nn.Parameter(torch.tensor(1.0))
        self.blank_weight = torch.nn.Parameter(torch.tensor(initial_blank_weight))
        self.use_straight_through = use_straight_through

    def forward(self, encoder_output: torch.Tensor, model_input: torch.Tensor):
        is_blank = get_is_blank(model_input, self.blank_tokens_ids)
        is_first_blank = get_first_blanks_in_series(is_blank)
        is_preblank = shift_left(is_first_blank)

        current_accumulator_positions = is_preblank.unsqueeze(-1)
        encoder_accumulator = encoder_output * current_accumulator_positions

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
                current_accumulator_positions = shift_right(
                    current_accumulator_positions
                )
                encoder_accumulator = shift_right(encoder_accumulator)
                encoder_output.add_(encoder_accumulator * abs(self.preblank_weight))
                encoder_accumulator = encoder_output * current_accumulator_positions

        else:
            for _ in range(self.n_blanks):
                current_accumulator_positions = shift_right(
                    current_accumulator_positions
                )
                encoder_accumulator = shift_right(encoder_accumulator)
                encoder_output.add_(encoder_accumulator)
                encoder_accumulator = encoder_output * current_accumulator_positions

        return self.linear(encoder_output)


class BlankSeparateHead(torch.nn.Module):
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
        self.linear = misc.Linear(embedding_dim, output_size, bias=False)
        self.blank_token_id = blank_token_id
        self.n_blanks = n_blanks

        self.learnable_weights = learnable_weights
        self.preblank_weight = torch.nn.Parameter(torch.tensor(1.0))
        self.blank_weight = torch.nn.Parameter(torch.tensor(initial_blank_weight))
        self.use_straight_through = use_straight_through

    ...


class BlankLLM(torch.nn.Module):
    def __init__(self, embedding_layer, encoder_tower, head):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            OrderedDict(
                [
                    ("embedding_layer", embedding_layer),
                    ("encoder", encoder_tower),
                ]
            )
        )
        self.full_model = torch.nn.Sequential(
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

    def forward(self, input_tokens, attention_mask):
        with self.attention_manager.set_mask(attention_mask):
            if isinstance(self.head, BlankDiffPredictionHead):
                encoder_output = self.encoder.forward(input_tokens)
                return self.head(encoder_output, input_tokens)
            else:
                return self.full_model.forward(input_tokens)


class BlankEmbedding(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        blank_tokens_ids: torch.Tensor,
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
        self.blank_tokens_ids = blank_tokens_ids
        self.n_blanks = n_blanks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding_output = self.embedding(x)
        is_blank = get_is_blank(x, self.blank_tokens_ids)
        is_first_blank = get_first_blanks_in_series(is_blank)
        is_preblank = shift_left(is_first_blank)
        current_accumulator_positions = is_preblank.unsqueeze(-1)
        embedding_accumulator = embedding_output * current_accumulator_positions
        for _ in range(self.n_blanks):
            current_accumulator_positions = shift_right(current_accumulator_positions)
            embedding_accumulator = shift_right(embedding_accumulator)
            embedding_output.add_(embedding_accumulator)
            embedding_accumulator = embedding_output * current_accumulator_positions
        return embedding_output


class BlankAttention(torch.nn.Module):
    def __init__(
        self,
        dmodel: int,
        heads: int,
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
        self.flash = flash

        self.input_projection = misc.Linear(
            dmodel,
            3 * heads * dhead,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.output_projection = misc.Linear(
            heads * dhead,
            dmodel,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.mask = None

    def set_mask(self, mask: torch.Tensor):
        self.mask = mask

    def remove_mask(self):
        self.mask = None

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
            mask=self.mask.to(x.device),
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
                attn_mask=mask,
            )
    else:
        # implementation without flash assumes other dim order
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        a = torch.einsum("... l h d, ... L h d -> ... h l L", query, key)
        a = a * (1 / dhead**0.5)
        a.masked_fill_(~mask, float("-inf"))
        a = torch.softmax(a, dim=-1)
        output = torch.einsum("... h l L, ... L h d -> ... l h d", a, value)
        output = output.transpose(1, 2)

    return output


class BlankAttentionManager:
    def __init__(
        self,
        model: torch.nn.Module,
    ):
        self._layers = []
        self._register_layers(model)

    def _register_layers(self, model: torch.nn.Module):
        for _, layer in model.named_modules():
            if isinstance(layer, BlankAttention):
                self._layers.append(layer)

    def set_mask(self, mask: torch.Tensor):
        mask.unsqueeze_(1)
        return MaskSetter(self._layers, mask)


class MaskSetter:
    def __init__(self, layers: List[BlankAttention], mask: torch.Tensor):
        self.layers = layers
        self.mask = mask

    def __enter__(self):
        for layer in self.layers:
            layer.set_mask(self.mask)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for layer in self.layers:
            layer.remove_mask()


class BlankPositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        max_length,
        embedding_dim,
        init_type: Literal["kaiming_uniform", "truncated_normal"],
        init_scale: float,
        blank_tokens_ids: torch.Tensor,
        n_blanks: int,
    ):
        super(BlankPositionalEmbedding, self).__init__()
        self.layer = torch.nn.Embedding(max_length, embedding_dim)
        default_weight = self.layer.weight.data
        self.layer.weight.data = get_init_weight(
            shape=default_weight.shape,
            fan_in=1,
            init_type=init_type,
            scale=init_scale,
            dtype=default_weight.dtype,
        )
        self.blank_tokens_ids = blank_tokens_ids
        self.n_blanks = n_blanks

    def forward(self, x):
        positions = make_blanks_fixed_positions(
            x, self.blank_tokens_ids, n_blanks_block=self.n_blanks
        )
        embeddings = self.layer(positions)
        return embeddings
