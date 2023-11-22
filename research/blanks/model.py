from collections import OrderedDict
from typing import Callable, Literal, Optional
from lizrd.core import llm
from research.blanks.utils import (
    get_first_blanks_in_series,
    shift_left,
    shift_right,
    make_blanks_fixed_positions,
)
from lizrd.core.initialization import get_init_weight


import torch

from lizrd.core import llm
import lizrd.core.misc as misc


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
            blank_token_id=blank_id,
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
                blank_token_id=blank_id,
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


class BlankDiffPredictionHead(torch.nn.Module):
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
        self.linear = misc.Linear(
            embedding_dim,
            output_size,
            init_type=init_type,
            init_scale=init_scale,
            bias=False,
        )
        self.blank_token_id = blank_token_id
        self.n_blanks = n_blanks

        self.learnable_weights = learnable_weights
        self.preblank_weight = torch.nn.Parameter(torch.tensor(1.0))
        self.blank_weight = torch.nn.Parameter(torch.tensor(initial_blank_weight))
        self.use_straight_through = use_straight_through

    def forward(self, encoder_output: torch.Tensor, model_input: torch.Tensor):
        is_blank = model_input.eq(self.blank_token_id)
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
        init_type: str,
        init_scale: float,
        blank_token_id: int,
        n_blanks: int,
        learnable_weights: bool,
        initial_blank_weight: float,
        use_straight_through: bool = False,
        use_residual_blank: bool = False,
    ):
        super().__init__()
        self.regular_head = misc.Linear(
            embedding_dim,
            output_size,
            init_type=init_type,
            init_scale=init_scale,
            bias=False,
        )
        self.blank_head = misc.Linear(
            embedding_dim,
            output_size,
            init_type=init_type,
            init_scale=init_scale,
            bias=False,
        )

        if (learnable_weights or use_straight_through) and not use_residual_blank:
            raise ValueError(
                "learnable_weights and use_straight_through only work with use_residual_blank"
            )

        self.blank_token_id = blank_token_id
        self.n_blanks = n_blanks

        self.learnable_weights = learnable_weights
        self.preblank_weight = torch.nn.Parameter(torch.tensor(1.0))
        self.blank_weight = torch.nn.Parameter(torch.tensor(initial_blank_weight))
        self.use_straight_through = use_straight_through
        self.use_residual_blank = use_residual_blank

    def non_residual_forward(
        self, encoder_output: torch.Tensor, model_input: torch.Tensor
    ):
        is_blank = model_input.eq(self.blank_token_id)
        is_not_blanks = ~is_blank
        assert is_not_blanks.dtype == is_blank.dtype == torch.bool
        assert self.regular_head.bias == self.blank_head.bias == None
        return self.regular_head(
            encoder_output * is_not_blanks.unsqueeze(-1)
        ) + self.blank_head(encoder_output * is_blank.unsqueeze(-1))

    def residual_forward(self, encoder_output: torch.Tensor, model_input: torch.Tensor):
        is_blank = model_input.eq(self.blank_token_id)
        is_not_blank = ~is_blank
        assert is_not_blank.dtype == torch.bool

        is_first_blank = get_first_blanks_in_series(is_blank)
        is_preblank = shift_left(is_first_blank)

        if self.learnable_weights:
            if self.use_straight_through:
                full_output = self.regular_head(
                    encoder_output * is_not_blank.unsqueeze(-1)
                ) + self.blank_head(
                    (
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
                non_blanks_output = self.regular_head(
                    encoder_output * is_not_blank.unsqueeze(-1)
                )
                blanks_output = self.blank_head(
                    encoder_output * is_blank.unsqueeze(-1) * abs(self.blank_weight)
                )
                full_output = non_blanks_output + blanks_output

            summing_positions = is_preblank.unsqueeze(-1)

            for _ in range(self.n_blanks):
                full_output.add_(
                    shift_right(full_output * summing_positions)
                    * abs(self.preblank_weight)
                )

        else:
            non_blanks_output = self.regular_head(
                encoder_output * is_not_blank.unsqueeze(-1)
            )
            blanks_output = self.blank_head(encoder_output * is_blank.unsqueeze(-1))
            full_output = non_blanks_output + blanks_output
            summing_positions = is_preblank.unsqueeze(-1)

            for _ in range(self.n_blanks):
                full_output.add_(shift_right(full_output * summing_positions))

        return full_output

    def forward(self, encoder_output: torch.Tensor, model_input: torch.Tensor):
        if self.use_residual_blank:
            return self.residual_forward(encoder_output, model_input)
        else:
            return self.non_residual_forward(encoder_output, model_input)
        # return self.linear(encoder_output)


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

    def forward(self, *args, **kwargs):
        if isinstance(self.head, BlankDiffPredictionHead):
            encoder_output = self.encoder.forward(*args, **kwargs)
            return self.head(encoder_output, *args, **kwargs)
        else:
            return self.full_model.forward(*args, **kwargs)


class BlankEmbedding(torch.nn.Module):
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
        current_accumulator_positions = is_preblank.unsqueeze(-1)
        embedding_accumulator = embedding_output * current_accumulator_positions
        for _ in range(self.n_blanks):
            current_accumulator_positions = shift_right(current_accumulator_positions)
            embedding_accumulator = shift_right(embedding_accumulator)
            embedding_output.add_(embedding_accumulator)
            embedding_accumulator = embedding_output * current_accumulator_positions
        return embedding_output


class BlankPositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        max_length,
        embedding_dim,
        init_type: Literal["kaiming_uniform", "truncated_normal"],
        init_scale: float,
        blank_token_id: int,
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
        self.blank_token_id = blank_token_id
        self.n_blanks = n_blanks

    def forward(self, x):
        positions = make_blanks_fixed_positions(
            x, self.blank_token_id, n_blanks_block=self.n_blanks
        )
        embeddings = self.layer(positions)
        return embeddings
