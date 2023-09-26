from collections import OrderedDict
from typing import Callable
import lizrd.core.nn as nn
from research.blanks.utils import get_first_blanks_in_series, shift_left, shift_right


import torch


def straight_through(forward: Callable, backward: Callable) -> Callable:
    def ste():
        backward_output = backward()
        forward_output = forward()
        return forward_output.detach() + (backward_output - backward_output.detach())

    return ste()


class BlankDiffPredictionHead(nn.Module):
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
        super(BlankDiffPredictionHead, self).__init__()
        self.linear = nn.Linear(embedding_dim, output_size, bias=False)
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

    def forward(self, *args, **kwargs):
        if isinstance(self.head, BlankDiffPredictionHead):
            encoder_output = self.encoder.forward(*args, **kwargs)
            return self.head(encoder_output, *args, **kwargs)
        else:
            return self.full_model.forward(*args, **kwargs)


class BlankEmbedding(nn.Module):
    def __init__(
        self, vocab_size: int, embedding_dim: int, blank_token_id: int, n_blanks: int
    ):
        super(BlankEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
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
