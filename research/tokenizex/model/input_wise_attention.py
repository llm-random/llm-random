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


def get_attention_layer(args):
    return lambda: BlankAttention(
        dmodel=args.dmodel,
        heads=args.n_att_heads,
        dhead=args.dhead,
        flash=False,
        init_type=args.init_type,
        init_scale=args.init_scale,
    )


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

    def set_mask(self, mask: torch.Tensor): #dev ? context manager
        self.mask = mask

    def remove_mask(self): #dev ? context manager
        self.mask = None

    def forward(self, x):
        projected = self.input_projection(x)

        batch, seq_len = x.shape[:-1]
        projected = projected.view(
            batch, seq_len, self.heads, 3 * self.dhead
        ).transpose(1, 2)
        q, k, v = torch.chunk(projected, chunks=3, dim=-1)

        attention_output = custom_mask_attention_mechanism( # po prostu bierze maskę z context managera?
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
        is_blank = get_is_blank(model_input, self.blank_tokens_ids) #del
        is_first_blank = get_first_blanks_in_series(is_blank) #del
        is_preblank = shift_left(is_first_blank) #del

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
