from abc import ABC
from collections import OrderedDict
from typing import Any, Callable, Literal, Optional, List
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


class InputWiseMask:
    def __init__(self) -> None:
        self.mask:torch.Tensor = None

    def set_mask(self, mask: torch.Tensor):
        self.mask = mask

    def remove_mask(self):
        self.mask = None


class ManagerMaskSetter:
    def __init__(self, model: Any, mask: torch.Tensor):
        self.mask = mask
        self._layers:list[InputWiseMask] = []
        for _, layer in model.named_modules():
            if isinstance(layer, InputWiseMask):
                self._layers.append(layer)
        if len(self._layers) == 0:
            raise Exception("No InputWiseMask modules in provided model")

    def __enter__(self):
        self.mask.unsqueeze_(1)
        for layer in self._layers:
            layer.set_mask(self.mask)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.mask.squeeze_()
        for layer in self._layers:
            layer.remove_mask()
        

def custom_mask_attention_mechanism(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dhead: int,
    mask: torch.Tensor,
    flash: bool,
):
    if mask is None:
        raise Exception("Mask was not set, not managed")
            
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

class TokenizexAttention(torch.nn.Module, InputWiseMask):
    def __init__(
        self,
        dmodel: int,
        heads: int,
        init_type: str,
        init_scale: float,
        dhead=None,
        flash=False,
    ):
        torch.nn.Module.__init__(self)
        InputWiseMask.__init__(self)
        # super(TokenizexAttention, self).__init__()
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


def get_attention_layer(args):
    return lambda: TokenizexAttention(
        dmodel=args.dmodel,
        heads=args.n_att_heads,
        dhead=args.dhead,
        flash=False,
        init_type=args.init_type,
        init_scale=args.init_scale,
    )



# class BlankAttentionManager:
#     def __init__(
#         self,
#         model: torch.nn.Module,
#     ):
#         self._layers = []
#         self._register_layers(model)

#     def _register_layers(self, model: torch.nn.Module):
#         for _, layer in model.named_modules():
#             if isinstance(layer, BlankAttention):
#                 self._layers.append(layer)

#     def set_mask(self, mask: torch.Tensor):
#         mask.unsqueeze_(1)
#         return MaskSetter(self._layers, mask)


# class MaskSetter:
#     def __init__(self, layers: List[BlankAttention], mask: torch.Tensor):
#         self.layers = layers
#         self.mask = mask

#     def __enter__(self):
#         for layer in self.layers:
#             layer.set_mask(self.mask)

#     def __exit__(self, exc_type, exc_value, exc_traceback):
#         for layer in self.layers:
#             layer.remove_mask()
