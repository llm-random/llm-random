from collections import OrderedDict
from typing import Callable, Optional

import torch
import torch.nn as nn

from lizrd.core import misc
from research.grad_norm.modules.gn_transformer_block import GradModifiedTransformerBlock
from research.grad_norm.modules.grad_modif_placement import BlockGradModifPlacement


class GradModiedTransformerTower(nn.Module):
    def __init__(
        self,
        n_blocks,
        dmodel,
        layer_dict,
        device: torch.device = None,
        model_fragmentation: Optional[list[int]] = None,
        gn_placement: Optional[BlockGradModifPlacement] = None,
        grad_modif_fn: Callable[[], nn.Module] = None,
        grad_log_fn: Callable[[], nn.Module] = None,
    ):
        super().__init__()
        misc.check_layer_funs(*layer_dict.values())
        self.blocks = []
        self.model_fragmentation = [] if model_fragmentation is None else model_fragmentation
        self.device = device

        for i_block in range(n_blocks):
            layers_info = [(name, layer_fun()) for name, layer_fun in layer_dict.items()]

            for name, layer in layers_info:
                layer.layer_type = name
                layer.block_number = i_block

            _, current_device = self.get_current_device(i_block)
            block = GradModifiedTransformerBlock(
                dmodel, layers_info, gn_placement=gn_placement, grad_modif_fn=grad_modif_fn, grad_log_fn=grad_log_fn
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

        return block_num in self.model_fragmentation, torch.device(f"cuda:{len(self.model_fragmentation)}")
