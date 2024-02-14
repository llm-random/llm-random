from typing import Literal, Optional

import torch

from lizrd.core import nn

# from lizrd.core.initialization import get_init_weight
from lizrd.core.misc import Linear
from research.conditional.utils.layer_manager import LoggingLayer


class SimpleScale(torch.nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))

    def forward(self, x):
        return x * self.scale


def sequence_to_module(
    seq: str, use_ln: bool, dmodel: int, init_type, init_scale
) -> torch.nn.Module:
    blocks_list: list[torch.nn.Module] = (
        [] if not use_ln else [torch.nn.LayerNorm(dmodel)]
    )
    multipliers = seq.split(">")
    if multipliers[0].lower() == "ln":
        blocks_list.append(torch.nn.LayerNorm(dmodel))
        multipliers = multipliers[1:]
    prev_in_x = 1
    for in_x, out_x in zip(multipliers, multipliers[1:]):
        if out_x.lower() == "ln":
            in_x = float(in_x)
            prev_in_x = in_x
            blocks_list.append(torch.nn.LayerNorm(int(in_x * dmodel)))
        elif out_x.lower() == "sc":
            in_x = float(in_x)
            prev_in_x = in_x
            blocks_list.append(SimpleScale(scale=1.0))
        else:
            if in_x.lower() == "ln":
                in_x, out_x = prev_in_x, float(out_x)
            else:
                in_x, out_x = float(in_x), float(out_x)
            blocks_list.append(
                Linear(
                    in_features=int(in_x * dmodel),
                    out_features=int(out_x * dmodel),
                    init_type=init_type,
                    init_scale=init_scale,
                ),
            )
    blocks = torch.nn.Sequential(*blocks_list)
    return blocks


class DBBFF(torch.nn.Module):
    def __init__(
        self,
        dmodel: int,
        dff: int,
        n_blocks: int,
        init_type: Literal["kaiming_uniform", "truncated_normal"],
        init_scale: float,
        sanity_check: Optional[str] = None,
    ):
        super().__init__()
        # self.lin1_weight = nn.Parameter(
        #     get_init_weight(
        #         shape=(n_blocks, dmodel, dff),
        #         fan_in=dmodel,
        #         init_type=init_type,
        #         scale=init_scale,
        #     )
        # )
        # self.lin2_weight = nn.Parameter(
        #     get_init_weight(
        #         shape=(n_blocks, dff, dmodel),
        #         fan_in=dff,
        #         init_type=init_type,
        #         scale=init_scale,
        #     )
        # )
        self.n_blocks = n_blocks
        # self.out_block = self.lin2_weight = nn.Parameter(
        #     get_init_weight(
        #         shape=(dff, dmodel),
        #         fan_in=dff,
        #         init_type=init_type,
        #         scale=init_scale,
        #     )
        # )
        self.in_blocks = []
        for _ in range(n_blocks):
            self.in_blocks.append(
                nn.Sequential(
                    nn.LayerNorm(dmodel),
                    Linear(
                        in_features=dmodel,
                        out_features=dff,
                        init_type=init_type,
                        init_scale=init_scale,
                    ),
                )
            )
        self.in_blocks = nn.ModuleList(self.in_blocks)
        self.out_block = Linear(
            in_features=dff,
            out_features=dmodel,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.skip_ln = True
        # if sanity_check == "collapse":
        #     with torch.no_grad():
        #         self.lin1_weight.data = self.lin1_weight.sum(dim=0, keepdim=True)
        #         self.lin2_weight.data = self.lin2_weight.sum(dim=0, keepdim=True)

    def forward(self, x):
        if self.training:
            return self.train_forward(x)
        else:
            return self.eval_forward(x)
        # else:
        #     raise ValueError(f"Unknown mode: {mode}")

    def train_forward(self, x):
        # from icecream import ic

        # ic(x.shape)
        # x = torch.einsum("bld,ndf->blf", x, self.lin1_weight)
        block_outputs = []
        for i in range(self.n_blocks):
            block_outputs.append(self.in_blocks[i](x))
        x = sum(block_outputs)
        x = x.relu()
        # x = torch.einsum("blf,fd->bld", x, self.lin2_weight)
        x = self.out_block(x)
        # ic(x.shape)
        # x = torch.einsum("blf,nfd->bld", x, self.lin2_weight)
        # ic(x.shape)
        return x

    def eval_forward(self, x):
        return self.train_forward(x)
        # aggregated_lin1_weight = self.lin1_weight.sum(dim=0)
        # aggregated_lin2_weight = self.lin2_weight.sum(dim=0)
        # x = torch.einsum("bld,df->blf", x, aggregated_lin1_weight)
        # x = x.relu()
        # x = torch.einsum("blf,fd->bld", x, aggregated_lin2_weight)
        # return x


class LRDBBFF(LoggingLayer):

    def __init__(
        self,
        dmodel: int,
        # dff: int,
        # n_blocks: int,
        use_ln: bool,
        init_type: Literal["kaiming_uniform", "truncated_normal"],
        init_scale: float,
        topology="1>4|4>1",  # e.g. "1>4,1>0.5>4|4>1". "1>4|4>1" denotes regular FF.
        # sanity_check: Optional[str] = None,
    ):
        super().__init__()
        in_, out = topology.split("|")
        in_blocks = []
        out_blocks = []
        for pairs in in_.split(","):
            in_blocks.append(
                sequence_to_module(pairs, use_ln, dmodel, init_type, init_scale)
            )
        for pairs in out.split(","):
            out_blocks.append(
                sequence_to_module(pairs, False, dmodel, init_type, init_scale)
            )
        self.in_blocks = torch.nn.ModuleList(in_blocks)
        self.out_blocks = torch.nn.ModuleList(out_blocks)

    def forward(self, x):
        if self.training:
            return self.train_forward(x)
        else:
            return self.eval_forward(x)

    def train_forward(self, x):
        in_outputs = []
        for i, block in enumerate(self.in_blocks):
            block_output = block(x)
            in_outputs.append(block_output)
            self.update_cache_for_logging(f"in_outputs/{i}", block_output)
        x = sum(in_outputs)
        x = x.relu()
        out_outputs = []
        for i, block in enumerate(self.out_blocks):
            block_output = block(x)
            out_outputs.append(block_output)
            self.update_cache_for_logging(f"out_outputs/{i}", block_output)
        return sum(out_outputs)

    def log_light(self):
        scales = []
        for block in self.in_blocks:
            for layer in block:
                if isinstance(layer, SimpleScale):
                    scales.append(layer.scale.detach().cpu().numpy())
        return {
            **{f"scales/{i}": scale for i, scale in enumerate(scales)},
            **{key: torch.norm(value) for key, value in self.logging_cache.items()},
        }

    def eval_forward(self, x):
        return self.train_forward(x)
        # aggregated_lin1_weight = self.lin1_weight.sum(dim=0)
        # aggregated_lin2_weight = self.lin2_weight.sum(dim=0)
        # x = torch.einsum("bld,df->blf", x, aggregated_lin1_weight)
        # x = x.relu()
        # x = torch.einsum("blf,fd->bld", x, aggregated_lin2_weight)
        # return x
