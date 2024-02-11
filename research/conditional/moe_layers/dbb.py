from typing import Literal, Optional

import torch

from lizrd.core import nn

# from lizrd.core.initialization import get_init_weight
from lizrd.core.misc import Linear


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
