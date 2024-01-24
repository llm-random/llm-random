from typing import Literal

import mamba_ssm
import torch

from lizrd.core import nn
from lizrd.core.misc import Linear
from research.conditional.utils.layer_manager import LoggingLayer


# class DummyMamba:
#     def __init__(self, d_model):
#         self.dmodel = d_model
#
#     def forward(self, x: torch.Tensor):
#         batch_size, seq_len = x.shape[0], x.shape[1]
#         return torch.rand(batch_size, seq_len, self.dmodel)


class TiTraMamba(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        init_type: Literal["kaiming_uniform", "truncated_normal"],
        init_scale: float,
    ):
        super().__init__()
        self.dmodel = dmodel
        self.mamba = mamba_ssm.Mamba(d_model=self.dmodel)
        # self.mamba = DummyMamba(d_model=self.dmodel)

        self.log_lookback = [0, 1, 2, 4, 8, 16, 32, 64, 128]
        self.weight = Linear(
            dmodel,
            9,
            bias=True,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.non_neg = nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        Returns: same shape as x
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        lookback_limit = (
            torch.arange(seq_len, device=x.device, dtype=torch.int64)
            .view(1, -1, 1)
            .expand(batch_size, seq_len, self.dmodel)
        )

        lookback_weight = self.softmax(self.weight(x))
        mamba_input = torch.zeros_like(x)
        for slice_idx, lookback_val in enumerate(self.log_lookback):
            lookback_idx = self.non_neg(torch.sub(lookback_limit, lookback_val))
            lookback = torch.gather(x, 1, lookback_idx)
            mamba_input = mamba_input + lookback * torch.unsqueeze(
                torch.select(lookback_weight, dim=2, index=slice_idx), dim=-1
            )

        mamba_output = self.mamba.forward(mamba_input)
        return mamba_output
