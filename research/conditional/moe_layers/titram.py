from typing import Literal

import mamba_ssm
import torch

from lizrd.core import nn
from lizrd.core.initialization import get_init_weight
from lizrd.support.logging import make_histogram
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
        self.regression = nn.Parameter(
            get_init_weight(
                self.dmodel,
                fan_in=self.dmodel,
                init_type=init_type,
                scale=init_scale,
            )
        )
        self.weight = nn.Parameter(
            get_init_weight(
                self.dmodel,
                fan_in=self.dmodel,
                init_type=init_type,
                scale=init_scale,
            )
        )
        self.non_neg = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        Returns: same shape as x
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        lookback_limit = torch.arange(seq_len, device=x.device, dtype=torch.int64)

        mamba_output = self.mamba.forward(x)
        lookback_weight = torch.matmul(mamba_output, self.weight).view(
            batch_size, seq_len, 1
        )
        lookback_regression = torch.clamp(
            torch.add(-torch.matmul(mamba_output, self.regression), 1), 0, 1
        )
        self.update_cache_for_logging("regression", lookback_regression)
        self.update_cache_for_logging("weight", lookback_weight)
        lookback_regression = lookback_limit * lookback_regression
        lookback_regression = torch.round(lookback_regression).type(torch.int64)
        lookback_regression, _ = torch.cummax(lookback_regression, dim=1)
        lookback_regression = lookback_regression.view(batch_size, seq_len, 1).expand(
            batch_size, seq_len, self.dmodel
        )
        lookback = torch.gather(mamba_output, 1, lookback_regression)
        return mamba_output + lookback

    def log_heavy(self):
        return {
            "lookback_regression": make_histogram(
                self.logging_cache["regression"].flatten().float()
            ),
            "lookback_weight": make_histogram(
                self.logging_cache["weight"].flatten().float()
            ),
        }
