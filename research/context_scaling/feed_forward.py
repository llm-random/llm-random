from collections import OrderedDict
from typing import Literal, Callable, Optional
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from lizrd.core import misc
from lizrd.core.misc import default, Aggregate
from lizrd.core.initialization import get_init_weight, ValidInitType
from lizrd.core.misc import Linear, LoggingLayer


class SwiGLUFeedForward(LoggingLayer):
    def __init__(
        self,
        dmodel,
        dff,
        init_type: ValidInitType,
        init_scale: float,
    ):
        super().__init__()
        self.w1_gate = Linear(
            dmodel, dff * 2, init_type=init_type, init_scale=init_scale, bias=False
        )
        self.w2 = Linear(
            dff, dmodel, init_type=init_type, init_scale=init_scale, bias=False
        )

    def forward(self, x):
        pre_activation, gate = torch.chunk(self.w1_gate(x), 2, dim=-1)
        activation = nn.functional.silu(pre_activation)
        return self.w2(activation * gate)
