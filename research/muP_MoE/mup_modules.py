import torch
import torch.nn as nn
from typing import Literal

from lizrd.core.llm import LLM
from lizrd.core.misc import LoggingLayer, Linear
from lizrd.core.initialization import ValidInitType


class nonResidual(LoggingLayer):
    def __init__(self, layer, alpha=1.0, m_d=1.0):
        super(nonResidual, self).__init__()
        self.layer = layer
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.bfloat16))
        self.register_buffer("m_d", torch.tensor(m_d, dtype=torch.bfloat16))

    def forward(self, x):
        out = self.layer(x)
        out *= self.alpha / self.m_d  # muP scaling
        self.update_cache_for_logging("update", out)
        return out

    def log_heavy(self):
        updates = self.logging_cache["update"]

        mean_abs_update = torch.mean(torch.abs(updates))

        return {
            "muP/mean_abs_update": mean_abs_update,
        }


class muP_LLM(LLM):
    def __init__(self, embedding_layer, encoder_tower, head, mup_config: dict = None):
        super(muP_LLM, self).__init__(embedding_layer, encoder_tower, head)

        alpha_in = 1.0
        alpha_out = 1.0
        m_d = 1.0

        self.mup = False
        if mup_config is not None:
            self.mup = True
            # Register alpha_in and alpha_out as buffers to make them non-trainable
            alpha_in = mup_config["alpha_in"]
            alpha_out = mup_config["alpha_out"]
            m_d = mup_config["m_d"]

        self.embedding_layer = nonResidual(
            self.embedding_layer, alpha=alpha_in, m_d=1.0
        )
        self.head = nonResidual(self.head, alpha=alpha_out, m_d=m_d)

    def forward(self, *args, **kwargs):
        x = self.embedding_layer(*args, **kwargs)
        x = self.encoder(x)
        x = self.head(x)
        return x


def decode_bias_string(bias):
    assert bias in ["both", "first", "second", "none"]
    if bias == "both":
        bias_first = bias_second = True
    elif bias == "first":
        bias_first = True
        bias_second = False
    elif bias == "second":
        bias_first = False
        bias_second = True
    else:
        bias_first = bias_second = False
    return bias_first, bias_second


class FeedForward(LoggingLayer):
    def __init__(
        self,
        dmodel,
        dff,
        init_type: ValidInitType,
        init_scale: float,
        bias: Literal["both", "first", "second", "none"] = "none",
    ):
        super().__init__()
        bias_first, bias_second = decode_bias_string(bias)
        self.lin1_weight = Linear(
            dmodel,
            dff,
            bias=bias_first,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.relu = nn.ReLU()
        self.lin2_weight = Linear(
            dff,
            dmodel,
            bias=bias_second,
            init_type=init_type,
            init_scale=init_scale,
        )

    def forward(self, x):
        out = self.lin1_weight(x)
        out = self.relu(out)
        out = self.lin2_weight(out)
        return out
