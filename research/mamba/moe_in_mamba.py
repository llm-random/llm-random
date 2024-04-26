import torch
from einops import rearrange

from lizrd.core.misc import LoggingLayer


class MambaInProj(LoggingLayer):
    def __init__(self, batch_size, conv_proj, gate_proj, dtype):
        super().__init__()
        self.conv_proj = conv_proj
        self.gate_proj = gate_proj
        self.bias = None
        self.batch_size = batch_size
        self.dtype = dtype

    @property
    def weight(self):
        return self

    def forward(self, x):
        x = rearrange(x, "d (b l) -> b l d", b=self.batch_size)
        return torch.cat(
            (
                rearrange(self.conv_proj(x), "b l d -> d (b l)"),
                rearrange(self.gate_proj(x), "b l d -> d (b l)"),
            )
        ).type(x.dtype)

    def __matmul__(self, other):
        return self.forward(other)
