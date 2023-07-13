import numpy as np
import torch.nn
from plotly import express as px

from lizrd.core import misc, nn
from research.conditional.utils.layer_manager import LoggingLayer, measure_time
from performer_pytorch import FastAttention


def resolve_kernel_type(kernel_type):
    if kernel_type == "relu":
        return nn.ReLU()
    elif kernel_type == "gelu":
        return torch.nn.GELU()
    elif kernel_type == "tanh":
        return torch.nn.Tanh()
    else:
        raise ValueError(f"Unrecognized kernel type: {kernel_type}")


class FCKernelized(LoggingLayer):
    def __init__(self, dmodel, dff, kernel_r, no_batch=False, kernel_type="relu", use_bias=False):
        super().__init__()
        self.dmodel = dmodel
        self.kernel_r = kernel_r
        self.dff = dff
        self.use_bias = use_bias
        self.no_batch = no_batch
        self.logging_ff_pre_relu = misc.Linear(dmodel, dff)
        self.activation = resolve_kernel_type(kernel_type)
        self.logging_ff_post_relu = misc.Linear(dff, dmodel)
        self.fast_attention = FastAttention(dim_heads=1, nb_features=self.kernel_r, generalized_attention=True,
                                            kernel_fn=self.activation)
        self.K = lambda: self.logging_ff_pre_relu.weight.reshape(1, 1, self.dff, self.dmodel)
        self.V = lambda: self.logging_ff_post_relu.weight.T.reshape(1, 1, self.dff, self.dmodel)
        # TODO, here reverse bias on init
        self.add_bias1 = (lambda x: x + self.logging_ff_pre_relu.bias) \
            if self.use_bias and self.logging_ff_pre_relu.bias is not None else lambda x: x
        self.add_bias2 = (lambda x: x + self.logging_ff_post_relu.bias) \
            if self.use_bias and self.logging_ff_post_relu.bias is not None else lambda x: x

    def forward(self, x):
        with measure_time(self, "fc_kern_prep"):
            bs = x.shape[0]
            Q = self.add_bias1(x).reshape(1, 1, -1, self.dmodel)
        with measure_time(self, "fc_kern_performer"):
            Y = self.fast_attention(Q, self.K(), self.V())
        return self.add_bias2(Y).reshape(bs, -1, self.dmodel)

    def log_heavy(self):
        instr_names = list(self.cached_data["time"].keys())
        instr_times = list(self.cached_data["time"].values())
        times_fig = px.bar(x=instr_names, y=instr_times)
        out = {"instruction_times_plot": times_fig}
        out.update(self.cached_data["time"])
        return out
