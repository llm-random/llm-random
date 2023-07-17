import numpy as np
import torch.nn
from plotly import express as px

from lizrd.core import misc, nn
from research.conditional.utils.layer_manager import LoggingLayer, measure_time
from functools import partial
from performer_pytorch.performer_pytorch import gaussian_orthogonal_random_matrix, generalized_kernel


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
    def __init__(self, dmodel, dff, kernel_r, no_batch=False, redraw_projections_interval=10,
                 kernel_type="relu", use_bias=False,):
        super().__init__()
        self.dmodel = dmodel
        self.kernel_r = kernel_r
        self.dff = dff
        self.use_bias = use_bias
        self.no_batch = no_batch
        self.redraw_projections_interval = redraw_projections_interval
        self.current_projection_count = 0
        self.logging_ff_pre_relu = misc.Linear(dmodel, dff)
        self.activation = resolve_kernel_type(kernel_type)
        self.logging_ff_post_relu = misc.Linear(dff, dmodel)
        self.fast_attention = FastAttention(dmodel=self.dmodel, nb_features=self.kernel_r, kernel_fn=self.activation)
        self.K = lambda: self.logging_ff_pre_relu.weight.reshape(1, 1, self.dff, self.dmodel)
        self.V = lambda: self.logging_ff_post_relu.weight.T.reshape(1, 1, self.dff, self.dmodel)
        # TODO, here reverse bias on init
        self.add_bias1 = (lambda x: x + self.logging_ff_pre_relu.bias) \
            if self.use_bias and self.logging_ff_pre_relu.bias is not None else lambda x: x
        self.add_bias2 = (lambda x: x + self.logging_ff_post_relu.bias) \
            if self.use_bias and self.logging_ff_post_relu.bias is not None else lambda x: x

    def check_redraw_projections(self, device):
        if self.current_projection_count >= self.redraw_projections_interval:
            self.current_projection_count = 0
            self.fast_attention.redraw_projection_matrix(device)
        self.current_projection_count += 1

    def forward(self, x):
        with measure_time(self, "redraw_projections"):
            self.check_redraw_projections(x.device)
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


class FastAttention(LoggingLayer):
    def __init__(self, dmodel, nb_features=None, ortho_scaling=0, kernel_fn=nn.ReLU()):
        super().__init__()
        self.dmodel = dmodel
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows=self.nb_features,
                                         nb_columns=dmodel, scaling=ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)
        self.kernel_fn = kernel_fn

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def linear_attention(self, q, k, v):
        with measure_time(self, "lin_attn_d"):
            k_cumsum = k.sum(dim=-2)
            D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
        with measure_time(self, "lin_attn_k_v"):
            context = torch.einsum('...nd,...ne->...de', k, v)
        with measure_time(self, "lin_attn_q"):
            out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
        return out

    def forward(self, q, k, v):
        create_kernel = partial(generalized_kernel, kernel_fn=self.kernel_fn, projection_matrix=self.projection_matrix,
                                device=q.device)

        with measure_time(self, "kernel_q"):
            q = create_kernel(q)
        with measure_time(self, "kernel_k"):
            k = create_kernel(k)

        with measure_time(self, "lin_attn"):
            out = self.linear_attention(q, k, v)
        return out