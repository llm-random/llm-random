import torch.nn
from plotly import express as px

from lizrd.core import misc
from lizrd.core.misc import resolve_activation_name
from research.conditional.utils.layer_manager import LoggingLayer, measure_time
from functools import partial
from performer_pytorch.performer_pytorch import (
    gaussian_orthogonal_random_matrix,
    generalized_kernel,
    softmax_kernel,
)


def create_kernel_base(kernel_type, normalization):
    if kernel_type == "softmax":
        return lambda x, is_query=False, **a: softmax_kernel(x, is_query=is_query, **a)
    else:
        return lambda x, is_query=False, **a: generalized_kernel(
            x,
            normalize_data=normalization,
            kernel_fn=resolve_activation_name(kernel_type),
            **a
        )


class FCKernelized(LoggingLayer):
    def __init__(
        self,
        dmodel,
        dff,
        kernel_r,
        kernel_type="relu",
        redraw_projections_interval=10,
        no_kernel_norm=False,
        no_average_attn=False,
        nystrom=False,
        xfavor=False,
        use_bias=False,
        ortho_scaling=0,
    ):
        super().__init__()
        self.dmodel = dmodel
        self.nb_features = kernel_r
        self.dff = dff
        self.use_bias = use_bias
        self.redraw_projections_interval = redraw_projections_interval
        self.current_projection_count = 0
        self.logging_ff_pre_relu = misc.Linear(dmodel, dff)
        self.logging_ff_post_relu = misc.Linear(dff, dmodel)
        self.average_attn = not no_average_attn

        self.K = lambda: self.logging_ff_pre_relu.weight.reshape(1, 1, dff, dmodel)
        self.V = lambda: self.logging_ff_post_relu.weight.T.reshape(1, 1, dff, dmodel)
        # TODO, here reverse bias on init
        self.add_bias1 = (
            (lambda x: x + self.logging_ff_pre_relu.bias)
            if self.use_bias and self.logging_ff_pre_relu.bias is not None
            else lambda x: x
        )
        self.add_bias2 = (
            (lambda x: x + self.logging_ff_post_relu.bias)
            if self.use_bias and self.logging_ff_post_relu.bias is not None
            else lambda x: x
        )
        self.custom_attn = not nystrom and not xfavor

        if nystrom:
            assert not xfavor
            from xformers.components.attention import NystromAttention

            self.fast_attention = NystromAttention(
                num_heads=self.dmodel, num_landmarks=self.nb_features, dropout=0.1
            )
            return
        if xfavor:
            from xformers.components.attention import FavorAttention
            from xformers.components.attention.feature_maps import FeatureMapType

            self.fast_attention = FavorAttention(
                dim_head=self.dmodel,
                dim_features=self.nb_features,
                feature_map_type=FeatureMapType.SMOrf,
                iter_before_redraw=self.redraw_projections_interval,
            )
            return

        self.create_projection = partial(
            gaussian_orthogonal_random_matrix,
            nb_rows=self.nb_features,
            nb_columns=dmodel,
            scaling=ortho_scaling,
        )
        projection_matrix = self.create_projection()
        self.register_buffer("projection_matrix", projection_matrix)

        create_kernel = create_kernel_base(kernel_type, not no_kernel_norm)
        self.create_kernel = lambda x, **y: create_kernel(
            x, device=x.device, projection_matrix=self.projection_matrix, **y
        )
        self.fast_attention = self.fast_attention_custom

    def check_redraw_projections(self, device):
        with measure_time(self, "redraw_projections"):
            if (
                self.current_projection_count >= self.redraw_projections_interval
                and self.custom_attn
            ):
                self.current_projection_count = 0
                self.redraw_projection_matrix(device)
            self.current_projection_count += 1

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, x):
        self.check_redraw_projections(x.device)
        with measure_time(self, "fc_kern_prep"):
            bs = x.shape[0]
            Q = self.add_bias1(x).reshape(1, 1, -1, self.dmodel)
        with measure_time(self, "fc_kern_fast_attn"):
            Y = self.fast_attention(Q, self.K(), self.V())
        return self.add_bias2(Y).reshape(bs, -1, self.dmodel)

    def fast_attention_custom(self, q, k, v):
        with measure_time(self, "kernel_q"):
            q = self.create_kernel(q, is_query=True)
        with measure_time(self, "kernel_k"):
            k = self.create_kernel(k)
        with measure_time(self, "lin_attn"):
            out = self.linear_attention(q, k, v)
        return out

    def linear_attention(self, q, k, v):
        if self.average_attn:
            with measure_time(self, "lin_attn_d"):
                sum = k.sum(dim=-2)
                D_inv = 1.0 / torch.einsum("...nd,...d->...n", q, sum.type_as(q))
        with measure_time(self, "lin_attn_k_v"):
            context = torch.einsum("...nd,...ne->...de", k, v)
        with measure_time(self, "lin_attn_q"):
            if self.average_attn:
                out = torch.einsum("...de,...nd,...n->...ne", context, q, D_inv)
            else:
                out = torch.einsum("...de,...nd->...ne", context, q)
        return out

    def log_heavy(self):
        instr_names = list(self.cached_data["time"].keys())
        instr_times = list(self.cached_data["time"].values())
        times_fig = px.bar(x=instr_names, y=instr_times)
        out = {"instruction_times_plot": times_fig}
        out.update(self.cached_data["time"])
        return out
