from typing import Literal, Union, Optional
import plotly.express as px
import torch
import torch.nn.functional as F
from torch.nn import LayerNorm

import torch.nn as nn
from lizrd.core.initialization import get_init_weight
from research.conditional.utils.layer_manager import LoggingLayer
from research.conditional.utils.layer_manager import measure_time
from research.conditional.moe_layers.expert_choice import ExpertGating, ExpertChoiceFF


class ExpertDoubleChoiceFF(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        n_experts: int,
        expert_size: int,
        topk_fraction: float,
        init_type: Literal["kaiming_uniform", "truncated_normal"],
        init_scale: float,
        doutput: Optional[int] = None,
        random_perm: bool = False,
        group_by_batch: bool = False,
        one_hot_impl: bool = False,
        softmax_ungrouped: bool = False,
        use_full_einsum: bool = False,
        softmax_over: Literal["tokens", "experts"] = "tokens",
        n_gating_heatmaps: int = 4,
        group_size: int = 1,
        use_torch_bmm: bool = False,
        use_layer_norm: bool = True,
    ):
        """
        Args:
            dmodel: dimension of the input
            n_experts: number of experts
            expert_size: size of each expert
            topk_fraction: fraction of tokens that will be chosen for each expert
            random_perm: randomly permute tokens for experts (ablation). Note that
                network can still learn which tokens to choose,
                but not which expert to choose for token
        """
        super().__init__()

        self.dmodel = dmodel
        self.doutput = dmodel if doutput is None else doutput
        self.n_experts = n_experts
        self.expert_size = expert_size
        self.topk_fraction = topk_fraction
        self.random_perm = random_perm
        self.group_by_batch = group_by_batch
        self.one_hot_impl = one_hot_impl
        self.softmax_ungrouped = softmax_ungrouped
        self.use_full_einsum = use_full_einsum
        self.group_size = group_size
        self.use_torch_bmm = use_torch_bmm
        self.use_layer_norm = use_layer_norm

        assert (
            not self.one_hot_impl or self.group_by_batch
        ), "Not implemented, would require a lot of memory"
        assert softmax_over in ["tokens", "experts"]
        assert not self.softmax_ungrouped or self.group_by_batch
        assert not self.use_full_einsum or self.one_hot_impl  # Not implemented
        assert not self.use_torch_bmm or not self.use_full_einsum  # Not implemented

        init_weight = lambda shape, fan_in:  nn.Parameter(
            get_init_weight(
                shape,
                fan_in=fan_in,
                init_type=init_type,
                scale=init_scale,
            ),
        ).requires_grad_(True)

        self.lin1_weight = init_weight((n_experts, dmodel, expert_size), dmodel)
        self.lin2_weight = init_weight((n_experts, expert_size, self.doutput),
                                       int(n_experts * expert_size * topk_fraction))
        self.ln = LayerNorm(self.doutput) if use_layer_norm else None
        self.softmax_over = softmax_over

        create_gating = lambda in_size: ExpertGating(
            n_experts=n_experts,
            group_by_batch=group_by_batch,
            softmax_ungrouped=softmax_ungrouped,
            softmax_over=softmax_over,
            topk_fraction=topk_fraction,
            one_hot_impl=one_hot_impl,
            random_perm=random_perm,
            use_torch_bmm=use_torch_bmm,
            gate=init_weight((in_size, n_experts), fan_in=in_size),
            n_gating_heatmaps=n_gating_heatmaps,
        )

        self.expert_gating = create_gating(dmodel)
        self.expert_gating_2 = create_gating(expert_size)

    def forward(self, x: torch.Tensor):
        # x is (batch, seq_len, dmodel)
        batch_size, seq_len = x.shape[0], x.shape[1]
        orig_bs, orig_seq_len = batch_size, seq_len

        if self.group_size > 1:
            assert batch_size % self.group_size == 0
            batch_size, seq_len = (
                self.group_size,
                seq_len * (batch_size // self.group_size),
            )
            x = x.reshape(batch_size, seq_len, self.dmodel)

        x = self.full_bmm(x, batch_size)

        if self.use_layer_norm:
            with measure_time(self, "layer_norm"):
                x = self.ln(x)

        if self.group_size > 1:
            x = x.reshape(orig_bs, orig_seq_len, self.doutput)

        return x

    def route_one_linear(
        self, x: torch.Tensor, batch_size, weight, expert_gating, num,
    ):
        seq_len = x.shape[1]
        topk, topk_indices, topk_values = expert_gating(x, batch_size, seq_len)

        # emit
        with measure_time(self, "one_hot_instanciate" + num):
            one_hot = F.one_hot(topk_indices, num_classes=batch_size).type(x.dtype)
        n_exp, topk, seq_len, _ = one_hot.shape
        _, dmodel_in, dmodel_out = weight.shape

        with measure_time(self, "shuffle_preprocess_perm" + num):
            x = x.permute(1, 2, 0)
            one_hot_perm = one_hot.permute(2, 3, 0, 1).reshape(
                seq_len, batch_size, n_exp * topk
            )
        with measure_time(self, "shuffle_preprocess" + num):
            x = torch.bmm(x, one_hot_perm).reshape(seq_len, dmodel_in, n_exp, topk)

        # linear
        with measure_time(self, "lin_perm" + num):
            x = x.permute(2, 0, 3, 1).reshape(n_exp, seq_len * topk, dmodel_in)
        with measure_time(self, "lin" + num):
            x = torch.bmm(x, weight)

        assert x.shape == (n_exp, seq_len * topk, dmodel_out)

        # merge
        x = x.reshape(n_exp, seq_len, topk, dmodel_out)

        with measure_time(self, "gating_weight_mul" + num):
            x *= topk_values.permute(0, 2, 1).unsqueeze(-1)
        with measure_time(self, "shuffle_postprocess_permute" + num):
            x = x.permute(1, 0, 2, 3).reshape(seq_len, n_exp * topk, dmodel_out)
        with measure_time(self, "shuffle_postprocess" + num):
            x = torch.bmm(one_hot_perm, x).permute(1, 0, 2)

        assert x.shape == (batch_size, seq_len, dmodel_out)
        return x

    def full_bmm(
        self, x: torch.Tensor, batch_size
    ):
        x = self.route_one_linear(
            x, batch_size, self.lin1_weight, self.expert_gating, "1"
        )
        with measure_time(self, "activation"):
            x = F.relu(x)
        x = self.route_one_linear(
            x, batch_size, self.lin2_weight, self.expert_gating_2, "2"
        )
        return x

    def log_light(self):
        return dict()

    def log_time(self):
        log = {}
        if "time" in self.logging_cache:
            instr_names = list(self.logging_cache["time"].keys())
            instr_times = list(self.logging_cache["time"].values())
            if "time" in self.expert_gating.logging_cache:
                instr_names += list(self.expert_gating.logging_cache["time"].keys())
                instr_times += list(self.expert_gating.logging_cache["time"].values())
            times_fig = px.bar(x=instr_names, y=instr_times)
            log["time"] = times_fig
        return log


def make_heatmap(tensor, expert_num, **kwargs):
    logits_for_expert = tensor[expert_num]
    batch_size, seq_len = logits_for_expert.shape
    flatten_dist = logits_for_expert.flatten()
    dist_for_expert = torch.softmax(flatten_dist.float(), dim=-1)
    dist_for_expert = dist_for_expert.reshape(batch_size, seq_len)
    return px.imshow(dist_for_expert.detach().cpu().numpy(), **kwargs)
