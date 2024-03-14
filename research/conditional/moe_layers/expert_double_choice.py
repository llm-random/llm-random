from typing import Literal, Optional
import plotly.express as px
import torch
import torch.nn.functional as F
from torch.nn import LayerNorm

import torch.nn as nn
from lizrd.core.initialization import get_init_fun
from research.conditional.utils.layer_manager import LoggingLayer
from research.conditional.utils.layer_manager import measure_time
from research.conditional.moe_layers.moe_gating import ExpertGating
from research.conditional.moe_layers.expert_choice import ExpertChoiceFF
from research.conditional.moe_layers.token_choice import TokenChoiceFF
from research.conditional.moe_layers.expert_types import ExpertFF, ExpertGated, ExpertLinear


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
        single_route=False,
        both_from_start=False,
        use_second_ln=False,
        route_before_relu=False,
        use_mot=False,
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
        self.single_route = single_route
        self.route_start = both_from_start
        self.use_mot = use_mot
        self.use_second_ln = use_second_ln
        self.route_before_relu = route_before_relu

        assert (
            not self.one_hot_impl or self.group_by_batch
        ), "Not implemented, would require a lot of memory"
        assert softmax_over in ["tokens", "experts"]
        assert not self.softmax_ungrouped or self.group_by_batch
        assert not self.use_full_einsum or self.one_hot_impl  # Not implemented
        assert not self.use_torch_bmm or not self.use_full_einsum  # Not implemented

        init = get_init_fun(init_type=init_type, init_scale=init_scale)

        # TODO merging weights correctly?

        self.lin1_weight = init((n_experts, dmodel, expert_size), dmodel)
        self.lin2_weight = init(
            (n_experts, expert_size, self.doutput),
            int(n_experts * expert_size * topk_fraction),
        )
        self.ln = self.measure(LayerNorm(self.doutput), "layer_norm", use_layer_norm)
        self.ln_2 = self.measure(
            LayerNorm(self.doutput), "layer_norm_mid", use_second_ln
        )
        self.activation = self.measure(nn.ReLU(), "activation")
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
            gate=init((in_size, n_experts), fan_in=in_size),
            n_gating_heatmaps=n_gating_heatmaps,
        )

        self.expert_gating = create_gating(dmodel)
        if not self.single_route:
            self.expert_gating_2 = create_gating(
                dmodel if self.route_start else expert_size
            )

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

        x = self.full_double_routing(x)
        x = self.ln(x)

        if self.group_size > 1:
            x = x.reshape(orig_bs, orig_seq_len, self.doutput)

        return x

    def route_one_mot(self, x: torch.Tensor, weight, gate, num):
        n_experts, dmodel_in, dmodel_out = weight.shape
        batch_size, seq_len, dmodel_in_2 = x.shape
        n_experts_2, batch_size_2, seq_len_2 = gate.shape
        assert (
            dmodel_in == dmodel_in_2
            and batch_size == batch_size_2
            and seq_len == seq_len_2
        )

        # emit (seq_len, dmodel_in, batch_size) x (seq_len, batch_size, n_experts) -> (seq_len, dmodel_in, n_experts)
        with measure_time(self, "mot_emit" + num):
            x = torch.bmm(x.permute(1, 2, 0), gate.permute(2, 1, 0)).permute(2, 0, 1)
            assert x.shape == (n_experts, seq_len, dmodel_in)

        # linear (n_experts, seq_len, dmodel_in) x (n_experts, dmodel_in, dmodel_out) -> (n_experts, seq_len, dmodel_out)
        with measure_time(self, "mot_lin" + num):
            x = torch.bmm(x, weight)
            assert x.shape == (n_experts, seq_len, dmodel_out)

        # merge (seq_len, dmodel_out, n_experts) x (seq_len, n_experts, batch_size) -> (seq_len, dmodel_out, batch_size)
        with measure_time(self, "mot_merge" + num):
            x = torch.bmm(x.permute(1, 2, 0), gate.permute(2, 0, 1)).permute(2, 0, 1)
            assert x.shape == (batch_size, seq_len, dmodel_out)

        return x

    def route_one_linear(
        self,
        x: torch.Tensor,
        weight,
        gate,
        num,
    ):
        topk, topk_indices, topk_values = gate
        batch_size, seq_len, dmodel_in = x.shape
        n_exp, topk_2, seq_len_2 = topk_indices.shape
        _, dmodel_in_2, dmodel_out = weight.shape
        assert dmodel_in == dmodel_in_2 and seq_len == seq_len_2 and topk == topk_2

        # emit
        with measure_time(self, "one_hot_instanciate" + num):
            one_hot = F.one_hot(topk_indices, num_classes=batch_size).type(x.dtype)
        with measure_time(self, "shuffle_preprocess_perm" + num):
            x = x.permute(1, 2, 0)
            one_hot = one_hot.permute(2, 3, 0, 1).reshape(
                seq_len, batch_size, n_exp * topk
            )
        with measure_time(self, "shuffle_preprocess" + num):
            x = torch.bmm(x, one_hot).reshape(seq_len, dmodel_in, n_exp, topk)

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
            x = torch.bmm(one_hot, x).permute(1, 0, 2)

        assert x.shape == (batch_size, seq_len, dmodel_out)
        return x

    def route_one(self, x: torch.Tensor, weight, gate, num):
        if self.use_mot:
            return self.route_one_mot(x, weight, gate, num)
        else:
            return self.route_one_linear(x, weight, gate, num)

    def get_gate(self, expert_gating, x):
        batch_size, seq_len = x.shape[:2]
        if self.use_mot:
            return expert_gating.calculate_gate(x, batch_size, seq_len)
        else:
            return expert_gating(x, batch_size, seq_len)

    def full_double_routing(self, x: torch.Tensor):
        gate, gate_2 = self.get_gate(self.expert_gating, x), None
        if self.route_start:
            gate_2 = self.get_gate(self.expert_gating_2, x)
        x = self.route_one(x, self.lin1_weight, gate, "1")

        x = self.ln_2(x)
        if self.route_before_relu:
            gate_2 = self.get_gate(self.expert_gating_2, x)
        x = self.activation(x)

        if self.single_route:
            gate_2 = gate
        if gate_2 is None:
            gate_2 = self.get_gate(self.expert_gating_2, x)
        x = self.route_one(x, self.lin2_weight, gate_2, "2")
        return x

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


def get_router(routing_type, *args, **kwargs):
    if routing_type == "expert_choice":
        return ExpertChoiceFF(*args, **kwargs)
    elif routing_type == "token_choice":
        return TokenChoiceFF(*args, **kwargs)
    else:
        raise ValueError(f"Unknown routing type: {routing_type}")


class DoubleChoiceInner(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        n_experts: int,
        expert_size: int,
        topk_fraction: float,
        *args,
        **kwargs,
    ):
        self.dmodel = dmodel
        self.n_experts = n_experts
        self.expert_size = expert_size
        self.topk_fraction = topk_fraction
        self.linear_1 = ExpertLinear(dmodel, n_experts, expert_size, *args, **kwargs)
        linear_2 = ExpertLinear(expert_size, n_experts, dmodel, *args, **kwargs)
        self.router = get_router(
            dmodel, n_experts, topk_fraction,
            expert_inner_function=linear_2,
            *args, **kwargs
        )

    def forward(self, x: torch.Tensor):
        x = self.linear_1(x)
        x = self.router(x)
        return x


class DoubleChoiceFF(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        n_experts: int,
        expert_size: int,
        topk_fraction: float,
        *args,
        **kwargs,
    ):
        self.dmodel = dmodel
        self.n_experts = n_experts
        self.expert_size = expert_size
        self.topk_fraction = topk_fraction
        inner_router = DoubleChoiceInner(
            dmodel, n_experts, expert_size, topk_fraction, *args, **kwargs
        )
        self.router = get_router(
            dmodel, n_experts, topk_fraction,
            expert_inner_function=inner_router,
            *args, **kwargs
        )

    def forward(self, x: torch.Tensor):
        x = self.router(x)
        return x
