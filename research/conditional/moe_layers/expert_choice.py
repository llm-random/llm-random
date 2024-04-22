from typing import Literal
import plotly.express as px
import torch
import torch.nn.functional as F
from fancy_einsum import einsum
from torch.nn import LayerNorm

from lizrd.core.initialization import get_init_fun
from research.conditional.utils.layer_manager import LoggingLayer
from research.conditional.utils.layer_manager import measure_time, time_measured
from research.conditional.moe_layers.moe_gating import ExpertGating


class ExpertChoiceFF(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        n_experts: int,
        topk_fraction: float,
        init_type: Literal["kaiming_uniform", "truncated_normal"],
        init_scale: float,
        expert_inner_function: LoggingLayer,
        random_perm: bool = False,
        group_by_batch: bool = False,
        one_hot_impl: bool = False,
        softmax_ungrouped: bool = False,
        softmax_over: Literal["tokens", "experts"] = "tokens",
        n_gating_heatmaps: int = 4,
        group_size: int = 1,
        use_torch_bmm: bool = False,
        use_layer_norm: bool = True,
        principled_moe: bool = False,
    ):
        """
        Args:
            dmodel: dimension of the input
            n_experts: number of experts
            topk_fraction: fraction of tokens that will be chosen for each expert
            random_perm: randomly permute tokens for experts (ablation). Note that
                network can still learn which tokens to choose,
                but not which expert to choose for token
        """
        super().__init__()

        self.dmodel = dmodel
        self.n_experts = n_experts
        self.topk_fraction = topk_fraction
        self.random_perm = random_perm
        self.group_by_batch = group_by_batch
        self.one_hot_impl = one_hot_impl
        self.softmax_ungrouped = softmax_ungrouped
        self.group_size = group_size
        self.use_torch_bmm = use_torch_bmm
        self.use_layer_norm = use_layer_norm
        self.expert_inner_function = expert_inner_function
        self.doutput = self.expert_inner_function.doutput

        assert (
            not self.one_hot_impl or self.group_by_batch
        ), "Not implemented, would require a lot of memory"
        assert not self.softmax_ungrouped or self.group_by_batch

        init = get_init_fun(init_type=init_type, init_scale=init_scale)
        gate = init((dmodel, n_experts), dmodel)

        self.ln = self.measure(LayerNorm(self.doutput), "layer_norm", use_layer_norm)
        self.softmax_over = softmax_over

        if use_torch_bmm:
            self.extract = self.extract_bmm
            self.merge = self.merge_bmm
        elif one_hot_impl:
            self.extract = self.extract_einsum
            self.merge = self.merge_einsum
        else:
            self.extract = self.extract_index_select
            self.merge = self.merge_index_select

        self.expert_gating = ExpertGating(
            n_experts=n_experts,
            group_by_batch=group_by_batch,
            softmax_ungrouped=softmax_ungrouped,
            softmax_over=softmax_over,
            topk_fraction=topk_fraction,
            one_hot_impl=one_hot_impl,
            random_perm=random_perm,
            use_torch_bmm=use_torch_bmm,
            gate=gate,
            n_gating_heatmaps=n_gating_heatmaps,
            principled_moe=principled_moe,
        )

        self.principled_moe = principled_moe

    def forward(self, x: torch.Tensor):
        # x is (batch, seq_len, dmodel)
        batch_size, seq_len, _ = x.shape
        orig_bs, orig_seq_len = batch_size, seq_len

        if self.group_size > 1:
            assert batch_size % self.group_size == 0
            batch_size, seq_len = (
                self.group_size,
                seq_len * (batch_size // self.group_size),
            )
            x = x.reshape(batch_size, seq_len, self.dmodel)

        topk, topk_indices, topk_values = self.expert_gating(x, batch_size, seq_len)

        # from icecream import ic

        # ic(self.expert_gating.gate.shape)
        # ic(x.shape)
        # ic(topk, topk_indices.shape, topk_values.shape)

        # topk -> how many tokens each expert gets
        # topk_indices -> which tokens each expert gets (indices)
        # topk_values -> how much of each token each expert gets (values), [n_experts, topk]
        if self.principled_moe:
            x, one_hot = self.extract(x, topk, topk_indices)
            x = self.expert_inner_function(x, topk_values)
            # einsum(
            #     "n_exp topk doutput, n_exp topk -> n_exp topk doutput", x, topk_values
            # )
            x = self.merge(
                x, batch_size, topk, seq_len, topk_values, topk_indices, one_hot
            )

        else:
            x, one_hot = self.extract(x, topk, topk_indices)
            # ic(x.shape)  # n_experts, topk, dmodel
            x = self.expert_inner_function(x)
            x = self.merge(
                x, batch_size, topk, seq_len, topk_values, topk_indices, one_hot
            )

        x = self.ln(x)

        if self.group_size > 1:
            x = x.reshape(orig_bs, orig_seq_len, self.doutput)

        return x

    # extract implementations

    @time_measured("extract_einsum")
    def extract_einsum(self, x: torch.Tensor, topk, topk_indices: torch.Tensor):
        batch_size, _, _ = x.shape
        one_hot = F.one_hot(topk_indices, num_classes=batch_size).type(x.dtype)
        # one_hot is (n_experts, topk, seq_len, batch_size)
        x = einsum(
            "batch_size seq_len dmodel, n_exp topk seq_len batch_size "
            "-> n_exp topk seq_len dmodel",
            x,
            one_hot,
        )
        x = x.reshape((self.n_experts, topk, self.dmodel))
        return x, one_hot

    def extract_bmm(self, x: torch.Tensor, topk, topk_indices: torch.Tensor):
        batch_size, _, _ = x.shape
        with measure_time(self, "one_hot_instanciate"):
            one_hot = F.one_hot(topk_indices, num_classes=batch_size).type(x.dtype)
        n_exp, topk, seq_len, _ = one_hot.shape

        # BROAD here means that dimension is broadcasted, N means it's copied from left,
        # M means it's copied from right and MUL means it's multiplied
        # maybe we should rewrite it as "fancy_bmm" with similar notation to einsum

        # batch_size seq_len dmodel, n_exp topk seq_len batch_size,
        # x * one_hot (BROAD seq_len, MUL=batch_size, N=dmodel, M=(topk, n_exp))
        # -> seq_len dmodel n_exp topk
        with measure_time(self, "shuffle_preprocess_perm"):
            x = x.permute(1, 2, 0)
            one_hot_perm = one_hot.permute(2, 3, 0, 1).reshape(
                seq_len, batch_size, n_exp * topk
            )
        with measure_time(self, "shuffle_preprocess"):
            x = torch.bmm(x, one_hot_perm).reshape(seq_len, self.dmodel, n_exp, topk)

        with measure_time(self, "lin1_perm"):
            x = x.permute(2, 0, 3, 1).reshape(n_exp, seq_len * topk, self.dmodel)
        return x, one_hot_perm

    def extract_principled(
        self,
        x: torch.Tensor,
        topk,
        topk_indices: torch.Tensor,
        topk_values: torch.Tensor,
    ):
        batch_size, _, _ = x.shape
        # flatten x s. t. first dimension is tokens instead of batch_size x seq_len
        with measure_time(self, "first_flatten"):
            x = x.flatten(start_dim=0, end_dim=1)
            topk_values = topk_values.flatten()
        with measure_time(self, "index_select"):
            x = torch.index_select(x, dim=0, index=topk_indices.flatten())
            topk_values = torch.index_select(
                topk_values, dim=0, index=topk_indices.flatten()
            )
        with measure_time(self, "reshape"):
            x = x.reshape((self.n_experts, topk, self.dmodel))
        return x, None

    def extract_index_select(self, x: torch.Tensor, topk, topk_indices: torch.Tensor):
        batch_size, _, _ = x.shape
        # flatten x s. t. first dimension is tokens instead of batch_size x seq_len
        with measure_time(self, "first_flatten"):
            x = x.flatten(start_dim=0, end_dim=1)
        with measure_time(self, "index_select"):
            x = torch.index_select(x, dim=0, index=topk_indices.flatten())
        with measure_time(self, "reshape"):
            x = x.reshape((self.n_experts, topk, self.dmodel))
        return x, None

    # postprocess implementations

    @time_measured("postprocess_einsum")
    def merge_einsum(
        self, x, batch_size, topk, seq_len, topk_values, topk_indices, one_hot
    ):
        topk_per_batch = topk // seq_len
        x = x.reshape(self.n_experts, topk_per_batch, seq_len, self.doutput)
        return einsum(
            "n_exp topk_per_batch seq_len doutput, n_exp topk_per_batch seq_len, "
            "n_exp topk_per_batch seq_len batch_size "
            "-> batch_size seq_len doutput",
            x,
            topk_values,
            one_hot,
        )

    def merge_bmm(
        self, x, batch_size, topk, seq_len, topk_values, topk_indices, one_hot
    ):
        n_exp = self.n_experts
        _, topk, _ = topk_values.shape
        assert x.shape == (n_exp, seq_len * topk, self.doutput)

        x = x.reshape(n_exp, seq_len, topk, self.doutput)

        # n_exp seq_len topk dmodel, n_exp topk seq_len,
        # x * topk_values -> n_exp seq_len topk dmodel
        with measure_time(self, "gating_weight_mul"):
            x *= topk_values.permute(0, 2, 1).unsqueeze(-1)

        #  n_exp topk seq_len batch_size, n_exp seq_len topk dmodel
        # x * one_hot (BROAD seq_len, MUL=(n_exp, topk), N=batch_size, M=dmodel)
        # -> batch_size seq_len dmodel

        with measure_time(self, "shuffle_postprocess_permute"):
            x = x.permute(1, 0, 2, 3).reshape(seq_len, n_exp * topk, self.doutput)

        with measure_time(self, "shuffle_postprocess"):
            x = torch.bmm(one_hot, x).permute(1, 0, 2)

        assert x.shape == (batch_size, seq_len, self.doutput)
        return x

    def merge_index_select(
        self, x, batch_size, topk, seq_len, topk_values, topk_indices, one_hot
    ):
        # multiply by softmax
        with measure_time(self, "multiply_softmax"):
            if not self.principled_moe:
                x = einsum(
                    "n_exp topk doutput, n_exp topk -> n_exp topk doutput",
                    x,
                    topk_values,
                )

        # flatten x s. t. first dimension is tokens instead of n_experts x topk
        with measure_time(self, "second_flatten"):
            x = x.flatten(start_dim=0, end_dim=1)

        # add tokens that have been processed by more than one expert
        with measure_time(self, "add_tokens_many_experts"):
            z = (
                torch.zeros((batch_size * seq_len, self.doutput))
                .type(x.type())
                .to(x.device)
            )
            z.index_add_(dim=0, index=topk_indices.flatten().to(int), source=x)

            # reshape to (batch_size, seq_len, doutput)
            x = z.reshape((batch_size, seq_len, self.doutput))
        return x

    # logging

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
