from typing import Literal
import plotly.express as px
import torch
import torch.nn.functional as F
from fancy_einsum import einsum
from torch.nn import LayerNorm

from lizrd.core import nn
from lizrd.core.initialization import get_init_weight
from lizrd.support import ash
from lizrd.support.logging import make_histogram
from research.conditional.utils.layer_manager import LoggingLayer
from research.conditional.utils.layer_manager import measure_time


class ExpertChoiceFF(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        n_experts: int,
        expert_size: int,
        topk_fraction: float,
        init_type: Literal["kaiming_uniform", "truncated_normal"],
        init_scale: float,
        random_perm: bool = False,
        group_by_batch: bool = False,
        one_hot_impl: bool = False,
        softmax_ungrouped: bool = False,
        use_full_einsum: bool = False,
        softmax_over: Literal["tokens", "experts"] = "tokens",
        n_gating_heatmaps: int = 4,
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
        self.n_experts = n_experts
        self.expert_size = expert_size
        self.topk_fraction = topk_fraction
        self.random_perm = random_perm
        self.group_by_batch = group_by_batch
        self.one_hot_impl = one_hot_impl
        self.softmax_ungrouped = softmax_ungrouped
        self.n_gating_heatmaps = n_gating_heatmaps
        self.use_full_einsum = use_full_einsum

        assert (
            not self.one_hot_impl or self.group_by_batch
        ), "Not implemented, would require a lot of memory"
        assert softmax_over in ["tokens", "experts"]
        assert not self.softmax_ungrouped or self.group_by_batch
        assert not self.use_full_einsum or self.one_hot_impl  # Not implemented

        self.lin1_weight = nn.Parameter(
            get_init_weight(
                (n_experts, dmodel, expert_size),
                fan_in=dmodel,
                init_type=init_type,
                scale=init_scale,
            ),
        )
        self.lin2_weight = nn.Parameter(
            get_init_weight(
                (n_experts, expert_size, dmodel),
                fan_in=int(n_experts * expert_size * topk_fraction),
                init_type=init_type,
                scale=init_scale,
            )
        )
        self.gate = nn.Parameter(
            get_init_weight(
                (dmodel, n_experts),
                fan_in=dmodel,
                init_type=init_type,
                scale=init_scale,
            )
        ).requires_grad_(True)
        self.ln = LayerNorm(dmodel)
        self.softmax_over = softmax_over
        self.extract_chosen_tokens = (
            self.extract_chosen_tokens_onehot
            if one_hot_impl
            else self.extract_chosen_tokens_select
        )
        self.gating_postprocess = (
            self.gating_postprocess_onehot
            if one_hot_impl
            else self.gating_postprocess_select
        )

    def forward(self, x: torch.Tensor):
        # x is (batch, seq_len, dmodel)
        batch_size, seq_len = x.shape[0], x.shape[1]

        topk, topk_indices, topk_values = self.expert_gating(x, batch_size, seq_len)
        if self.use_full_einsum:
            x = self.full_einsum(x, topk_indices, topk_values, batch_size)
        else:
            x, one_hot = self.extract_chosen_tokens(x, topk, topk_indices, batch_size)
            x = self.feed_forward(x, topk)
            x = self.gating_postprocess(
                x, batch_size, topk, seq_len, topk_values, topk_indices, one_hot
            )

        with measure_time(self, "layer_norm"):
            x = self.ln(x)

        return x

    def expert_gating(self, x: torch.Tensor, batch_size: int, seq_len: int):
        # expert embedding
        with measure_time(self, "expert_embedding"):
            gate_out = einsum(
                "batch_size seq_len dmodel, dmodel n_experts -> n_experts batch_size seq_len ",
                x,
                self.gate,
            )
        self.update_cache_for_logging("unflatten_gate_out", gate_out)

        # each expert chooses k within dimension 1
        if not self.group_by_batch and not self.softmax_ungrouped:
            gate_out = gate_out.reshape(self.n_experts, batch_size * seq_len)

        # perform softmax either over tokens for each expert or over experts for each token
        with measure_time(self, "softmax"):
            if self.softmax_over == "tokens":
                gate_out = torch.softmax(gate_out, dim=1)
            elif self.softmax_over == "experts":
                gate_out = torch.softmax(gate_out, dim=0)

        if self.softmax_ungrouped:
            gate_out = gate_out.reshape(self.n_experts, batch_size * seq_len)

        topk = round(self.topk_fraction * gate_out.shape[1])

        self.update_cache_for_logging("gate_softmax_all_values", gate_out)
        # choose topk tokens for each expert
        with measure_time(self, "topk"):
            topk_values, topk_indices = torch.topk(gate_out, k=topk, dim=1)

        with measure_time(self, "indexing_change"):
            if self.group_by_batch and not self.one_hot_impl:
                topk *= seq_len
                # change indexing to recall to batch_size x seq_len
                row_number = torch.arange(seq_len).to(topk_indices.device)
                topk_indices = topk_indices * seq_len + row_number
                topk_indices = topk_indices.reshape(self.n_experts, topk)
                topk_values = topk_values.reshape(self.n_experts, topk)
            elif self.group_by_batch:
                topk *= seq_len

        # cache values for logging
        self.update_cache_for_logging("gate_softmax_topk_vals", topk_values)
        self.update_cache_for_logging("topk_indices", topk_indices)
        self.update_cache_for_logging("n_tokens", torch.Tensor([batch_size * seq_len]))

        # Randomly permute tokens for experts if random_perm is True
        # Note this is not total randomness, since topk values are already chosen
        if self.random_perm:
            topk_values = topk_values.flatten()[
                torch.randperm(self.n_experts * topk)
            ].reshape((self.n_experts, topk))

        return topk, topk_indices, topk_values

    def extract_chosen_tokens_onehot(
        self, x: torch.Tensor, topk, topk_indices: torch.Tensor, batch_size
    ):
        with measure_time(self, "one_hot"):
            one_hot = F.one_hot(topk_indices, num_classes=batch_size).type(x.dtype)
            # one_hot is (n_experts, topk, seq_len, batch_size)
            x = einsum(
                "batch_size seq_len dmodel, n_exp topk seq_len batch_size "
                "-> n_exp topk seq_len dmodel",
                x,
                one_hot,
            )
        with measure_time(self, "reshape"):
            x = x.reshape((self.n_experts, topk, self.dmodel))
        return x, one_hot

    def extract_with_linear(
        self, x: torch.Tensor, topk_indices: torch.Tensor, batch_size, weight
    ):
        with measure_time(self, "gate_preprocess_with_linear"):
            one_hot = F.one_hot(topk_indices, num_classes=batch_size).type(x.dtype)
            x = einsum(
                "batch_size seq_len dmodel, n_exp topk seq_len batch_size, "
                "n_exp dmodel exp_size "
                "-> n_exp topk seq_len exp_size",
                x,
                one_hot,
                weight,
            )
            return x, one_hot

    def gating_postprocess_onehot_with_linear(self, x, topk_values, one_hot, weight):
        with measure_time(self, "gating_postprocess_with_linear"):
            return einsum(
                "n_exp topk seq_len exp_size, n_exp topk seq_len, "
                "n_exp topk seq_len batch_size, n_exp exp_size dmodel"
                "-> batch_size seq_len dmodel",
                x,
                topk_values,
                one_hot,
                weight,
            )

    def full_einsum(
        self, x: torch.Tensor, topk_indices: torch.Tensor, topk_values, batch_size
    ):
        x, one_hot = self.extract_with_linear(
            x, topk_indices, batch_size, self.lin1_weight
        )
        x = F.relu(x)
        x = self.gating_postprocess_onehot_with_linear(
            x, topk_values, one_hot, self.lin2_weight
        )
        return x

    def extract_chosen_tokens_select(
        self, x: torch.Tensor, topk, topk_indices: torch.Tensor, batch_size
    ):
        # flatten x s. t. first dimension is tokens instead of batch_size x seq_len
        with measure_time(self, "first_flatten"):
            x = x.flatten(start_dim=0, end_dim=1)
        with measure_time(self, "index_select"):
            x = torch.index_select(x, dim=0, index=topk_indices.flatten())
        with measure_time(self, "reshape"):
            x = x.reshape((self.n_experts, topk, self.dmodel))
        return x, None

    def feed_forward(self, x: torch.Tensor, topk: int) -> torch.Tensor:
        # feed through ff
        with measure_time(self, "ff"):
            # lin1 maps from (n_experts, topk, dmodel) to (n_experts, topk, exp_size)
            x = einsum(
                "n_exp topk dmodel, n_exp dmodel exp_size -> n_exp topk exp_size",
                x,
                self.lin1_weight,
            )
            x = F.relu(x)

            # lin2 maps from (n_experts, topk, exp_size) to (n_experts, topk, dmodel)
            x = einsum(
                "n_exp topk exp_size, n_exp exp_size dmodel -> n_exp topk dmodel",
                x,
                self.lin2_weight,
            )
            ash.assert_shape("e k m", x, e=self.n_experts, k=topk, m=self.dmodel)
        return x

    def gating_postprocess_onehot(
        self, x, batch_size, topk, seq_len, topk_values, topk_indices, one_hot
    ):
        topk //= seq_len
        with measure_time(self, "multiply_softmax"):
            x = x.reshape(self.n_experts, topk, seq_len, self.dmodel)
            x = einsum(
                "n_exp topk seq_len dmodel, n_exp topk seq_len, n_exp topk seq_len batch_size "
                "-> batch_size seq_len dmodel",
                x,
                topk_values,
                one_hot,
            )
        return x

    def gating_postprocess_select(
        self, x, batch_size, topk, seq_len, topk_values, topk_indices, one_hot
    ):
        # multiply by softmax
        with measure_time(self, "multiply_softmax"):
            ash.assert_shape("e k", topk_values, e=self.n_experts, k=topk)
            x = einsum(
                "n_exp topk dmodel, n_exp topk -> n_exp topk dmodel", x, topk_values
            )

        # flatten x s. t. first dimension is tokens instead of n_experts x topk
        with measure_time(self, "second_flatten"):
            x = x.flatten(start_dim=0, end_dim=1)

        # add tokens that have been processed by more than one expert
        with measure_time(self, "add_tokens_many_experts"):
            z = (
                torch.zeros((batch_size * seq_len, self.dmodel))
                .type(x.type())
                .to(x.device)
            )

            z.index_add_(dim=0, index=topk_indices.flatten().to(int), source=x)

            # reshape to (batch_size, seq_len, dmodel)
            x = z.reshape((batch_size, seq_len, self.dmodel))
        return x

    def log_light(self):
        return dict()

    def log_heavy(self):
        # calculate indexes choose counts
        chosen_indexes = self.logging_cache["topk_indices"].flatten()
        chosen_indexes = torch.cat(
            (
                chosen_indexes,
                torch.Tensor([self.logging_cache["n_tokens"] - 1]).type(
                    chosen_indexes.type()
                ),
            )
        )  # make sure bincount takes into account the whole range of indexes
        indexes_choose_counts = chosen_indexes.bincount()

        # make bar plot of values cached in forward with measure_time
        instr_names = list(self.logging_cache["time"].keys())
        instr_times = list(self.logging_cache["time"].values())
        times_fig = px.bar(x=instr_names, y=instr_times)

        return {
            "gradient of gate distribution": make_histogram(self.gate.grad.flatten()),
            "gate_softmax_topk_vals": make_histogram(
                self.logging_cache["gate_softmax_topk_vals"].flatten()
            ),
            "gate_softmax_all_values": make_histogram(
                self.logging_cache["gate_softmax_all_values"].flatten()
            ),
            "indexes_choose_counts": make_histogram(indexes_choose_counts),
            "instruction_times": times_fig,
            **{
                f"gating_heatmap_{i}": make_heatmap(
                    self.logging_cache["unflatten_gate_out"], i
                )
                for i in range(min(self.n_gating_heatmaps, self.n_experts))
            },
        }


def make_heatmap(tensor, expert_num, **kwargs):
    logits_for_expert = tensor[expert_num]
    batch_size, seq_len = logits_for_expert.shape
    flatten_dist = logits_for_expert.flatten()
    dist_for_expert = torch.softmax(flatten_dist.float(), dim=-1)
    dist_for_expert = dist_for_expert.reshape(batch_size, seq_len)
    return px.imshow(dist_for_expert.detach().cpu().numpy(), **kwargs)
