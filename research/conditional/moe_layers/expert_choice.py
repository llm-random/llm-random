from typing import Literal, Union
import plotly.express as px
import torch
import torch.nn.functional as F
from fancy_einsum import einsum
from torch.nn import LayerNorm

from lizrd.core import nn
from lizrd.core.initialization import get_init_weight
from lizrd.support import ash
from lizrd.support.logging import make_histogram
from lizrd.train import checkpointing
from research.conditional.utils.layer_manager import LoggingLayer
from research.conditional.utils.layer_manager import measure_time


class ExpertGating(LoggingLayer):
    def __init__(
        self,
        n_experts,
        group_by_batch,
        softmax_ungrouped,
        softmax_over,
        topk_fraction,
        one_hot_impl,
        random_perm,
        use_torch_bmm,
        gate,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.group_by_batch = group_by_batch
        self.softmax_ungrouped = softmax_ungrouped
        self.softmax_over = softmax_over
        self.topk_fraction = topk_fraction
        self.one_hot_impl = one_hot_impl
        self.random_perm = random_perm
        self.use_torch_bmm = use_torch_bmm
        self.gate = gate
        self._checkpointed_topk_indices: Union[None, torch.Tensor] = None

    def forward(self, x: torch.Tensor, batch_size: int, seq_len: int):
        # expert embedding
        with measure_time(self, "expert_embedding"):
            if self.use_torch_bmm:
                gate = self.gate.unsqueeze(0).expand(batch_size, -1, -1)
                gate_out = torch.bmm(x, gate).permute(2, 0, 1)
                assert gate_out.shape == (self.n_experts, batch_size, seq_len)
            else:
                gate_out = einsum(
                    "batch_size seq_len dmodel, dmodel n_experts "
                    "-> n_experts batch_size seq_len ",
                    x,
                    self.gate,
                )

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
        assert topk > 0, "topk is 0, increase topk_fraction or batch_size"

        self.update_cache_for_logging("gate_softmax_all_values", gate_out)
        # choose topk tokens for each expert
        with measure_time(self, "topk"):
            checkpointing_enabled = (
                checkpointing.is_in_first_forward()
                or checkpointing.is_in_second_forward()
            )

            if (
                checkpointing.is_in_first_forward()
                and checkpointing.is_in_second_forward()
            ):
                raise NotImplementedError(
                    "Both first and second forward are = TRUE. You are probably using wrapped and nested checkpointed modules, which is not supported with ExpertGating."
                )

            if checkpointing_enabled:
                # In first forward we discard the first result of topk (topk_values)
                # and instead use gather.
                # This is needed if activation checkpointing is used, because
                # torch aligns tensors in both forward passes by the order in
                # which they are created and that is the easiest way to do that.
                if checkpointing.is_in_first_forward():
                    with torch.no_grad():
                        _, topk_indices = torch.topk(gate_out, k=topk, dim=1)
                        self._checkpointed_topk_indices = topk_indices

                if checkpointing.is_in_second_forward():
                    with torch.no_grad():
                        topk_indices = self._checkpointed_topk_indices

                topk_values = gate_out.gather(dim=1, index=topk_indices)
            else:
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
        self.n_experts = n_experts
        self.expert_size = expert_size
        self.topk_fraction = topk_fraction
        self.random_perm = random_perm
        self.group_by_batch = group_by_batch
        self.one_hot_impl = one_hot_impl
        self.softmax_ungrouped = softmax_ungrouped
        self.n_gating_heatmaps = n_gating_heatmaps
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
        gate = nn.Parameter(
            get_init_weight(
                (dmodel, n_experts),
                fan_in=dmodel,
                init_type=init_type,
                scale=init_scale,
            )
        ).requires_grad_(True)
        self.ln = LayerNorm(dmodel) if use_layer_norm else None
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

        expert_gating = ExpertGating(
            n_experts=n_experts,
            group_by_batch=group_by_batch,
            softmax_ungrouped=softmax_ungrouped,
            softmax_over=softmax_over,
            topk_fraction=topk_fraction,
            one_hot_impl=one_hot_impl,
            random_perm=random_perm,
            use_torch_bmm=use_torch_bmm,
            gate=gate,
        )
        self.expert_gating = expert_gating

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

        topk, topk_indices, topk_values = self.expert_gating(x, batch_size, seq_len)
        if self.use_torch_bmm:
            x = self.full_bmm(x, topk_indices, topk_values, batch_size)
        elif self.use_full_einsum:
            x = self.full_einsum(x, topk_indices, topk_values, batch_size)
        else:
            x, one_hot = self.extract_chosen_tokens(x, topk, topk_indices, batch_size)
            x = self.feed_forward(x, topk)
            x = self.gating_postprocess(
                x, batch_size, topk, seq_len, topk_values, topk_indices, one_hot
            )

        if self.use_layer_norm:
            with measure_time(self, "layer_norm"):
                x = self.ln(x)

        if self.group_size > 1:
            x = x.reshape(orig_bs, orig_seq_len, self.dmodel)

        return x

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

    def extract_with_linear_bmm(
        self, x: torch.Tensor, topk_indices: torch.Tensor, batch_size, weight
    ):
        with measure_time(self, "gate_preprocess_with_linear"):
            one_hot = F.one_hot(topk_indices, num_classes=batch_size).type(x.dtype)
            n_exp, topk, seq_len, _ = one_hot.shape
            _, dmodel, exp_size = weight.shape

            # BROAD here means that dimension is broadcasted, N means it's copied from left,
            # M means it's copied from right and MUL means it's multiplied
            # maybe we should rewrite it as "fancy_bmm" with similar notation to einsum

            # batch_size seq_len dmodel, n_exp topk seq_len batch_size,
            # x * one_hot (BROAD seq_len, MUL=batch_size, N=dmodel, M=(topk, n_exp))
            # -> seq_len dmodel n_exp topk
            x = x.permute(1, 2, 0)
            one_hot_perm = one_hot.permute(2, 3, 0, 1).reshape(
                seq_len, batch_size, n_exp * topk
            )
            x = torch.bmm(x, one_hot_perm).reshape(seq_len, dmodel, n_exp, topk)

            # seq_len dmodel n_exp topk, n_exp dmodel exp_size
            # x * weight (BROAD n_exp, MUL=dmodel, N=seq_len, M=exp_size)
            # -> n_exp seq_len topk exp_size
            x = x.permute(2, 0, 3, 1).reshape(n_exp, seq_len * topk, dmodel)
            x = torch.bmm(x, weight)

            assert x.shape == (n_exp, seq_len * topk, exp_size)
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

    def gating_postprocess_bmm(self, x, topk_values, one_hot, weight):
        with measure_time(self, "gating_postprocess_with_linear"):
            n_exp, exp_size, dmodel = weight.shape
            _, topk, seq_len, batch_size = one_hot.shape
            assert x.shape == (n_exp, seq_len * topk, exp_size)

            # n_exp seq_len*topk exp_size, n_exp exp_size dmodel,
            # x * weight (BROAD n_exp, MUL=exp_size, N=(seq_len, topk), M=dmodel)
            # -> n_exp seq_len topk dmodel
            x = torch.bmm(x, weight).reshape(n_exp, seq_len, topk, dmodel)

            # n_exp seq_len topk dmodel, n_exp topk seq_len,
            # x * topk_values -> n_exp seq_len topk dmodel
            x *= topk_values.permute(0, 2, 1).unsqueeze(-1)

            # n_exp seq_len topk dmodel, n_exp topk seq_len batch_size,
            # x * one_hot (BROAD seq_len, MUL=(n_exp, topk), N=dmodel, M=batch_size)
            # -> batch_size seq_len dmodel
            x = x.permute(1, 3, 0, 2).reshape(seq_len, dmodel, n_exp * topk)
            one_hot = one_hot.permute(2, 0, 1, 3).reshape(
                seq_len, n_exp * topk, batch_size
            )
            x = torch.bmm(x, one_hot).permute(2, 0, 1)

            assert x.shape == (batch_size, seq_len, dmodel)
            return x

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

    def full_bmm(
        self, x: torch.Tensor, topk_indices: torch.Tensor, topk_values, batch_size
    ):
        x, one_hot = self.extract_with_linear_bmm(
            x, topk_indices, batch_size, self.lin1_weight
        )
        x = F.relu(x)
        x = self.gating_postprocess_bmm(x, topk_values, one_hot, self.lin2_weight)
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
        return {
            "gate_softmax_topk_vals": make_histogram(
                self.logging_cache["gate_softmax_topk_vals"].flatten()
            ),
            "gate_softmax_all_values": make_histogram(
                self.logging_cache["gate_softmax_all_values"].flatten()
            ),
            "indexes_choose_counts": make_histogram(indexes_choose_counts),
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
