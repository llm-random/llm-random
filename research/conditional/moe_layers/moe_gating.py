from typing import Union, Literal
import torch
from fancy_einsum import einsum
from plotly import express as px

from lizrd.support.logging import make_histogram
from lizrd.train import checkpointing
import torch.nn.functional as F

from research.conditional.moe_layers.load_balancing_loss import (
    calculate_load_balancing_loss,
)
from research.conditional.utils.layer_manager import LoggingLayer
from research.conditional.utils.layer_manager import measure_time
from lizrd.core.initialization import get_init_fun


class MoeGating(LoggingLayer):
    def __init__(
        self,
        n_experts: int,
        group_by_batch: bool = False,
        softmax_ungrouped: bool = False,
        softmax_over: Literal["tokens", "experts"] = "tokens",
        use_torch_bmm: bool = False,
        gate=None,
        **kwargs,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.group_by_batch = group_by_batch
        self.softmax_ungrouped = softmax_ungrouped
        self.softmax_over = softmax_over
        self.use_torch_bmm = use_torch_bmm
        self.gate = gate
        self._checkpointed_topk_indices: Union[None, torch.Tensor] = None
        assert softmax_over in ["tokens", "experts"]
        assert not self.softmax_ungrouped or self.group_by_batch

    def calculate_gate(self, x, batch_size, seq_len):
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

        self.update_cache_for_logging("gate_softmax_all_values", gate_out)
        return gate_out

    def calculate_topk(self, gate_out, topk):
        is_in_first = checkpointing.is_in_first_forward()
        is_in_second = checkpointing.is_in_second_forward()
        checkpointing_enabled = is_in_first or is_in_second
        if is_in_first and is_in_second:
            raise NotImplementedError(
                "Both first and second forward are = TRUE. You are probably using wrapped and nested checkpointed modules, which is not supported with ExpertGating."
            )
        if checkpointing_enabled:
            # In first forward we discard the first result of topk (topk_values)
            # and instead use gather.
            # This is needed if activation checkpointing is used, because
            # torch aligns tensors in both forward passes by the order in
            # which they are created and that is the easiest way to do that.

            with torch.no_grad():
                if is_in_first:
                    _, topk_indices = torch.topk(gate_out, k=topk, dim=1)
                    self._checkpointed_topk_indices = topk_indices
                if is_in_second:
                    topk_indices = self._checkpointed_topk_indices

            topk_values = gate_out.gather(dim=1, index=topk_indices)
        else:
            topk_values, topk_indices = torch.topk(gate_out, k=topk, dim=1)
        return topk_indices, topk_values


class ExpertGating(MoeGating):
    def __init__(
        self,
        topk_fraction: float,
        one_hot_impl: bool = False,
        random_perm: bool = False,
        n_gating_heatmaps: int = 4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.topk_fraction = topk_fraction
        self.one_hot_impl = one_hot_impl
        self.random_perm = random_perm
        self.n_gating_heatmaps = n_gating_heatmaps
        assert (
            not one_hot_impl or self.group_by_batch
        ), "Not implemented, would require a lot of memory"

    def forward(self, x: torch.Tensor, batch_size: int, seq_len: int):
        # expert embedding
        gate_out = self.calculate_gate(x, batch_size, seq_len)

        topk = round(self.topk_fraction * gate_out.shape[1])
        assert topk > 0, "topk is 0, increase topk_fraction or batch_size"

        # choose topk tokens for each expert
        with measure_time(self, "topk"):
            topk_indices, topk_values = self.calculate_topk(gate_out, topk)

        if self.group_by_batch and not self.one_hot_impl:
            with measure_time(self, "indexing_change"):
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

    def log_heavy(self):
        if "topk_indices" not in self.logging_cache:
            return {}

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

        uf_gate_out = (
            {
                f"gating_heatmap_{i}": make_heatmap(
                    self.logging_cache["unflatten_gate_out"], i
                )
                for i in range(min(self.n_gating_heatmaps, self.n_experts))
            }
            if "unflatten_gate_out" in self.logging_cache
            else {}
        )
        return {
            "gate_softmax_topk_vals": make_histogram(
                self.logging_cache["gate_softmax_topk_vals"].flatten()
            ),
            "gate_softmax_all_values": make_histogram(
                self.logging_cache["gate_softmax_all_values"].flatten()
            ),
            "indexes_choose_counts": make_histogram(indexes_choose_counts),
            **uf_gate_out,
        }


class TokenGating(MoeGating):
    def __init__(
        self,
        dmodel: int,
        n_experts: int,
        capacity_factor: float,
        load_balancing_loss_weight: float,
        init_type: str,
        init_scale: float,
        routing_top_k: int = 1,
        use_einsum: bool = False,
        **kwargs,
    ):
        init = get_init_fun(init_type=init_type, init_scale=init_scale)
        gate = init(shape=(dmodel, n_experts), fan_in=dmodel)

        super().__init__(
            n_experts,
            group_by_batch=False,
            softmax_ungrouped=False,
            softmax_over="experts",
            use_torch_bmm=not use_einsum,
            gate=gate,
        )

        self.dmodel = dmodel
        self.capacity_factor = capacity_factor
        self.load_balancing_loss_weight = load_balancing_loss_weight
        self.use_einsum = use_einsum
        self.routing_top_k = routing_top_k

    def forward(self, x: torch.Tensor):
        # x is (batch, seq_len, dmodel)
        batch_size, seq_len, _ = x.shape
        n_tokens = batch_size * seq_len
        capacity = min(
            int(self.capacity_factor * n_tokens * self.routing_top_k / self.n_experts),
            n_tokens,
        )
        self.update_cache_for_logging("n_tokens", torch.Tensor([n_tokens]))

        gate_out = self.calculate_gate(x, batch_size, seq_len).T
        assert gate_out.shape == (n_tokens, self.n_experts)

        with measure_time(self, "choose_expert"):
            expert_index, expert_gate = self.calculate_topk(
                gate_out, self.routing_top_k
            )

        self.update_cache_for_logging("gate_softmax_values", expert_gate)
        self.update_cache_for_logging("max_indices", expert_index)

        return self.apply_capacity(capacity, expert_index, gate_out, n_tokens)

    def calculate_balancing_loss(self, gate_out, expert_mask):
        with measure_time(self, "calculate aux loss"):
            tokens_per_expert = expert_mask.sum(dim=0, dtype=gate_out.dtype)
            load_balancing_loss = calculate_load_balancing_loss(
                self.load_balancing_loss_weight,
                gate_out,
                tokens_per_expert,
                use_einsum=self.use_einsum,
            )
        if "load_balancing_losses" not in self.forward_pass_cache:
            self.forward_pass_cache["load_balancing_losses"] = [load_balancing_loss]
        else:
            self.forward_pass_cache["load_balancing_losses"].append(load_balancing_loss)
        self.update_cache_for_logging("tokens_per_expert", tokens_per_expert)
        self.update_cache_for_logging("load_balancing_loss", load_balancing_loss)

    def apply_capacity(self, capacity, expert_index, gate_out, n_tokens):
        # create a mask telling if a token is assigned to an expert
        with measure_time(self, "create_expert_mask"):
            expanded_expert_mask = F.one_hot(expert_index, num_classes=self.n_experts)
            assert expanded_expert_mask.shape == (
                n_tokens,
                self.routing_top_k,
                self.n_experts,
            )
            expert_mask = expanded_expert_mask.sum(dim=1)
            assert expert_mask.shape == (n_tokens, self.n_experts)

        # now apply fixed capacity: for a given expert we can have only capacity tokens
        with measure_time(self, "experts_lists"):
            (
                top_tokens_per_expert_values,
                top_tokens_per_expert_indices,
            ) = expert_mask.topk(k=capacity, dim=0)

        self.log_dropped_tokens(
            top_tokens_per_expert_values,
            top_tokens_per_expert_indices,
            expert_mask,
            n_tokens,
        )
        # from a list of finally chosen tokens, create a mask with their respective values
        expert_values = (
            torch.gather(gate_out, 0, top_tokens_per_expert_indices)
            * top_tokens_per_expert_values
        )
        self.calculate_balancing_loss(gate_out, expert_mask)
        return top_tokens_per_expert_indices, expert_values

    def log_dropped_tokens(
        self,
        top_tokens_per_expert_values,
        top_tokens_per_expert_indices,
        expert_mask,
        n_tokens,
    ):
        # TODO this below is just for logging, we maybe should remove it
        with measure_time(self, "create_truncated_mask"):
            truncated_expert_mask = torch.zeros_like(expert_mask)
            truncated_expert_mask.scatter_(
                dim=0,
                index=top_tokens_per_expert_indices,
                src=top_tokens_per_expert_values,
            )
        n_selected_tokens = truncated_expert_mask.sum().item()
        self.update_cache_for_logging(
            "dropped_tokens_ratio",
            ((n_tokens * self.routing_top_k) - n_selected_tokens)
            / (n_tokens * self.routing_top_k),
        )

    def log_light(self):
        return {
            "dropped_tokens_ratio": self.logging_cache["dropped_tokens_ratio"],
            "load_balancing_loss": self.logging_cache["load_balancing_loss"],
        }

    def log_heavy(self):
        return {
            "gate_softmax_all_values": make_histogram(
                self.logging_cache["gate_softmax_all_values"].flatten()  # move
            ),
            "tokens_per_expert_counts": make_histogram(
                self.logging_cache["tokens_per_expert"]
            ),
        }


def make_heatmap(tensor, expert_num, **kwargs):
    logits_for_expert = tensor[expert_num]
    batch_size, seq_len = logits_for_expert.shape
    flatten_dist = logits_for_expert.flatten()
    dist_for_expert = torch.softmax(flatten_dist.float(), dim=-1)
    dist_for_expert = dist_for_expert.reshape(batch_size, seq_len)
    return px.imshow(dist_for_expert.detach().cpu().numpy(), **kwargs)
