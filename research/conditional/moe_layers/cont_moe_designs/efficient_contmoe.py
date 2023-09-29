import torch
from plotly import express as px

from lizrd.core import nn, misc
from research.conditional.moe_layers.continuous_moe import ContinuousMoeBaseClass
from research.conditional.utils.misc_tools import stable_softmax_temperature


class EfficientContMoE(ContinuousMoeBaseClass):
    def forward(self, x):
        x = self.reshape_into_token_groups(x)
        if self.max_group_size:
            merge_weights, emit_weights = self.get_merge_and_emit_weights(x)
        else:
            merge_weights, emit_weights = self.manygroups_get_merge_and_emit_weights(x)
        x = self.merge_map_emit(x, merge_weights, None)
        x = self.reshape_into_original(x)
        return x

    def reshape_into_token_groups(self, x):
        """
        :param x: normal input tensor of shape (B, S, dmodel)
        :return: x reshaped so that one of dimensions is split into groups of size self.group_size, (the dimension is determined by self.sparsity_dim)
        """

        if self.group_size == x.shape[self.sparsity_dim]:
            self.max_group_size = True
        else:
            self.max_group_size = False

        if self.sparsity_dim == 0:
            x = torch.permute(x, [1, 0, 2])
            return x
        elif self.sparsity_dim == 1:
            return x
        else:
            raise NotImplementedError

    def get_merge_and_emit_weights(self, x):
        # shape of x is free_dimension, aggr_dimension, dmodel
        merge_logits = torch.matmul(x, self.controller)
        # shape of merge_logits is free_dimension, aggr_dimension, n_experts
        merge_weights = stable_softmax_temperature(
            merge_logits, self.temperature, dim=1
        )
        assert merge_weights.shape == (
            x.shape[0],
            self.group_size,
            self.n_experts,
        ), f"merge_weights shape is {merge_weights.shape}, instead of 'x.shape[0],self.group_size,self.n_experts ':{(x.shape[0], self.group_size, self.n_experts)}"
        # for ease in merge_map_emit, we permute the dimensions so that the group_size dimension is last
        merge_weights = torch.permute(merge_weights, [0, 2, 1])
        return merge_weights, merge_weights

    def manygroups_get_merge_and_emit_weights(self, x):
        # shape of x is free_dimension, aggr_dimension, dmodel
        merge_logits = torch.matmul(
            x.view(x.shape[0], -1, self.group_size, self.dm), self.controller
        )
        # shape of merge_logits is free_dimension agrr_dimension/group_size, group_size, n_experts
        merge_weights = (
            stable_softmax_temperature(merge_logits, self.temperature, dim=-1)
            .view(x.shape[0], -1, self.n_experts)
            .permute(0, 2, 1)
        )
        return merge_weights, merge_weights

    def merge_map_emit(self, x, merge_weights, emit_weights):
        # x shape is free_dimension, aggr_dimension, dmodel
        # merge_weights shape is free_dimension, n_experts, aggr_dimension
        x = torch.bmm(merge_weights, x)
        # x shape is free_dimension, n_experts, dmodel ||| lin1 shape is n_experts, dmodel, expert_size
        x = torch.bmm(x.permute(1, 0, 2), self.lin1)
        x = torch.relu_(x)
        # x shape is n_experts, free_dimension, expert_size ||| lin2 shape is n_experts, expert_size, dmodel
        x = torch.bmm(x, self.lin2)
        # x shape is n_experts, free_dimension, dmodel ||| merge_weights shape is free_dimension, aggr_dimension, n_experts
        # after permute, x shape is free_dimension, dmodel, n_experts, and merge_weights shape is free_dimension, n_experts, aggr_dimension
        x = torch.bmm(x.permute(1, 2, 0), merge_weights)
        return x

    def reshape_into_original(self, x):
        if self.sparsity_dim == 0:
            # sequence dimension is the new "batch size" when you think about it
            x = x.permute(2, 0, 1)
            # shape is free_dimension, aggr_dimension, dmodel
            return x
        elif self.sparsity_dim == 1:
            return x
        else:
            raise NotImplementedError

    def init_parameters(self):
        # lin1 is parameter, one dimension for experts of size dmodel to dff/n_experts
        self.lin1 = nn.Parameter(
            misc.get_init_weight(
                (self.n_experts, self.dm, self.expert_size), fan_in=self.dm
            )
        )

        self.lin2 = nn.Parameter(
            misc.get_init_weight(
                (self.n_experts, self.expert_size, self.dm),
                fan_in=self.expert_size,
            )
        )
        # controller: send a token of size dmodel to n_experts scalars
        self.controller = nn.Parameter(
            misc.get_init_weight((self.dm, self.n_experts), fan_in=self.dm)
        )

    def log_light(self):
        return {}

    def log_heavy(self):
        log = {}
        if self.group_size == 1:
            return log

        if "time" not in self.logging_cache:
            return log
        instr_names = list(self.logging_cache["time"].keys())
        instr_times = list(self.logging_cache["time"].values())
        times_fig = px.bar(x=instr_names, y=instr_times)
        log["time"] = times_fig

        return log
