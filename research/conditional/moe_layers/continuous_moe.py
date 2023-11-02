import dataclasses
from typing import Union

import einops
import numpy as np
import torch
from plotly import express as px

from lizrd.core import misc, nn
import lizrd.core.initialization
from research.conditional.utils.misc_tools import stable_softmax_temperature, entropy
from research.conditional.utils.layer_manager import LoggingLayer


@dataclasses.dataclass(eq=False, repr=False)
class ContinuousMoeBaseClass(LoggingLayer):
    """
    1. Groups tokens into groups of fixed size,
    2. Each expert independently aggregates tokens within a group (aggregate means take a weighted combination, weights sum to 1) into a single token,
    3. Each expert processes the token constructed above to output a token of size dmodel
    4. The mapped token is then redistributed to the original tokens, with weights determined by the expert's weighting from step 2.
    """

    dm: int
    dff: int
    n_experts: int
    group_size: int
    sparsity_dim: int
    temperature: float
    init_type: str
    init_scale: float
    expert_size: Union[int, None]
    use_opt_einsum: bool = False
    flop_matched: bool = False
    emit_softmax_over_experts: bool = False

    def __post_init__(self):
        super().__init__()
        if self.flop_matched:
            assert (
                self.dff == 4 * self.dm
            ), f"dff = {self.dff} is not equal to 4*dm = {4*self.dm} as in vanilla transformer"
            self.dff *= self.group_size
        if self.expert_size is None:
            assert (
                self.dff % self.n_experts == 0
            ), f"dff = {self.dff} is not divisible by n_experts = {self.n_experts}"
            print(
                f"expert_size is None, setting it to dff // n_experts = {self.dff // self.n_experts}"
            )
            self.expert_size = self.dff // self.n_experts
        self.init_core_parameters()
        self.init_additional_parameters()
        self.original_group_size = self.group_size

    def forward(self, x):
        x = self.rearrange_for_grouping(x)
        merge_weights, emit_weights = self.get_merge_and_emit_weights(x)
        x = self.merge_map_emit(x, merge_weights, emit_weights)
        x = self.reshape_into_original(x)
        return x * (self.group_size / self.original_group_size)

    def rearrange_for_grouping(self, x):
        """
        :param x: normal input tensor of shape (B, S, dmodel)
        :return: x transposed so that the dimension to group over is second last
        """
        if self.sparsity_dim == 0:
            x = torch.permute(x, [1, 0, 2])  # x = x.transpose(0,1)
            return x
        elif self.sparsity_dim == 1:
            raise NotImplementedError
        else:
            raise NotImplementedError

    def get_merge_and_emit_weights(self, x):
        # shape of x is free_dimension, aggr_dimension, dmodel
        merge_logits = torch.matmul(
            x.view(x.shape[0], -1, self.group_size, self.dm), self.controller
        )
        self.update_cache_for_logging("merge_logits", merge_logits)
        # shape of merge_logits is free_dimension, agrr_dimension // group_size, group_size, n_experts
        temp_merge, temp_emit = self.get_temperature()
        merge_weights = stable_softmax_temperature(merge_logits, temp_merge, dim=-2)
        # on default we use the same weights for emitting and merging, but if the temperature is learnable or we want to take softmax over experts for emitting, we will use different weights
        if isinstance(temp_merge, torch.nn.Parameter) or self.emit_softmax_over_experts:
            softmax_dim = -1 if self.emit_softmax_over_experts else -2
            emit_weights = stable_softmax_temperature(
                merge_logits, temp_emit, dim=softmax_dim
            )
        else:
            emit_weights = merge_weights
        self.update_cache_for_logging("merge_weights", merge_weights)
        self.update_cache_for_logging("emit_weights", emit_weights)
        return merge_weights, emit_weights

    def get_temperature(self):
        return self.temperature, self.temperature

    def merge_map_emit(self, x, merge_weights, emit_weights):
        # x shape is free_dimension, aggr_dimension, dmodel
        # merge_weights shape is free_dimension, aggr_dimension // group_size, group_size, n_experts
        x = torch.matmul(
            merge_weights.transpose(-1, -2),
            x.view(x.size(0), -1, self.group_size, x.size(-1)),
        )
        # x shape is free_dimension, aggr_dimension // group_size, n_experts, dmodel ||| lin1 shape is n_experts, dmodel, expert_size
        x = torch.bmm(x.view(-1, self.n_experts, x.size(-1)).transpose(0, 1), self.lin1)
        x = torch.relu_(x)
        # x shape is n_experts, free_dimension * aggr_dimension // group_size, expert_size ||| lin2 shape is n_experts, expert_size, dmodel
        x = torch.bmm(x, self.lin2)
        # x shape is n_experts, free_dimension * aggr_dimension // group_size, dmodel ||| merge_weights shape is free_dimension, aggr_dimension // group_size, group_size, n_experts
        # view x to be n_experts, free_dimension, aggr_dimension // group_size, dmodel
        # permute it to be free_dimension, aggr_dimension // group_size, n_experts, dmodel
        x = (
            torch.matmul(
                emit_weights,
                x.view(x.size(0), emit_weights.size(0), -1, x.size(-1)).permute(
                    1, 2, 0, 3
                ),
            )
            .view(emit_weights.size(0), -1, x.size(-1))
            .transpose(1, 2)
        )
        return x

    def reshape_into_original(self, x):
        if self.sparsity_dim == 0:
            # sequence dimension is the new "batch size" when you think about it
            x = x.permute(2, 0, 1)
            # shape is free_dimension, aggr_dimension, dmodel
            return x
        elif self.sparsity_dim == 1:
            raise NotImplementedError
        else:
            raise NotImplementedError

    def init_core_parameters(self):
        # lin1 is parameter, one dimension for experts of size dmodel to dff/n_experts
        self.lin1 = nn.Parameter(
            misc.get_init_weight(
                (self.n_experts, self.dm, self.expert_size),
                fan_in=self.dm,
                init_type=self.init_type,
                scale=self.init_scale,
            )
        )

        self.lin2 = nn.Parameter(
            misc.get_init_weight(
                (self.n_experts, self.expert_size, self.dm),
                fan_in=self.expert_size,
                init_type=self.init_type,
                scale=self.init_scale,
            )
        )
        # controller: send a token of size dmodel to n_experts scalars
        self.controller = nn.Parameter(
            misc.get_init_weight(
                (self.dm, self.n_experts),
                fan_in=self.dm,
                init_type=self.init_type,
                scale=self.init_scale,
            )
        )

    def init_additional_parameters(self):
        pass

    def log_light(self):
        return {}

    def log_heavy(self):
        log = {}
        if self.group_size == 1:
            return log

        merge_logits = self.logging_cache["merge_logits"]
        merge_weights = self.logging_cache["merge_weights"]
        emit_weights = self.logging_cache["emit_weights"]
        print("shapes are", merge_weights.shape, emit_weights.shape)

        max_entropy = np.log(self.group_size)

        merge_entropy_dim = -2
        if self.emit_softmax_over_experts:
            emit_entropy_dim = -1
        else:
            emit_entropy_dim = -2

        merge_weights_sum = merge_weights.sum(dim=merge_entropy_dim)
        emit_weights_sum = emit_weights.sum(dim=emit_entropy_dim)
        assert torch.allclose(
            merge_weights_sum, torch.ones_like(merge_weights_sum)
        ), f"merge_weights_sum = {merge_weights_sum} does not sum to 1 along dim {merge_entropy_dim}"
        assert torch.allclose(
            emit_weights_sum, torch.ones_like(emit_weights_sum)
        ), f"emit_weights_sum = {emit_weights_sum} does not sum to 1 along dim {emit_entropy_dim}"

        for name, weights, entropy_dim in [
            ("merge_weights", merge_weights, merge_entropy_dim),
            ("emit_weights", emit_weights, emit_entropy_dim),
        ]:
            log[f"{name}/mean"] = weights.mean()
            log[f"{name}/std"] = weights.std()
            normalized_entropy = entropy(weights, dim=entropy_dim) / max_entropy
            log[f"{name}/normalised_entropy/mean"] = normalized_entropy.mean()
            log[f"{name}/normalised_entropy/mean"] = normalized_entropy.std()

        log[f"logits/mean"] = merge_logits.mean()
        log[f"logits/std"] = merge_logits.std()

        if "time" in self.logging_cache:
            instr_names = list(self.logging_cache["time"].keys())
            instr_times = list(self.logging_cache["time"].values())
            times_fig = px.bar(x=instr_names, y=instr_times)
            log["time"] = times_fig

        return log


class ContinuousMoE(ContinuousMoeBaseClass):
    pass


class LegacyContinuousMoE(ContinuousMoeBaseClass):
    def forward(self, x):
        x = self.rearrange_for_grouping(x)
        merge_weights, emit_weights = self.get_merge_and_emit_weights(x)
        x = self.merge_map_emit(x, merge_weights, emit_weights)
        x = self.reshape_into_original(x)
        return x * (self.group_size * 1.0 / self.original_group_size)

    def rearrange_for_grouping(self, x):
        """
        :param x: normal input tensor of shape (B, S, dmodel)
        :return: x reshaped so that one of dimensions is split into groups of size self.group_size, (the dimension is determined by self.sparsity_dim)
        """
        # we want to split the input into groups of size self.group_size according to sparsity_dim
        if self.sparsity_dim == 0:
            # gather tokens from the same position in each sequence (mixes data from different examples within a batch)
            x = einops.rearrange(x, "(B g) S d -> B S g d", g=self.group_size)
        elif self.sparsity_dim == 1:
            # gather tokens from the same sequence (does not mix data from different examples within a batch)
            x = einops.rearrange(x, "B (S g) d -> B S g d", g=self.group_size)
        else:
            raise NotImplementedError("sparsity_dim must be 0 or 1")
        return x

    def get_merge_and_emit_weights(self, x):
        merge_logits = misc.einsum("B S g d, d e -> B S e g", x, self.controller)
        self.update_cache_for_logging("merge_logits", merge_logits)
        merge_weights = stable_softmax_temperature(merge_logits, self.temperature)
        self.update_cache_for_logging("merge_weights", merge_weights)
        return merge_weights, merge_weights

    def merge_map_emit(self, x, merge_weights, emit_weights):
        x = misc.einsum(
            "B S c d, B S e c, d e f -> B S e f",
            x,
            merge_weights,
            self.lin1,
            use_opt_einsum=self.use_opt_einsum,
        )
        x = torch.relu_(x)
        x = misc.einsum(
            "B S e f, d e f, B S e c -> B S c d",
            x,
            self.lin2,
            emit_weights,
            use_opt_einsum=self.use_opt_einsum,
        )
        return x

    def reshape_into_original(self, x):
        if self.sparsity_dim == 0:
            out = einops.rearrange(x, "B S g d -> (B g) S d")
        elif self.sparsity_dim == 1:
            out = einops.rearrange(x, "B S g d -> B (S g) d")
        else:
            raise NotImplementedError("sparsity_dim must be 0 or 1")
        return out

    def init_core_parameters(self):
        # lin1 is parameter, one dimension for experts of size dmodel to dff/n_experts
        self.lin1 = nn.Parameter(
            lizrd.core.initialization.get_init_weight(
                (self.dm, self.n_experts, self.expert_size),
                fan_in=self.dm,
                init_type=self.init_type,
                scale=self.init_scale,
            )
        )

        self.lin2 = nn.Parameter(
            lizrd.core.initialization.get_init_weight(
                (self.dm, self.n_experts, self.expert_size),
                fan_in=self.expert_size,
                init_type=self.init_type,
                scale=self.init_scale,
            )
        )
        # controller: send a token of size dmodel to n_experts scalars
        self.controller = nn.Parameter(
            lizrd.core.initialization.get_init_weight(
                (self.dm, self.n_experts),
                fan_in=self.dm,
                init_type=self.init_type,
                scale=self.init_scale,
            )
        )
