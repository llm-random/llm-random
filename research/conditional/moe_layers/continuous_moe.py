import dataclasses
from typing import Union

import einops
import numpy as np
import torch
from plotly import express as px

from lizrd.core import misc, nn
from lizrd.support.logging import make_histogram
from research.conditional.utils.misc_tools import stable_softmax_temperature, entropy
from research.conditional.utils.layer_manager import LoggingLayer, measure_time


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
    expert_size: Union[int, None]
    use_opt_einsum: bool = False
    flop_matched: bool = False

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
        self.init_parameters()

    def forward(self, x):
        x = self.reshape_into_token_groups(x)
        merge_weights, emit_weights = self.get_merge_and_emit_weights(x)
        x = self.separate_map_emit(x, merge_weights, emit_weights)
        x = self.reshape_into_original(x)
        return x

    def reshape_into_token_groups(self, x):
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
        with measure_time(self, "merge_process"):
            x = misc.einsum(
                "B S c d, B S e c, d e f -> B S e f",
                x,
                merge_weights,
                self.lin1,
                use_opt_einsum=self.use_opt_einsum,
            )
        with measure_time(self, "relu"):
            x = torch.relu_(x)
        with measure_time(self, "emit_process"):
            x = misc.einsum(
                "B S e f, d e f, B S e c -> B S c d",
                x,
                self.lin2,
                emit_weights,
                use_opt_einsum=self.use_opt_einsum,
            )
        return x

    def separate_map_emit(self, x, merge_weights, emit_weights):
        # input_shape = x.shape
        x = misc.einsum("B S c d, B S e c -> B S e d", x, merge_weights)
        S = x.shape[1]
        x = einops.rearrange(x, "a b c d -> c (a b) d")
        with measure_time(self, "operation_1"):
            x = torch.bmm(x, self.lin1)
        with measure_time(self, "relu"):
            x = torch.relu_(x)
        with measure_time(self, "operation_2"):
            x = torch.bmm(x, self.lin2)
        # with measure_time(self, "operation_1_again"):
        #     x = torch.bmm(x, self.lin1)
        # with measure_time(self, "operation_1_again_again"):
        #     x = torch.bmm(x, self.lin1)
        # with measure_time(self, "operation_2_again"):
        #     x = torch.bmm(x, self.lin2)
        # with measure_time(self, "operation_2_again_again"):
        #     x = torch.bmm(x, self.lin2)
        x = einops.rearrange(x, "c (a b) d -> a b c d", b=S)
        x = misc.einsum("B S e d, B S e c -> B S c d", x, emit_weights)
        # assert x.shape == input_shape
        return x

    def reshape_into_original(self, x):
        if self.sparsity_dim == 0:
            out = einops.rearrange(x, "B S g d -> (B g) S d")
        elif self.sparsity_dim == 1:
            out = einops.rearrange(x, "B S g d -> B (S g) d")
        else:
            raise NotImplementedError("sparsity_dim must be 0 or 1")
        return out

    def init_parameters(self):
        # lin1 is parameter, one dimension for experts of size dmodel to dff/n_experts

        self.lin2 = nn.Parameter(
            misc.get_init_weight(
                (self.n_experts, self.expert_size, self.dm), fan_in=self.expert_size
            )
        )
        self.lin3 = nn.Parameter(
            misc.get_init_weight(
                (self.n_experts, self.expert_size, self.dm), fan_in=self.expert_size
            )
        )
        self.lin1 = nn.Parameter(
            misc.get_init_weight(
                (self.n_experts, self.dm, self.expert_size), fan_in=self.dm
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
        merge_weights = torch.flatten(
            self.logging_cache["merge_weights"], start_dim=0, end_dim=-2
        )
        merge_logits = torch.flatten(
            self.logging_cache["merge_logits"], start_dim=0, end_dim=-2
        )
        sample_weight_distros = merge_weights[:5]
        sample_logits_distros = merge_logits[:5]

        for i, sample in enumerate(sample_weight_distros):
            sample = sample.sort(descending=True).values
            sample = sample.tolist()
            fig = px.bar(x=range(len(sample)), y=sample, title=f"sample {i}")
            log[f"merge_weights/sample_{i}"] = fig

        for i, sample in enumerate(sample_logits_distros):
            sample = sample.sort(descending=True).values
            sample = sample.tolist()
            fig = px.bar(x=range(len(sample)), y=sample, title=f"sample {i}")
            log[f"merge_logits/sample_{i}"] = fig

        ent = entropy(merge_weights)
        max_entropy = np.log(self.group_size)
        normalised_ent = ent / max_entropy
        log["merge_weights/normalised_entropy"] = make_histogram(
            normalised_ent, title="merge logits entropy (normalised to [0,1])"
        )

        # make bar plot of values cached in forward with measure_time
        instr_names = list(self.logging_cache["time"].keys())
        instr_times = list(self.logging_cache["time"].values())
        times_fig = px.bar(x=instr_names, y=instr_times)
        log["operation_times"] = times_fig

        return log


class ContinuousMoE(ContinuousMoeBaseClass):
    pass
