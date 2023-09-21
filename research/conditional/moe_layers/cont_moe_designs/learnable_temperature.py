import dataclasses

import torch

from lizrd.core import misc, nn
from research.conditional.moe_layers.continuous_moe import ContinuousMoeBaseClass
from research.conditional.utils.misc_tools import stable_softmax_temperature


@dataclasses.dataclass(eq=False, repr=False)
class ContinuousMoEAdaTemp(ContinuousMoeBaseClass):
    """
    learnable temperature,
    either shared by experts or not,
    either shared for merge and emit or not
    """

    separate_temp_for_experts: bool = False
    separate_temp_for_emit_merge: bool = False

    def get_merge_and_emit_weights(self, x):
        merge_logits = misc.einsum("B S c d, d e -> B S e c", x, self.controller)
        self.update_cache_for_logging("merge_logits", merge_logits)
        merge_weights = stable_softmax_temperature(merge_logits, self.temperature_merge)
        self.update_cache_for_logging("merge_weights", merge_weights)
        emit_logits = misc.einsum("B S c d, d e -> B S e c", x, self.controller)
        self.update_cache_for_logging("emit_logits", emit_logits)
        emit_weights = stable_softmax_temperature(emit_logits, self.temperature_emit)
        self.update_cache_for_logging("emit_weights", emit_weights)
        return merge_weights, emit_weights

    def init_parameters(self):
        if self.separate_temp_for_experts:
            if self.separate_temp_for_emit_merge:
                self.temperature_emit = nn.Parameter(torch.ones(self.n_experts, 1))
                self.temperature_merge = nn.Parameter(torch.ones(self.n_experts, 1))
            else:
                self.temperature_emit = nn.Parameter(torch.ones(self.n_experts, 1))
                self.temperature_merge = self.temperature_emit
        else:
            if self.separate_temp_for_emit_merge:
                self.temperature_emit = nn.Parameter(torch.ones(1))
                self.temperature_merge = nn.Parameter(torch.ones(1))
            else:
                self.temperature_emit = nn.Parameter(torch.ones(1))
                self.temperature_merge = self.temperature_emit

        self.lin1 = nn.Parameter(
            misc.get_init_weight(
                (self.dm, self.n_experts, self.expert_size), fan_in=self.dm
            )
        )
        self.lin2 = nn.Parameter(
            misc.get_init_weight(
                (self.dm, self.n_experts, self.expert_size), fan_in=self.expert_size
            )
        )
        self.controller = nn.Parameter(
            misc.get_init_weight((self.dm, self.n_experts), fan_in=self.dm)
        )

    def log_heavy(self):
        log = self.super().log_heavy()
        log[
            "merge_weights/merge_temperature"
        ] = self.temperature_merge.data.flatten().tolist()
        log[
            "merge_weights/emit_temperature"
        ] = self.temperature_emit.data.flatten().tolist()
        return log
