import dataclasses

import torch
from lizrd.core import misc, nn
from research.conditional.moe_layers.continuous_moe import ContinuousMoeBaseClass
from research.conditional.utils.misc_tools import stable_softmax_temperature


@dataclasses.dataclass(eq=False, repr=False)
class ContinuousMoECommonWeightedParameters(ContinuousMoeBaseClass):
    """
    Both the merge and emit logits are computed as base + merge/emit. The init variance is set so the sum of base + merge/emit is as usual
    """

    def get_merge_and_emit_weights(self, x):
        merge_combined_parameters = (
            self.parameters_matrix_weight * self.controller_merge
            + (1 - self.parameters_matrix_weight) * self.controller_base
        )
        merge_logits = misc.einsum(
            "B S c d, d e -> B S e c", x, merge_combined_parameters
        )
        self.cache("merge_logits", merge_logits)
        merge_weights = stable_softmax_temperature(merge_logits, self.temperature)
        self.cache("merge_weights", merge_weights)

        emit_combined_parameters = (
            self.parameters_matrix_weight * self.controller_emit
            + (1 - self.parameters_matrix_weight) * self.controller_base
        )
        emit_logits = misc.einsum(
            "B S c d, d e -> B S e c", x, emit_combined_parameters
        )
        self.cache("emit_logits", emit_logits)
        emit_weights = stable_softmax_temperature(emit_logits, self.temperature)
        self.cache("emit_weights", emit_weights)
        return merge_weights, emit_weights

    def init_parameters(self):
        self.parameters_matrix_weight = nn.Parameter(torch.Tensor([0.5]))
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
        self.controller_base = nn.Parameter(
            misc.get_init_weight((self.dm, self.n_experts), fan_in=self.dm * 2)
        )
        self.controller_merge = nn.Parameter(
            misc.get_init_weight((self.dm, self.n_experts), fan_in=self.dm * 2)
        )
        self.controller_emit = nn.Parameter(
            misc.get_init_weight((self.dm, self.n_experts), fan_in=self.dm * 2)
        )

    def log_light(self):
        log = super().log_light()
        log["parameters_matrix_weight"] = self.parameters_matrix_weight
        return log
