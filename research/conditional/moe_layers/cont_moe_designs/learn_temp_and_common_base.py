import dataclasses

import torch

from lizrd.core import misc, nn
from research.conditional.moe_layers.continuous_moe import ContinuousMoeBaseClass
from research.conditional.utils.misc_tools import stable_softmax_temperature


@dataclasses.dataclass(eq=False, repr=False)
class ContinuousMoEFinal(ContinuousMoeBaseClass):
    """
    learnable temperature,
    either shared by experts or not,
    either shared for merge and emit or not
    """

    share_by_experts: bool = True
    share_by_emit_merge: bool = True

    def get_merge_and_emit_weights(self, x):
        merge_logits = misc.einsum(
            "B S c d, d e -> B S e c", x, self.controller_merge + self.controller_base
        )
        self.cache("merge_logits", merge_logits)
        merge_weights = stable_softmax_temperature(merge_logits, self.temperature_merge)
        self.cache("merge_weights", merge_weights)
        emit_logits = misc.einsum(
            "B S c d, d e -> B S e c", x, self.controller_emit + self.controller_base
        )
        self.cache("emit_logits", emit_logits)
        emit_weights = stable_softmax_temperature(emit_logits, self.temperature_emit)
        self.cache("emit_weights", emit_weights)
        return merge_weights, emit_weights

    def init_parameters(self):
        if self.share_by_experts:
            if self.share_by_emit_merge:
                self.temperature_emit = nn.Parameter(torch.ones(1))
                self.temperature_merge = self.temperature_emit
            else:
                self.temperature_emit = nn.Parameter(torch.ones(1))
                self.temperature_merge = nn.Parameter(torch.ones(1))
        else:
            if self.share_by_emit_merge:
                self.temperature_emit = nn.Parameter(torch.ones(self.n_experts, 1))
                self.temperature_merge = self.temperature_emit
            else:
                self.temperature_emit = nn.Parameter(torch.ones(self.n_experts, 1))
                self.temperature_merge = nn.Parameter(torch.ones(self.n_experts, 1))

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
        log["temperature_merge"] = self.temperature_merge.data.flatten().tolist()
        log["temperature_emit"] = self.temperature_emit.data.flatten().tolist()
        return log
