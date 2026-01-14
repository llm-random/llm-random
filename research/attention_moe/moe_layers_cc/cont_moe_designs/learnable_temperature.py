import dataclasses

import torch

import torch.nn as nn
from research.conditional.moe_layers.continuous_moe import ContinuousMoeBaseClass


@dataclasses.dataclass(eq=False, repr=False)
class ContinuousMoEAdaTemp(ContinuousMoeBaseClass):
    """
    learnable temperature,
    either shared by experts or not,
    either shared for merge and emit or not
    """

    share_by_experts: bool = True
    share_by_emit_merge: bool = True

    def init_additional_parameters(self):
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

    def log_heavy(self):
        log = super().log_heavy()
        log[
            "merge_weights/merge_temperature"
        ] = self.temperature_merge.data.flatten().tolist()
        log[
            "merge_weights/emit_temperature"
        ] = self.temperature_emit.data.flatten().tolist()
        return log
