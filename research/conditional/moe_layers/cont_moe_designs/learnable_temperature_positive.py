import dataclasses

import torch

from lizrd.core import misc, nn
from research.conditional.moe_layers.continuous_moe import ContinuousMoeBaseClass


@dataclasses.dataclass(eq=False, repr=False)
class ContinuousMoEAdaTempPositive(ContinuousMoeBaseClass):
    """
    learnable temperature,
    either shared by experts or not,
    either shared for merge and emit or not
    """

    separate_temp_for_experts: bool = False
    separate_temp_for_emit_merge: bool = False

    def get_temperature(self):
        return torch.exp(self.temperature_merge), torch.exp(self.temperature_emit)

    def init_parameters(self):
        if self.separate_temp_for_experts:
            if self.separate_temp_for_emit_merge:
                self.temperature_emit = nn.Parameter(torch.zeros(self.n_experts, 0))
                self.temperature_merge = nn.Parameter(torch.zeros(self.n_experts, 0))
            else:
                self.temperature_emit = nn.Parameter(torch.zeros(self.n_experts, 0))
                self.temperature_merge = self.temperature_emit
        else:
            if self.separate_temp_for_emit_merge:
                self.temperature_emit = nn.Parameter(torch.zeros(0))
                self.temperature_merge = nn.Parameter(torch.zeros(0))
            else:
                self.temperature_emit = nn.Parameter(torch.zeros(0))
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
        log = super().log_heavy()
        log[
            "merge_weights/merge_temperature"
        ] = self.temperature_merge.data.flatten().tolist()
        log[
            "merge_weights/emit_temperature"
        ] = self.temperature_emit.data.flatten().tolist()
        return log
