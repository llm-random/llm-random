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

    share_by_experts: bool = True
    share_by_emit_merge: bool = True

    def get_temperature(self):
        return torch.exp(self.temperature_merge), torch.exp(self.temperature_emit)

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
