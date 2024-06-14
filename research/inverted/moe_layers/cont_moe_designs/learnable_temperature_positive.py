import dataclasses

import torch

from research.inverted.moe_layers.cont_moe_designs.learnable_temperature import (
    ContinuousMoEAdaTemp,
)


@dataclasses.dataclass(eq=False, repr=False)
class ContinuousMoEAdaTempPositive(ContinuousMoEAdaTemp):
    """
    learnable temperature,
    just like ContinuousMoEAdaTemp, but with temperature > 0
    inherit from ContinuousMoEAdaTemp
    """

    def get_temperature(self):
        return torch.exp(self.temperature_merge - 1.0), torch.exp(
            self.temperature_emit - 1.0
        )
