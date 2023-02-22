from torch import nn
import torch
import torch.nn.functional as F
from lizrd.core import misc
from research.reinitialization.core.pruner import Pruner


class InverseWeightDecayFF(nn.Module):
    def __init__(
        self,
        dmodel: int,
        dff: int,
        reg_type: str,
        reg_coeff: float,
        pruner: Pruner,
    ):
        super().__init__()
        self.lin1 = misc.Linear(dmodel, dff)
        self.lin2 = misc.Linear(dff, dmodel)
        self.reg_type = reg_type
        assert reg_type in ["l1", "l2"]
        self.reg_coeff = reg_coeff

        pruner.register(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x

    @property
    def neuron_magnitudes(self) -> torch.Tensor:
        weights1 = misc.einsum("f m -> f", self.lin1.weight**2)
        weights2 = misc.einsum("m f -> f", self.lin2.weight**2)

        weights = weights1 * weights2
        return weights.flatten()

    def get_auxiliary_loss(self) -> torch.Tensor:
        magnitudes = self.neuron_magnitudes
        mean = magnitudes.mean().detach()
        if self.reg_type == "l1":
            loss = ((magnitudes < mean) * abs(magnitudes - mean)).sum()
        elif self.reg_type == "l2":
            loss = ((magnitudes < mean) * (magnitudes - mean) ** 2).sum()
        return loss * self.reg_coeff
