import torch
import torch.nn as nn

from lizrd.core import misc
from lizrd.support import ash


@ash.check('... inp -> ... out')
class ReinitLinear(misc.Linear):
    """Linear layer with pruning"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = torch.ones_like(self.weight.data, requires_grad=False)

    def forward(self, x: torch.Tensor):
        A = self.weight.data * self.mask
        return torch.einsum('ij, jk -> ik', [A, x]) + self.bias.data

    def prune_unstr(self, prob: float):
        mask = torch.ones_like(self.weight.data)
        probs = torch.rand_like(self.weight)
        mask[probs <= prob] = 0
        self.mask = self.mask * mask


@ash.check('... d -> ... d')
class ReinitFF(nn.Module):
    """Feedforward layer (with bottleneck) for pruning/reinitialization
    """
    def __init__(self, dmodel: int, dff: int):
        super().__init__()
        self.linears = nn.Sequential(
            ReinitLinear(dmodel, dff),
            nn.ReLU(inplace=True),
            ReinitLinear(dff, dmodel)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linears(x)

    def prune_unstr(self, prob: float):
        self.linears[0].prune_unstr()
        self.linears[2].prune_unstr()
