import torch
import torch.nn as nn

from lizrd.core import misc
from lizrd.support import ash
from research.reinitialization.pruner import Pruner


@ash.check('... inp -> ... out')
class ReinitLinear(misc.Linear):
    """Linear layer with pruning"""
    def __init__(self, d_in, d_out, prune_scheduler, **kwargs):
        print('Reinit linear created')
        super().__init__(d_in, d_out, **kwargs)
        self.mask = nn.parameter.Parameter(torch.empty(self.weight.size()), requires_grad=False)
        self.mask.fill_(1)

    def forward(self, x: torch.Tensor):
        A = self.weight * self.mask
        res = misc.einsum('... i, o i -> ... o', x, A) + self.bias
        return res

    def prune_unstr(self, prob: float):
        mask = torch.ones_like(self.weight, requires_grad=False)
        probs = torch.rand_like(self.weight)
        mask[probs <= prob] = 0
        self.mask.data = self.mask.data * mask
        n_zero = torch.numel(self.mask) - torch.count_nonzero(self.mask)
        print(f'Pruned. Percent of zeros in self.mask: {n_zero * 100 / torch.numel(self.mask)}')


@ash.check('... d -> ... d')
class ReinitFF(nn.Module):
    """Feedforward layer (with bottleneck) for pruning/reinitialization
    """
    def __init__(self, dmodel: int, dff: int, pruner: Pruner):
        super().__init__()
        self.linears = nn.Sequential(
            ReinitLinear(dmodel, dff, pruner),
            nn.ReLU(inplace=True),
            ReinitLinear(dff, dmodel, pruner)
        )
        pruner.register(self.linears[0])
        pruner.register(self.linears[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linears(x)
