import torch
import torch.nn as nn
import torch.nn.functional as F

from lizrd.core import misc
from lizrd.support import ash
from research.reinitialization.pruner import Pruner


class RandomPruneLayer:
    """Base class for layers with random pruning"""
    def _prepare_mask(self, size: torch.Size):
        self.mask = nn.parameter.Parameter(torch.empty(size), requires_grad=False)
        self.mask.fill_(1)

    def prune(self, prob: float):
        mask = torch.ones_like(self.mask, requires_grad=False)
        probs = torch.rand_like(self.mask)
        mask[probs <= prob] = 0
        self.mask.data = self.mask.data * mask

        n_zero = torch.numel(self.mask) - torch.count_nonzero(self.mask)
        print(f'Pruned. Percent of zeros in self.mask: {n_zero * 100 / torch.numel(self.mask)}')


@ash.check('... inp -> ... out')
class PruneLinear(misc.Linear, RandomPruneLayer):
    """Linear layer with pruning"""
    def __init__(self, d_in, d_out, **kwargs):
        super().__init__(d_in, d_out, **kwargs)
        self._prepare_mask(self.weight.size())

    def forward(self, x: torch.Tensor):
        A = self.weight * self.mask
        res = misc.einsum('... i, o i -> ... o', x, A) + self.bias
        return res


@ash.check('... d -> ... d')
class UnstructPruneFF(nn.Module):
    def __init__(self, dmodel: int, dff: int, pruner: Pruner):
        super().__init__()
        self.lin1 = PruneLinear(dmodel, dff)
        self.lin2 = PruneLinear(dff, dmodel)
        pruner.register(self.lin1)
        pruner.register(self.lin2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x


@ash.check('... d -> ... d')
class StructPruneFF(nn.Module, RandomPruneLayer):
    def __init__(self, dmodel: int, dff: int, pruner: Pruner):
        super().__init__()
        self.lin1 = nn.Linear(dmodel, dff)
        self.lin2 = nn.Linear(dff, dmodel)
        self._prepare_mask(torch.Size([dff]))
        pruner.register(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = misc.einsum('... i, i -> ... i', x, self.mask)
        x = F.relu(x)
        x = self.lin2(x)
        return x
