import torch
import torch.nn as nn

from lizrd.core import misc
from lizrd.support import ash


@ash.check('... inp -> ... out')
class ReinitLinear(misc.Linear):
    """Linear layer with pruning"""
    def __init__(self, d_in, d_out, prune_scheduler, **kwargs):
        print('Reinit linear created')
        super().__init__(d_in, d_out, **kwargs)
        self.prune_scheduler = prune_scheduler
        self.mask = nn.parameter.Parameter(torch.empty(self.weight.size()), requires_grad=False)
        self.mask.fill_(1)

    def forward(self, x: torch.Tensor):
        prune_prob = self.prune_scheduler.step()
        if prune_prob:
            self.prune_unstr(prune_prob)
            n_zero = torch.numel(self.mask) - torch.count_nonzero(self.mask)
            print(f'Percent of zeros in self.mask: {n_zero * 100 / torch.numel(self.mask)}')
        A = self.weight * self.mask
        res = misc.einsum('... i, o i -> ... o', x, A) + self.bias
        return res

    def prune_unstr(self, prob: float):
        mask = torch.ones_like(self.weight, requires_grad=False)
        probs = torch.rand_like(self.weight)
        mask[probs <= prob] = 0
        self.mask.data = self.mask.data * mask


@ash.check('... d -> ... d')
class ReinitFF(nn.Module):
    """Feedforward layer (with bottleneck) for pruning/reinitialization
    """
    def __init__(self, dmodel: int, dff: int, prune_scheduler):
        super().__init__()
        self.linears = nn.Sequential(
            ReinitLinear(dmodel, dff, prune_scheduler),
            nn.ReLU(inplace=True),
            ReinitLinear(dff, dmodel, prune_scheduler)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linears(x)

    def prune_unstr(self, prob: float):
        self.linears[0].prune_unstr(prob)
        self.linears[2].prune_unstr(prob)
