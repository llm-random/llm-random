import torch
import torch.nn as nn
import torch.nn.functional as F

from lizrd.core import misc
from lizrd.support import ash
from research.reinitialization.core.pruner import Pruner


def mask_by_score(
    mask: torch.Tensor, scores: torch.Tensor, n_to_mask: int
) -> torch.Tensor:
    """`n_to_mask` `mask` entries with the lowest `scores` will be pruned."""
    assert mask.shape == scores.shape

    mask = torch.clone(mask).detach()
    scores = torch.clone(scores).detach()

    # Determine indices of least important elements
    scores[mask == 0] = torch.inf

    topk = torch.topk(torch.abs(scores).view(-1), n_to_mask, largest=False)

    mask.view(-1)[topk.indices] = 0
    return mask


def create_mask(size: torch.Size) -> torch.nn.parameter.Parameter:
    mask = nn.parameter.Parameter(torch.ones(size), requires_grad=False)
    return mask


@ash.check("... inp -> ... out")
class PruneLinear(misc.Linear):
    """Linear layer with pruning"""

    def __init__(self, d_in, d_out, **kwargs):
        super().__init__(d_in, d_out, **kwargs)
        self.mask = create_mask(self.weight.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = self.weight * self.mask
        res = misc.einsum("... i, o i -> ... o", x, A) + self.bias
        return res

    def prune(self, prob: float):
        self.mask.data = mask_by_score(
            self.mask, torch.rand_like(self.mask), round(self.mask.numel() * prob)
        )


@ash.check("... d -> ... d")
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


@ash.check("... d -> ... d")
class StructPruneFF(nn.Module):
    def __init__(self, dmodel: int, dff: int, pruner: Pruner):
        super().__init__()
        self.lin1 = nn.Linear(dmodel, dff)
        self.lin2 = nn.Linear(dff, dmodel)
        self.mask = create_mask(torch.Size([dff]))
        pruner.register(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = misc.einsum("... i, i -> ... i", x, self.mask)
        x = F.relu(x)
        x = self.lin2(x)
        return x

    def prune(self, prob: float):
        self.mask.data = mask_by_score(
            self.mask, torch.rand_like(self.mask), round(self.mask.numel() * prob)
        )


class LogFF(nn.Module):
    def log(self):
        pass


class MagnitudePruneLinear(misc.Linear):
    """Linear layer with magnitude pruning"""

    def __init__(self, d_in, d_out, **kwargs):
        super().__init__(d_in, d_out, **kwargs)
        self.mask = create_mask(self.weight.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = self.weight * self.mask
        res = misc.einsum("... i, o i -> ... o", x, A) + self.bias
        return res

    def prune(self, prob: float):
        self.mask.data = mask_by_score(
            self.mask, self.weight, round(self.mask.numel() * prob)
        )


@ash.check("... d -> ... d")
class UnstructMagnitudePruneFF(nn.Module):
    def __init__(self, dmodel: int, dff: int, pruner: Pruner):
        super().__init__()
        self.lin1 = MagnitudePruneLinear(dmodel, dff)
        self.lin2 = MagnitudePruneLinear(dff, dmodel)
        pruner.register(self.lin1)
        pruner.register(self.lin2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x


@ash.check("... d -> ... d")
class StructMagnitudePruneFF(nn.Module):
    def __init__(self, dmodel: int, dff: int, pruner: Pruner):
        super().__init__()
        self.lin1 = nn.Linear(dmodel, dff)
        self.lin2 = nn.Linear(dff, dmodel)
        self.mask = create_mask(torch.Size([dff]))
        pruner.register(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = misc.einsum("... i, i -> ... i", x, self.mask)
        x = F.relu(x)
        x = self.lin2(x)
        return x

    def prune(self, prob: float):
        weights1 = misc.einsum("i o -> i", self.lin1.weight**2)
        weights2 = misc.einsum("o i -> i", self.lin2.weight**2)
        scores = weights1 * weights2
        self.mask.data = mask_by_score(
            self.mask, scores, round(self.mask.numel() * prob)
        )


@ash.check("... d -> ... d")
class MaskedFF(nn.Module):
    """Fully masked Feed-Forward layer"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)
