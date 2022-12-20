import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_

from lizrd.core.misc import Linear
from lizrd.support import ash
from research.reinitialization.core.pruner import Pruner
from lizrd.core import misc


class RandomUnstructRecycleFF(nn.Module):
    """Feed-Forward layer with recycling"""

    def __init__(self, dmodel: int, dff: int, pruner: Pruner):
        super().__init__()
        self.lin1 = Linear(dmodel, dff)
        self.lin2 = Linear(dff, dmodel)
        pruner.register(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x

    def prune(self, prob: float):
        self._recycle_linear(self.lin1, prob)
        self._recycle_linear(self.lin2, prob)

    def _recycle_linear(self, layer: Linear, prob: float):
        # create mask and new_weights
        mask = torch.ones(layer.weight.size())
        new_weights = kaiming_uniform_(torch.empty_like(layer.weight), a=math.sqrt(5))
        new_weights *= 3**0.5

        # prepare mask according to prob
        probs = torch.rand_like(mask)
        mask[probs <= prob] = 0

        # apply mask to weights
        layer.weight.data = mask * layer.weight.data + (1 - mask) * new_weights


class RandomStructRecycleFF(nn.Module):
    """Feedforward layer with recycling"""

    def __init__(self, dmodel: int, dff: int, pruner: Pruner):
        super().__init__()
        self.lin1 = Linear(dmodel, dff)
        self.lin2 = Linear(dff, dmodel)
        self.dff = dff
        pruner.register(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x

    def prune(self, prob: float):
        # create mask
        mask = torch.ones(self.dff)

        # prepare mask according to prob
        probs = torch.rand_like(mask)
        mask[probs <= prob] = 0

        # apply mask to lin1
        new_weights = kaiming_uniform_(
            torch.empty_like(self.lin1.weight), a=math.sqrt(5)
        )
        new_weights *= 3**0.5

        self.lin1.weight.data = misc.einsum(
            "f, f m -> f m", mask, self.lin1.weight.data
        ) + misc.einsum("f, f m -> f m", 1 - mask, new_weights)
        self.lin1.bias.data = misc.einsum("f, f -> f", mask, self.lin1.bias.data)

        # apply mask to lin2
        # bias is intentionally not recycled here
        new_weights = kaiming_uniform_(
            torch.empty_like(self.lin2.weight), a=math.sqrt(5)
        )
        new_weights *= 3**0.5

        self.lin2.weight.data = misc.einsum(
            "f, m f -> m f", mask, self.lin2.weight.data
        ) + misc.einsum("f, m f -> m f", 1 - mask, new_weights)


class UnstructMagnitudeRecycleFF(nn.Module):
    def __init__(self, dmodel: int, dff: int, pruner: Pruner):
        super().__init__()
        self.lin1 = Linear(dmodel, dff)
        self.lin2 = Linear(dff, dmodel)
        pruner.register(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x

    def prune(self, prob: float):
        self._recycle_linear(self.lin1, prob)
        self._recycle_linear(self.lin2, prob)

    def _recycle_linear(self, layer: Linear, prob: float):
        # create mask and new_weights
        weights = layer.weight.data
        mask = torch.ones_like(weights, requires_grad=False)
        new_weights = kaiming_uniform_(
            torch.empty_like(layer.weight), a=math.sqrt(5 * 3)
        )

        # Determine indices of less important weights
        weights = layer.weight.data
        n_els_weights = torch.numel(weights)
        n_to_prune = round(prob * n_els_weights)
        topk = torch.topk(torch.abs(weights).view(-1), n_to_prune, largest=False)

        mask.view(-1)[topk.indices] = 0
        layer.weight.data = mask * layer.weight.data + (1 - mask) * new_weights


class StructMagnitudeRecycleFF(nn.Module):
    def __init__(self, dmodel: int, dff: int, pruner: Pruner):
        super().__init__()
        self.lin1 = Linear(dmodel, dff)
        self.lin2 = Linear(dff, dmodel)
        self.dff = dff
        pruner.register(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x

    def prune(self, prob: float):
        device = self.lin1.weight.device

        # create mask
        mask = torch.ones(self.dff).to(device)

        # prepare mask
        weights1 = misc.einsum("f m -> f", self.lin1.weight**2)
        weights2 = misc.einsum("m f -> f", self.lin2.weight**2)
        weights = weights1 * weights2
        n_els_weights = torch.numel(weights)
        assert n_els_weights == self.dff
        n_to_prune = round(prob * n_els_weights)
        topk = torch.topk(torch.abs(weights).view(-1), n_to_prune, largest=False)
        mask[topk.indices] = 0

        # apply mask to lin1
        new_weights = kaiming_uniform_(
            torch.empty_like(self.lin1.weight), a=math.sqrt(5)
        )
        new_weights *= 3**0.5

        self.lin1.weight.data = misc.einsum(
            "f, f m -> f m", mask, self.lin1.weight.data
        ) + misc.einsum("f, f m -> f m", 1 - mask, new_weights)
        self.lin1.bias.data = misc.einsum("f, f -> f", mask, self.lin1.bias.data)

        # apply mask to lin2
        new_weights = kaiming_uniform_(
            torch.empty_like(self.lin2.weight), a=math.sqrt(5)
        )
        new_weights *= 3**0.5

        self.lin2.weight.data = misc.einsum(
            "f, m f -> m f", mask, self.lin2.weight.data
        ) + misc.einsum("f, m f -> m f", 1 - mask, new_weights)


class RetrainRecycleFF(nn.Module):
    def __init__(self, dmodel: int, dff: int, pruner: Pruner):
        super().__init__()
        self.lin1 = Linear(dmodel, dff)
        self.lin2 = Linear(dff, dmodel)
        self.dff = dff
        pruner.register(self)

    def _regular_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x

    def _new_neurons_forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply FF1
        lin_weights_1 = misc.einsum(
            "f, f m -> f m", self.mask, self.lin1.weight.data
        ) + misc.einsum("f, f m -> f m", 1 - self.mask, self.new_weights_1)
        lin_bias_1 = misc.einsum("f, f -> f", self.mask, self.lin1.bias.data)
        x = misc.einsum("... i, o i -> ... o", x, lin_weights_1) + lin_bias_1

        # Appply FF2
        lin_weights_2 = misc.einsum(
            "f, m f -> m f", self.mask, self.lin2.weight.data
        ) + misc.einsum("f, m f -> m f", 1 - self.mask, self.new_weights_2)
        x = misc.einsum("... i, o i -> ... o", x, lin_weights_2) + self.lin2.bias.data

        return x

    def forward(self, x: torch.Tensor, mode="regular") -> torch.Tensor:
        if mode == "regular":
            return self._regular_forward(x)
        elif mode == "new_neurons":
            return self._new_neurons_forward(x)

    def prepare_new_weights(self, prob: float):
        device = self.lin1.weight.device

        # create mask
        self.mask = torch.ones(self.dff).to(device)

        # prepare mask
        weights1 = misc.einsum("f m -> f", self.lin1.weight**2)
        weights2 = misc.einsum("m f -> f", self.lin2.weight**2)
        weights = weights1 * weights2
        n_els_weights = torch.numel(weights)
        assert n_els_weights == self.dff
        n_to_prune = round(prob * n_els_weights)
        topk = torch.topk(torch.abs(weights).view(-1), n_to_prune, largest=False)
        self.mask[topk.indices] = 0

        # prepare new weights for lin1
        self.new_weights_1 = kaiming_uniform_(
            torch.empty_like(self.lin1.weight), a=math.sqrt(5)
        )
        self.new_weights_1 *= 3**0.5

        # prepare new weights for lin2
        self.new_weights_2 = kaiming_uniform_(
            torch.empty_like(self.lin2.weight), a=math.sqrt(5)
        )
        self.new_weights_2 *= 3**0.5

    def apply_new_weights(self):
        self.lin1.weight.data = misc.einsum(
            "f, f m -> f m", self.mask, self.lin1.weight.data
        ) + misc.einsum("f, f m -> f m", 1 - self.mask, self.new_weights_1)
        self.lin1.bias.data = misc.einsum("f, f -> f", self.mask, self.lin1.bias.data)

        self.lin2.weight.data = misc.einsum(
            "f, m f -> m f", self.mask, self.lin2.weight.data
        ) + misc.einsum("f, m f -> m f", 1 - self.mask, self.new_weights_2)

    def unfreeze_new_weights(self):
        self.new_weights_1.requires_grad = True
        self.new_weights_2.requires_grad = True
