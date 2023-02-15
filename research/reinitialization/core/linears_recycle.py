import math

import numpy as np
import plotly.express as px
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_

from lizrd.core import misc
from lizrd.core.misc import Linear
from lizrd.support.logging import (
    get_current_logger,
    log_plot as log_plot,
)
from research.reinitialization.core.pruner import Pruner


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


class StructMagnitudeRecycleImmunityFF(nn.Module):
    def __init__(
        self,
        dmodel: int,
        dff: int,
        pruner: Pruner,
        immunity_start_value: int,
        reinit_dist: str = "init",
    ):
        super().__init__()
        self.lin1 = Linear(dmodel, dff)
        self.lin2 = Linear(dff, dmodel)
        self.dff = dff
        self.immunity_start_value = immunity_start_value
        self.immunity = nn.parameter.Parameter(
            torch.full((dff,), immunity_start_value), requires_grad=False
        )
        assert reinit_dist in ["init", "zero", "follow_normal"]
        self.reinit_dist = reinit_dist
        pruner.register(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x

    def decrement_immunity(self):
        self.immunity = nn.parameter.Parameter(
            torch.max(
                torch.zeros_like(self.immunity, device=self.lin1.weight.device),
                self.immunity - 1,
            ),
            requires_grad=False,
        )

    def get_new_weight(self, layer):
        if self.reinit_dist == "zero":
            new_weights = torch.zeros_like(layer.weight)
        elif self.reinit_dist == "init":
            new_weights = kaiming_uniform_(
                torch.empty_like(layer.weight), a=math.sqrt(5)
            )
            new_weights *= 3**0.5
        elif self.reinit_dist == "follow_normal":
            std = layer.weight.std().detach().cpu().item()
            mean = layer.weight.mean().detach().cpu().item()
            new_weights = torch.normal(mean, std, size=layer.weight.shape)
        return new_weights

    def reinitialize_layer1(self, mask: torch.Tensor):
        layer = self.lin1

        new_weights = self.get_new_weight(layer)

        layer.weight.data = misc.einsum(
            "f, f m -> f m", mask, layer.weight.data
        ) + misc.einsum(
            "f, f m -> f m", 1 - mask, new_weights
        )  # type: ignore
        layer.bias.data = misc.einsum("f, f -> f", mask, layer.bias.data)  # type: ignore

    def reinitialize_layer2(self, mask: torch.Tensor):
        layer = self.lin2

        new_weights = self.get_new_weight(layer)

        self.lin2.weight.data = misc.einsum(
            "f, m f -> m f", mask, self.lin2.weight.data
        ) + misc.einsum("f, m f -> m f", 1 - mask, new_weights)

    def reinitialize(self, mask):
        self.reinitialize_layer1(self.lin1, mask)
        self.reinitialize_layer2(self.lin2, mask)

    def prune(self, prob: float):
        device = self.lin1.weight.device

        # create mask
        mask = torch.ones(self.dff).to(device)

        # prepare mask
        weights1 = misc.einsum("f m -> f", self.lin1.weight**2)
        weights2 = misc.einsum("m f -> f", self.lin2.weight**2)
        weights = weights1 * weights2
        weights[self.immunity > 0] = float("inf")
        n_els_weights = torch.numel(weights)
        assert n_els_weights == self.dff
        n_to_prune = round(prob * n_els_weights)
        topk = torch.topk(torch.abs(weights).view(-1), n_to_prune, largest=False)
        mask[topk.indices] = 0

        self.immunity[topk.indices] = self.immunity_start_value

        self.reinitialize(mask)


def prepare_for_logging(x):
    return x.view(-1).detach().cpu().numpy()


def prepare_subset_for_logging(xs, p=None, size=None):
    xs = [prepare_for_logging(x) for x in xs]
    if size is not None:
        random_indices = np.random.choice(len(xs[0]), size, replace=False)
    else:
        random_indices = np.random.choice(
            len(xs[0]), int(len(xs[0]) * p), replace=False
        )
    return [x[random_indices] for x in xs]


class RetrainRecycleFF(nn.Module):
    def __init__(
        self,
        dmodel: int,
        dff: int,
        pruner: Pruner,
        retrain_without_reinit: bool = False,
        random_indexes: bool = False,
    ):
        super().__init__()
        self.lin1 = Linear(dmodel, dff, bias=False)
        self.lin2 = Linear(dff, dmodel, bias=False)
        self.dff = dff
        self.new_weights_1 = nn.Parameter(torch.empty_like(self.lin1.weight))
        self.new_weights_2 = nn.Parameter(torch.empty_like(self.lin2.weight))
        pruner.register(self)
        self.mode = "regular"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.recycle_counter = torch.zeros(self.dff).to(device)
        self.recently_pruned = torch.full((dff,), False).to(device)
        self.current_activations = self.activate_ratio = np.zeros(dff)
        self.save_stats = False
        self.retrain_without_reinit = retrain_without_reinit
        self.random_indexes = random_indexes

    def _regular_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = F.relu(x)

        # save activation stats
        self._save_activation_stats(x)

        x = self.lin2(x)
        return x

    def _new_neurons_forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply FF1
        lin_weights_1 = misc.einsum(
            "f, f m -> f m", self.mask, self.lin1.weight.data
        ) + misc.einsum("f, f m -> f m", 1 - self.mask, self.new_weights_1)
        x = misc.einsum("... i, o i -> ... o", x, lin_weights_1)

        # relu
        x = F.relu(x)

        # save activation stats
        self._save_activation_stats(x)

        # Appply FF2
        assert self.lin2.weight.data.shape == self.new_weights_2.shape
        lin_weights_2 = misc.einsum(
            "f, m f -> m f", self.mask, self.lin2.weight.data
        ) + misc.einsum("f, m f -> m f", 1 - self.mask, self.new_weights_2)
        assert self.lin2.weight.data.shape == lin_weights_2.shape
        x = misc.einsum("... i, o i -> ... o", x, lin_weights_2)

        return x

    def _save_activation_stats(self, x: torch.Tensor):
        if self.save_stats:
            self.current_activations = x.sum(dim=[0, 1]).detach().cpu().numpy()
            self.activate_ratio = (x > 0).float().mean(dim=[0, 1]).cpu().numpy()
            x_flattened = x.flatten().detach().cpu().numpy()
            random_indices = np.random.choice(x_flattened.shape[0], 1024, replace=False)
            self.some_activations = x_flattened[random_indices]
            self.save_stats = False

    @property
    def neuron_magnitudes(self):
        if self.mode == "regular":
            weights1 = misc.einsum("f m -> f", self.lin1.weight**2)
            weights2 = misc.einsum("m f -> f", self.lin2.weight**2)
        elif self.mode == "new_neurons":
            lin_weights_1 = misc.einsum(
                "f, f m -> f m", self.mask, self.lin1.weight.data
            ) + misc.einsum("f, f m -> f m", 1 - self.mask, self.new_weights_1)
            weights1 = misc.einsum("f m -> f", lin_weights_1**2)
            lin_weights_2 = misc.einsum(
                "f, m f -> m f", self.mask, self.lin2.weight.data
            ) + misc.einsum("f, m f -> m f", 1 - self.mask, self.new_weights_2)
            weights2 = misc.einsum("m f -> f", lin_weights_2)

        weights = weights1 * weights2
        return weights.flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "regular":
            return self._regular_forward(x)
        elif self.mode == "new_neurons":
            return self._new_neurons_forward(x)

    def prepare_new_weights(self, prob: float):
        # prepare mask
        self.mask = torch.ones(self.dff, requires_grad=False).to(
            self.lin1.weight.device
        )
        weights1 = misc.einsum("f m -> f", self.lin1.weight**2)
        weights2 = misc.einsum("m f -> f", self.lin2.weight**2)
        weights = weights1 * weights2

        n_els_weights = torch.numel(weights)
        assert n_els_weights == self.dff

        n_to_prune = round(prob * n_els_weights)
        if self.random_indexes:
            self.mask[torch.randperm(self.dff)[: round(prob * self.dff)]] = 0
        else:
            topk = torch.topk(torch.abs(weights).view(-1), n_to_prune, largest=False)
            self.mask[topk.indices] = 0

        # prepare new weights for lin1
        with torch.no_grad():
            if self.retrain_without_reinit:
                self.new_weights_1 = nn.Parameter(self.lin1.weight.clone())
            else:
                self.new_weights_1.normal_(
                    mean=self.lin1.weight.mean(), std=self.lin1.weight.std()
                )

        # prepare new weights for lin2
        with torch.no_grad():
            if self.retrain_without_reinit:
                self.new_weights_2 = nn.Parameter(self.lin2.weight.clone())
            else:
                self.new_weights_2.normal_(
                    mean=self.lin2.weight.mean(), std=self.lin2.weight.std()
                )

        # save statistics
        self.recycle_counter += 1 - self.mask  # weights
        self.recently_pruned = (1 - self.mask).bool()

    def apply_new_weights(self):
        self.lin1.weight.data = misc.einsum(
            "f, f m -> f m", self.mask, self.lin1.weight.data
        ) + misc.einsum("f, f m -> f m", 1 - self.mask, self.new_weights_1)

        self.lin2.weight.data = misc.einsum(
            "f, m f -> m f", self.mask, self.lin2.weight.data
        ) + misc.einsum("f, m f -> m f", 1 - self.mask, self.new_weights_2)

    def pre_retrain(self):
        self.new_weights_1.requires_grad = True
        self.new_weights_2.requires_grad = True
        self.mode = "new_neurons"

    def post_retrain(self):
        self.mode = "regular"

    def log_recycle_magnitude(self, layer_name, step: int):
        tensor = self.recycle_counter.flatten().cpu()
        values = tensor.tolist()
        fig = px.histogram(values)
        log_plot(
            title="No. of times neurons have been recycled",
            series=layer_name,
            iteration=step,
            figure=fig,
        )

    def log_magnitude(self, layer_name, step: int):
        tensor = self.neuron_magnitudes.flatten().cpu()
        values = tensor.tolist()
        fig = px.histogram(values)
        log_plot(
            title="Magnitude of all neurons",
            series=layer_name,
            iteration=step,
            figure=fig,
        )

    def log_activations(self, layer_name: str, step: int):
        values = self.current_activations
        fig = px.histogram(values)
        get_current_logger().report_plotly(
            title="Average activations of all neurons",
            series=layer_name,
            iteration=step,
            figure=fig,
        )

    def log_activation_ratios(self, layer_name: str, step: int):
        values = self.activate_ratio
        fig = px.histogram(values)
        get_current_logger().report_plotly(
            title="Average ratio of activation per neuron",
            series=layer_name,
            iteration=step,
            figure=fig,
        )

    def log_activations_sampled(self, layer_name: str, step: int):
        values = self.some_activations
        fig = px.histogram(values)
        get_current_logger().report_plotly(
            title="Activations of sampled neurons",
            series=layer_name,
            iteration=step,
            figure=fig,
        )

    def log_recently_pruned_magnitude(self, layer_name, step: int):
        val = self.neuron_magnitudes[self.recently_pruned].mean().item()
        if not math.isnan(val) and not math.isinf(val):
            get_current_logger().report_scalar(
                title="mean_magn_of_recycled_layer",
                series=layer_name,
                iteration=step,
                value=val,
            )

    def log_heavy(self, layer_name: str, step: int):
        get_current_logger().flush_if_necessary()
        self.log_activations(layer_name, step)
        get_current_logger().flush_if_necessary()
        self.log_activation_ratios(layer_name, step)
        get_current_logger().flush_if_necessary()
        self.log_activations_sampled(layer_name, step)
        get_current_logger().flush_if_necessary()
        self.log_recycle_magnitude(layer_name, step)
        get_current_logger().flush_if_necessary()
        self.log_magnitude(layer_name, step)
        get_current_logger().flush_if_necessary()

    def log_light(self, layer_name: str, step: int):
        self.log_recently_pruned_magnitude(layer_name, step)
