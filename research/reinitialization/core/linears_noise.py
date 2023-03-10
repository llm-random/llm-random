from torch import nn
import torch
import torch.nn.functional as F
from lizrd.core import misc
from research.reinitialization.core.pruner import Pruner
import numpy as np
from lizrd.core.misc import get_default_device
from lizrd.support.logging import (
    get_current_logger,
    log_plot as log_plot,
)
import plotly_express as px


class NoiseFF(nn.Module):
    def __init__(
        self,
        dmodel: int,
        dff: int,
        pruner: Pruner,
        prune_ratio: float,
        n_steps_interpolate: int,
    ):
        super().__init__()

        self.lin1 = misc.Linear(dmodel, dff)
        self.lin2 = misc.Linear(dff, dmodel)
        self.current_activations = self.activate_ratio = np.zeros(dff)
        pruner.register(self)

        self.prune_ratio = prune_ratio
        self.n_steps_interpolate = n_steps_interpolate

        self.mask = torch.ones(dff, device=get_default_device(), requires_grad=False)
        self.frozen_weights_1 = (
            self.lin1.weight.data.detach().clone().requires_grad_(False)
        )
        self.frozen_weights_2 = (
            self.lin2.weight.data.detach().clone().requires_grad_(False)
        )
        self.target_weights_1 = (
            self.lin1.weight.data.detach().clone().requires_grad_(True)
        )
        self.target_weights_2 = (
            self.lin2.weight.data.detach().clone().requires_grad_(True)
        )
        self.alpha = 0.0

    def _save_activation_stats(self, x: torch.Tensor):
        self.latest_activations = x.detach().clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.alpha == 1:
            self.prepare_mask()

        # apply lin1
        new_weights = (
            self.alpha * self.frozen_weights_1
            + (1 - self.alpha) * self.target_weights_1
        )
        self.lin1.weight.data = misc.einsum(
            "f, f m -> f m", self.mask, self.lin1.weight.data
        ) + misc.einsum("f, f m -> f m", 1 - self.mask, new_weights)

        self._save_activation_stats(x)
        x = F.relu(x)

        # apply lin2
        new_weights = (
            self.alpha * self.frozen_weights_2
            + (1 - self.alpha) * self.target_weights_1
        )
        self.lin2.weight.data = misc.einsum(
            "f, f m -> f m", self.mask, self.lin2.weight.data
        ) + misc.einsum("f, f m -> f m", 1 - self.mask, new_weights)

        # update value of alpha
        self.alpha = self.alpha + 1 / self.n_steps_interpolate

        return x

    def prepare_mask(self):
        self.frozen_weights_1 = (
            self.lin1.weight.data.detach().clone().requires_grad_(False)
        )
        self.frozen_weights_2 = (
            self.lin2.weight.data.detach().clone().requires_grad_(False)
        )

        # prepare target weights
        self.target_weights_1 = self.get_new_weight(self.lin1)
        self.target_weights_2 = self.get_new_weight(self.lin2)

        # prepare mask
        self.mask = torch.ones(
            self.dff, device=get_default_device(), requires_grad=False
        )
        weights = self.neuron_magnitudes()

        n_els_weights = torch.numel(weights)
        assert n_els_weights == self.dff

        n_to_prune = round(self.prune_ratio * n_els_weights)
        topk = torch.topk(torch.abs(weights).view(-1), n_to_prune, largest=False)
        self.mask[topk.indices] = 0

    def get_new_weight(self, layer):
        std = layer.weight.std().detach().cpu().item()
        mean = layer.weight.mean().detach().cpu().item()

        new_weights = torch.normal(mean, std, size=layer.weight.shape)

        return new_weights.to(self.device)

    @property
    def neuron_magnitudes(self) -> torch.Tensor:
        return misc.get_neuron_magnitudes(self.lin1.weight, self.lin2.weight)

    def log_magnitude(self, layer_name, step: int):
        tensor = (self.neuron_magnitudes**2).flatten().cpu()
        values = tensor.tolist()
        fig = px.histogram(values)
        log_plot(
            title="Magnitude of all neurons",
            series=layer_name,
            iteration=step,
            figure=fig,
        )

    def log_activations(self, layer_name: str, step: int):
        values = self.latest_activations.sum(dim=[0, 1]).cpu().numpy()
        fig = px.histogram(values)
        log_plot(
            title="Average activations of all neurons",
            series=layer_name,
            iteration=step,
            figure=fig,
        )

    def log_activation_ratios(self, layer_name: str, step: int):
        values = (self.latest_activations > 0).float().mean(dim=[0, 1]).cpu().numpy()
        fig = px.histogram(values)
        log_plot(
            title="Average ratio of activation per neuron",
            series=layer_name,
            iteration=step,
            figure=fig,
        )

    def log_activations_sampled(self, layer_name: str, step: int):
        x_flattened = self.latest_activations.flatten().cpu().numpy()
        random_indices = np.random.choice(x_flattened.shape[0], 1024, replace=False)
        values = x_flattened[random_indices]
        fig = px.histogram(values)
        log_plot(
            title="Activations of sampled neurons",
            series=layer_name,
            iteration=step,
            figure=fig,
        )

    def log_heavy(self, layer_name: str, step: int):
        with torch.no_grad():
            get_current_logger().flush_if_necessary()
            self.log_activations(layer_name, step)
            get_current_logger().flush_if_necessary()
            self.log_activation_ratios(layer_name, step)
            get_current_logger().flush_if_necessary()
            self.log_activations_sampled(layer_name, step)
            get_current_logger().flush_if_necessary()
            self.log_magnitude(layer_name, step)
            get_current_logger().flush_if_necessary()
