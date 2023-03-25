from torch import nn
import torch
import torch.nn.functional as F
from lizrd.core import misc
from research.reinitialization.core.pruner import Pruner
import numpy as np
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
        new_weight_init: str = "random",
    ):
        super().__init__()

        self.lin1 = misc.Linear(dmodel, dff, bias=False)
        self.lin2 = misc.Linear(dff, dmodel, bias=False)
        self.current_activations = self.activate_ratio = np.zeros(dff)
        pruner.register(self)
        self.dff = dff

        self.prune_ratio = prune_ratio
        self.n_steps_interpolate = n_steps_interpolate

        self.register_buffer("mask", torch.ones(dff, requires_grad=False))
        self.register_buffer(
            "frozen_weights_1",
            self.lin1.weight.data.detach().clone().requires_grad_(False),
        )
        self.register_buffer(
            "frozen_weights_2",
            self.lin2.weight.data.detach().clone().requires_grad_(False),
        )

        self.alpha = 1.0

        self.noise_enabled = False

        assert new_weight_init in ["random", "mimic"]
        self.new_weight_init = new_weight_init

    def get_device(self):
        return self.lin1.weight.device

    def _save_activation_stats(self, x: torch.Tensor):
        self.latest_activations = x.detach().clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # make sure requires_grad is set correctly
        assert not self.frozen_weights_1.requires_grad
        assert not self.frozen_weights_2.requires_grad
        assert not self.mask.requires_grad
        assert self.mask.shape == (self.dff,)

        # noise interpolation is not applied before n_delay_steps controlled by Trainer
        if self.noise_enabled:
            # update value of alpha
            self.alpha = self.alpha + 1 / self.n_steps_interpolate

            if self.alpha > 1.0:
                self.alpha = 0.0
                self.prepare_mask()

        print(f"Noise enabled: {self.noise_enabled}")

        # apply lin1
        noisy_weights = (
            1 - self.alpha
        ) * self.frozen_weights_1 + self.alpha * self.lin1.weight.data
        # weight = misc.einsum(
        #     "f, f m -> f m", self.mask, self.lin1.weight.data
        # ) + misc.einsum("f, f m -> f m", 1 - self.mask, noisy_weights)
        weight = (
            self.mask.view(self.dff, 1) * self.lin1.weight.data
            + (1 - self.mask.view(self.dff, 1)) * noisy_weights
        )

        assert self.noise_enabled or (
            (self.alpha == 1.0)
            and ((weight == self.lin1.weight).count_nonzero() == weight.numel())
        )
        # x = misc.einsum("... m, f m -> ... f", x, weight)
        x = x @ weight.transpose(0, 1)

        self._save_activation_stats(x)
        x = F.relu(x)

        # apply lin2
        noisy_weights = (
            1 - self.alpha
        ) * self.frozen_weights_2 + self.alpha * self.lin2.weight.data
        # weight = misc.einsum(
        #     "f, m f -> m f", self.mask, self.lin2.weight.data
        # ) + misc.einsum("f, m f -> m f", 1 - self.mask, noisy_weights)
        weight = (
            self.mask.view(1, self.dff) * self.lin2.weight.data
            + (1 - self.mask.view(1, self.dff)) * noisy_weights
        )

        assert (
            self.noise_enabled
            or (self.alpha == 1.0)
            and ((weight == self.lin2.weight).count_nonzero() == weight.numel())
        )
        # x = misc.einsum("... f, m f -> ... m", x, weight)
        x = x @ weight.transpose(0, 1)

        return x

    def enable_noise_interpolation(self):
        self.noise_enabled = True

    def prepare_mask(self):
        self.frozen_weights_1.copy_(
            self.lin1.weight.detach().clone().requires_grad_(False)
        )
        self.frozen_weights_2.copy_(
            self.lin2.weight.detach().clone().requires_grad_(False)
        )

        # prepare mask
        self.mask.fill_(1)
        weights = self.neuron_magnitudes

        n_els_weights = torch.numel(weights)
        assert n_els_weights == self.dff

        n_to_prune = round(self.prune_ratio * n_els_weights)
        topk = torch.topk(torch.abs(weights).view(-1), n_to_prune, largest=False)
        self.mask[topk.indices] = 0

        # prepare target weights
        if self.new_weight_init == "random":
            target_weights_1 = self.get_random_weight(self.lin1)
            target_weights_2 = self.get_random_weight(self.lin2)
        elif self.new_weight_init == "mimic":
            perm = torch.randperm(
                self.dff
            )  # make sure permutation is the same in both layers
            target_weights_1 = self.get_mimicked_weight(self.lin1, perm)
            target_weights_2 = self.get_mimicked_weight(self.lin2, perm)

        self.lin1.weight.data = misc.einsum(
            "f, f m -> f m", self.mask, self.lin1.weight.data
        ) + misc.einsum("f, f m -> f m", 1 - self.mask, target_weights_1)
        self.lin2.weight.data = misc.einsum(
            "f, m f -> m f", self.mask, self.lin2.weight.data
        ) + misc.einsum("f, m f -> m f", 1 - self.mask, target_weights_2)

    def get_random_weight(self, layer):
        mean = layer.weight.mean().detach().cpu().item()
        std = layer.weight.std().detach().cpu().item()

        new_weights = torch.normal(
            mean, std, size=layer.weight.shape, device=self.get_device()
        )

        return new_weights

    def get_mimicked_weight(self, layer, permute_order):
        std = layer.weight.std().detach().cpu().item()

        if layer.weight.shape[0] == self.dff:
            new_weights = (
                layer.weight.data.detach()
                .clone()
                .requires_grad_(False)[permute_order, :]
            )
        else:
            new_weights = (
                layer.weight.data.detach()
                .clone()
                .requires_grad_(False)[:, permute_order]
            )

        # add small noise (std = 0.05 * std of weights in layer)
        new_weights += torch.normal(
            0, 0.05 * std, size=new_weights.shape, device=self.get_device()
        )

        return new_weights

    @property
    def neuron_magnitudes(self) -> torch.Tensor:
        return misc.get_neuron_magnitudes(self.lin1.weight, self.lin2.weight)

    def log_magnitude(self, layer_name, step: int):
        tensor = self.neuron_magnitudes.flatten().cpu()
        values = tensor.tolist()
        fig = px.histogram(values)
        log_plot(
            title="Magnitude of all neurons (no square)",
            series=layer_name,
            iteration=step,
            figure=fig,
        )

    def log_activations(self, layer_name: str, step: int):
        values = self.latest_activations.sum(dim=[0, 1]).cpu().numpy().tolist()
        fig = px.histogram(values)
        log_plot(
            title="Average activations of all neurons",
            series=layer_name,
            iteration=step,
            figure=fig,
        )

    def log_activation_ratios(self, layer_name: str, step: int):
        values = (
            (self.latest_activations > 0)
            .float()
            .mean(dim=[0, 1])
            .cpu()
            .numpy()
            .tolist()
        )
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
        values = x_flattened[random_indices].tolist()
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
