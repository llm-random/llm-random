from torch import nn
import torch
import torch.nn.functional as F
from lizrd.core import misc
from lizrd.core.misc import Linear
from lizrd.support.loss import LossDict
from research.reinitialization.core.pruner import Pruner
import numpy as np
from lizrd.support.logging import (
    get_current_logger,
    log_plot as log_plot,
)
import plotly_express as px


class BaseLossFF(nn.Module):
    def __init__(
        self,
        dmodel: int,
        dff: int,
        reg_pow: float,
        only_smaller_neurons: bool,
        midpoint_type: str,
        transform_type: str,
        pruner: Pruner,
        aggregation_type: str = "regular",
    ):
        super().__init__()
        self.reg_pow = reg_pow
        self.only_smaller_neurons = only_smaller_neurons
        assert midpoint_type in ["mean", "median"]
        self.midpoint_type = midpoint_type
        assert transform_type in ["linear", "log"]
        self.transform_type = transform_type
        assert aggregation_type in ["regular", "concat", "mixed", "dmodel"]
        self.aggregation_type = aggregation_type

        self.lin1 = Linear(dmodel, dff)
        self.lin2 = Linear(dff, dmodel)
        self.current_activations = self.activate_ratio = np.zeros(dff)
        pruner.register(self)

    def _save_activation_stats(self, x: torch.Tensor):
        self.latest_activations = x.detach().clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        self._save_activation_stats(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x

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

    def get_midpoint_loss(self) -> torch.Tensor:
        if self.aggregation_type == "concat":
            magnitudes = misc.get_split_neuron_magnitudes(
                self.lin1.weight, self.lin2.weight
            )
        elif self.aggregation_type == "mixed":
            magnitudes = misc.get_mixed_neuron_magnitudes(
                self.lin1.weight, self.lin2.weight
            )
        elif self.aggregation_type == "dmodel":
            magnitudes = misc.get_dmodel_magnitudes(self.lin1.weight, self.lin2.weight)
        else:
            magnitudes = self.neuron_magnitudes

        if self.midpoint_type == "median":
            midpoint = magnitudes.median().detach()
        elif self.midpoint_type == "mean":
            midpoint = magnitudes.mean().detach()
        else:
            raise ValueError(f"Unknown average type: {self.midpoint_type}")

        which_neurons = (
            (magnitudes < midpoint)
            if self.only_smaller_neurons
            else torch.ones_like(magnitudes)
        )
        penalty = torch.abs(magnitudes - midpoint) ** self.reg_pow

        loss = (which_neurons * penalty).sum()
        return loss

    def get_decay_loss(self) -> torch.Tensor:
        return (self.lin1.weight**2).sum() + (self.lin2.weight**2).sum()

    def get_auxiliary_loss(self) -> LossDict:
        return {"midpoint": self.get_midpoint_loss(), "decay": self.get_decay_loss()}
