from typing import TYPE_CHECKING, Optional
from abc import ABC, abstractmethod

import torch
from clearml import Logger
import plotly.express as px


class BasePruner(ABC):
    def __init__(self):
        self.layers = []

    def register(self, layer):
        self.layers.append(layer)

    @abstractmethod
    def prune(self, *args, **kwargs):
        ...


class Pruner(BasePruner):
    def prune(self, prob: float):
        print("Pruning step")
        for layer in self.layers:
            layer.prune(prob)


class RetrainPruner:
    def __init__(self):
        self.layers = []

    def register(self, layer):
        self.layers.append(layer)

    def pre_retrain(self):
        print("Pre retrain")
        for layer in self.layers:
            layer.pre_retrain()

    def post_retrain(self):
        print("Post retrain")
        for layer in self.layers:
            layer.post_retrain()

    def prepare_new(self, prob: float):
        print("Preparing new step")
        for layer in self.layers:
            layer.prepare_new_weights(prob)

    def apply_new_weights(self):
        print("Applying new weights")
        for layer in self.layers:
            layer.apply_new_weights()


# class Pruner(BasePruner):
#     def prune(self, prob: float):
#         print("Pruning step")
#         for layer in self.layers:
#             layer.prune(prob)

#     def _log_tensor_stats(self, tensor: torch.Tensor, step: int, title: str):
#         # Log statistics of a flat tensor (useful in case histogram doesn't work in ClearML)
#         minimum = tensor.min().item()
#         maximum = tensor.max().item()
#         mean = tensor.mean().item()
#         std = tensor.std().item()
#         print(f"Logging tensor stats for {title}")
#         self.writer.add_scalar(f"{title}_min", minimum, step)
#         print(f"{title}_min: {minimum} step: {step}")
#         self.writer.add_scalar(f"{title}_max", maximum, step)
#         print(f"{title}_max: {maximum} step: {step}")
#         self.writer.add_scalar(f"{title}_mean", mean, step)
#         print(f"{title}_mean: {mean} step: {step}")
#         self.writer.add_scalar(f"{title}_std", std, step)
#         print(f"{title}_std: {std} step: {step}")

#     def log_recycle_magnitude(self, step: int):
#         for i, layer in enumerate(self.layers):
#             tensor = layer.recycle_counter.flatten().cpu()
#             values = tensor.tolist()
#             fig = px.histogram(values)
#             Logger.current_logger().report_plotly(
#                 title="Number of recycled neurons",
#                 series=f"Linear {i}",
#                 iteration=step,
#                 figure=fig,
#             )
#             self._log_tensor_stats(tensor, step, f"n_recycled_neurons_layer_{i}")

#     def log_magnitude(self, step: int):
#         for i, layer in enumerate(self.layers):
#             tensor = layer.neuron_magnitudes.flatten().cpu()
#             values = tensor.tolist()
#             fig = px.histogram(values)
#             Logger.current_logger().report_plotly(
#                 title="Magnitude of all neurons",
#                 series=f"Linear {i}",
#                 iteration=step,
#                 figure=fig,
#             )
#             self._log_tensor_stats(tensor, step, f"magnitude_layer_{i}")

#     def log_recently_pruned_magnitude(self, step: int):
#         for i, layer in enumerate(self.layers):
#             Logger.current_logger().report_scalar(
#                 f"mean_magn_of_recycled_layer_{i}",
#                 f"Linear layer {i}",
#                 iteration=step,
#                 value=layer.neuron_magnitudes[layer.recently_pruned].mean().item(),
#             )

#     def log_hist_all_weights(self, step: int):
#         for i, ff_layer in enumerate(self.layers):
#             for j, lin_layer in enumerate([ff_layer.lin1, ff_layer.lin2]):
#                 tensor = lin_layer.weight.data.flatten().cpu()
#                 values = tensor.tolist()
#                 fig = px.histogram(values)
#                 Logger.current_logger().report_plotly(
#                     title="Values of all weights",
#                     series=f"Linear layer {2*i+j}",
#                     iteration=step,
#                     figure=fig,
#                 )
#                 self._log_tensor_stats(tensor, step, f"all_weights_lin_layer_{2*i+j}")
