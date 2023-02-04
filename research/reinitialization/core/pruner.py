from abc import ABC, abstractmethod

import torch


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

    def decrement_immunity(self):
        for layer in self.layers:
            if hasattr(layer, "decrement_immunity"):
                layer.decrement_immunity()

    def pre_retrain(self):
        print("Pre retrain")
        for layer in self.layers:
            if hasattr(layer, "pre_retrain"):
                layer.pre_retrain()

    def post_retrain(self):
        print("Post retrain")
        for layer in self.layers:
            if hasattr(layer, "post_retrain"):
                layer.post_retrain()

    def prepare_new(self, prob: float):
        print("Preparing new step")
        for layer in self.layers:
            if hasattr(layer, "prepare_new_weights"):
                layer.prepare_new_weights(prob)

    def apply_new_weights(self):
        print("Applying new weights")
        for layer in self.layers:
            if hasattr(layer, "apply_new_weights"):
                layer.apply_new_weights()

    def log_plots(self, step):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "log_plots"):
                layer.log_plots(f"FF no. {i}", step)

    def log_scalars(self, step):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "log_scalars"):
                layer.log_scalars(f"FF no. {i}", step)

    def set_saving_stats(self):
        for layer in self.layers:
            if hasattr(layer, "save_stats"):
                layer.save_stats = True

    def enable_neuron_diff(self, ff_layer_num: int, idx: torch.Tensor):
        self.layers[ff_layer_num].enable_neuron_diff(idx)

    def disable_neuron_diff(self):
        for layer in self.layers:
            layer.disable_neuron_diff()
