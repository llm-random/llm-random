from typing import TYPE_CHECKING
from abc import ABC, abstractmethod


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

    def unfreeze_new(self):
        print("Unfreezing new")
        for layer in self.layers:
            layer.unfreeze_new_weights()

    def prepare_new(self):
        print("Preparing new step")
        for layer in self.layers:
            layer.prepare_new_weights()

    def apply_new_weights(self):
        print("Applying new weights")
        for layer in self.layers:
            layer.apply_new_weights()
