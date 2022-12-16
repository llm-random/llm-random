from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from clearml import Logger
import plotly.express as px
from lizrd.core import misc


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
