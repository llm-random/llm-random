from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

# to avoid cycle import while using hints
if TYPE_CHECKING:
    from research.reinitialization.core.linears import RandomPruneLayer


class BasePruner(ABC):
    layers = []

    def register(self, layer: "RandomPruneLayer"):
        self.layers.append(layer)

    @abstractmethod
    def prune(self):
        ...


class Pruner:
    def prune(self, prob: float):
        print("Pruning step")
        for layer in self.layers:
            layer.prune(prob)
