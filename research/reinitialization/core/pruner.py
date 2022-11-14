from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

# to avoid cycle import while using hints
if TYPE_CHECKING:
    from research.reinitialization.core.linears import RandomPruneLayer

class BasePruner(ABC):
    layers = []

    def register(self, layer: 'RandomPruneLayer'):
        self.layers.append(layer)

    @abstractmethod    
    def step(self):
        ...

class Pruner(BasePruner):
    def __init__(self, n_steps_prune: int, prob: float):
        self.n_steps_prune = n_steps_prune
        self.prob = prob
        self.current_step = 0
        self.layers = []

    def step(self):
        if self.current_step % self.n_steps_prune == 0:
            print("Pruning step")
            for layer in self.layers:
                layer.prune(self.prob)
        self.current_step += 1

class VariableProbabilityPruner(BasePruner):
    def step(self, prob: float):
        for layer in self.layers:
            layer.prune(prob)
