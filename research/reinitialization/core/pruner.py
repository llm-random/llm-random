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
    scheduler = None

    def after_backprop(self, step):
        print("After backprop step")
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "after_backprop"):
                layer.after_backprop(f"Layer {i+1}", step)

    def log(self, step):
        print("Logging step")
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "log"):
                layer.log(f"Layer {i+1}", step)

    def prune(self, prob: float):
        print("Pruning step")
        for layer in self.layers:
            layer.prune(prob)

    def decrement_immunity(self):
        for layer in self.layers:
            if hasattr(layer, "decrement_immunity"):
                layer.decrement_immunity()
