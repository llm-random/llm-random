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
    scheduler = None

    def after_backprop(self, step):
        print("After backprop step")
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "after_backprop"):
                layer.after_backprop(f"Layer {i+1}", step)

    def log_light(self, step):
        print("Light logging step")
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "log_light"):
                layer.log_light(f"Layer {i+1}", step)

    def log_heavy(self, step):
        print("Heavy logging step")
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "log_heavy"):
                layer.log_heavy(f"Layer {i+1}", step)

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

    def set_saving_stats(self):
        for layer in self.layers:
            if hasattr(layer, "save_stats"):
                layer.save_stats = True

    def get_auxiliary_loss(self) -> torch.Tensor:
        aux_loss = torch.tensor(0, dtype=float)
        for layer in self.layers:
            if hasattr(layer, "get_auxiliary_loss"):
                layer_aux_loss = layer.get_auxiliary_loss()
                if layer_aux_loss.device != aux_loss.device:
                    aux_loss = aux_loss.to(layer_aux_loss.device)
                aux_loss += layer_aux_loss
        return aux_loss
