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

    def prepare_neuron_diff_idx(self, n_samples, sample_size):
        for layer in self.layers:
            if hasattr(layer, "prepare_neuron_diff_idx"):
                layer.prepare_neuron_diff_idx(n_samples, sample_size)

    def enable_neuron_diff(self, ff_layer_num: int, sample_number: int):
        self.layers[ff_layer_num].enable_neuron_diff(sample_number)

    def disable_neuron_diff(self):
        for layer in self.layers:
            if hasattr(layer, "disable_neuron_diff"):
                layer.disable_neuron_diff()

    def get_activation_ratios_of_masked_neurons(self, layer_num: int):
        if not hasattr(self.layers[layer_num], "activation_ratios_of_masked_neurons"):
            raise ValueError("Property activate ratio not present")
        return self.layers[layer_num].activation_ratios_of_masked_neurons()

    def get_magnitudes_of_masked_neurons(self, layer_num: int):
        if not hasattr(self.layers[layer_num], "neuron_magnitudes_of_masked_neurons"):
            raise ValueError("Property magnitudes not present")
        return self.layers[layer_num].neuron_magnitudes_of_masked_neurons()
