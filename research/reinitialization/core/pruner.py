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

    def log(self):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "log"):
                layer.log(f"FF no. {i}")


# -------------- COPY FROM HERE --------------
# class Pruner(BasePruner):
#     def prune(self, prob: float):
#         print("Pruning step")
#         for layer in self.layers:
#             layer.prune(prob)


# class MagnitudeStatPruner(BasePruner):
#     layers = []

#     def prune(self, prob: float, init: str = "kaiming"):
#         print("Pruning step")
#         for layer in self.layers:
#             layer.prune(prob, init)

#     def _log_tensor_stats(self, tensor: torch.Tensor, step: int, title: str):
#         # Log statistics of a flat tensor (useful in case histogram doesn't work in ClearML)
#         minimum = tensor.min().item()
#         maximum = tensor.max().item()
#         mean = tensor.mean().item()
#         std = tensor.std().item()
#         print(f"Logging tensor stats for {title}")
#         Logger.current_logger().report_scalar(
#             f"{title}_min", f"{title}_min", iteration=step, value=minimum
#         )
#         print(f"{title}_min: {minimum} step: {step}")
#         Logger.current_logger().report_scalar(
#             f"{title}_min", f"{title}_min", iteration=step, value=maximum
#         )
#         print(f"{title}_max: {maximum} step: {step}")
#         Logger.current_logger().report_scalar(
#             f"{title}_min", f"{title}_min", iteration=step, value=mean
#         )
#         print(f"{title}_mean: {mean} step: {step}")
#         Logger.current_logger().report_scalar(
#             f"{title}_min", f"{title}_min", iteration=step, value=std
#         )
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
#                 "mean_magn_of_recycled_layer",
#                 f"Layer {i}",
#                 iteration=step,
#                 value=layer.neuron_magnitudes[layer.recently_pruned].mean().item(),
#             )
#             # self._log_tensor_stats(tensor, step, f"magnitude_layer_{i}")

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

#     def zero_grad(self):
#         for layer in self.layers:
#             layer.zero_grad_first_ff()

#     def grad_correct(self):
#         for layer in self.layers:
#             layer.increase_magn_second_ff()
