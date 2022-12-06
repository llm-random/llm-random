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
    def histogram_neurons(self, step):
        for i, layer in enumerate(self.layers):
            weights1 = misc.einsum("f m -> f", layer.lin1.weight**2)
            weights2 = misc.einsum("m f -> f", layer.lin2.weight**2)
            weights = weights1 * weights2
            tensor = weights.flatten().cpu()
            values = tensor.tolist()
            fig = px.histogram(values)
            Logger.current_logger().report_plotly(
                title="Magnitude of all neurons",
                series=f"FF {i}",
                iteration=step,
                figure=fig,
            )

    def histogram_weights(self, step):
        for i, layer in enumerate(self.layers):
            tensor = layer.lin1.weight.flatten().cpu()
            values = tensor.tolist()
            fig = px.histogram(values).to_image(format="png")
            fig: bytes
            print(len(fig))
            Logger.current_logger().report_image(
                title="Magnitude of all weights",
                series=f"FF {i}",
                iteration=step,
                image=fig,
            )
            # Logger.current_logger().report_plotly(
            #     title="Magnitude of all neurons (l1)",
            #     series=f"FF {i}, l1",
            #     iteration=step,
            #     figure=fig,
            # )

            # tensor = layer.lin2.weight.flatten().cpu()
            # values = tensor.tolist()
            # fig = px.histogram(values)
            # Logger.current_logger().report_plotly(
            #     title="Magnitude of all neurons (l2)",
            #     series=f"FF {i}, l2",
            #     iteration=step,
            #     figure=fig,
            # )

    def histogram(self, step: int):
        self.histogram_neurons(step)
        # self.histogram_weights(step)

    def prune(self, prob: float):
        print("Pruning step")
        for layer in self.layers:
            layer.prune(prob)

    def decrement_immunity(self):
        for layer in self.layers:
            if hasattr(layer, "decrement_immunity"):
                layer.decrement_immunity()
