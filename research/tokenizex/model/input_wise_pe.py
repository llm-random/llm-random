from typing import Any, Literal
import torch
from lizrd.core.llm import PositionalEmbedding


class InputWisePE:
    def __init__(self) -> None:
        self.positions: torch.Tensor = None

    def set_positions(self, positions: torch.Tensor):
        self.positions = positions

    def remove_positions(self):
        self.positions = None


class ManagerPESetter:
    def __init__(self, model: Any, positions: torch.Tensor):
        self.positions = positions
        self._layers: list[InputWisePE] = []
        for _, layer in model.named_modules():
            if isinstance(layer, InputWisePE):
                self._layers.append(layer)
        if len(self._layers) == 0:
            raise Exception("No InputWisePE modules in provided model")

    def __enter__(self):
        # self.positions.unsqueeze_(1) #dev ? propably to not batch but batch example wise
        for layer in self._layers:
            layer.set_positions(self.positions)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # self.positions.squeeze_() #dev
        for layer in self._layers:
            layer.remove_positions()


class InputWisePositionalEmbedding(PositionalEmbedding, InputWisePE):
    def __init__(
        self,
        max_length,
        embedding_dim,
        init_type: Literal["kaiming_uniform", "truncated_normal"],
        init_scale: float,
    ):
        PositionalEmbedding.__init__(
            self, max_length, embedding_dim, init_type, init_scale
        )
        InputWisePE.__init__(self)
        # super(InputWisePositionalEmbedding, self).__init__(max_length, embedding_dim, init_type, init_scale)

    def forward(self, x):
        if self.positions is None:
            raise Exception("Positions were not managed (InputWisePE)")
        embeddings = self.layer(self.positions)
        return embeddings
