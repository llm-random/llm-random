from typing import Literal
from torch import nn
from lizrd.core.llm import PositionalEmbedding


class InputWisePositionalEmbedding(PositionalEmbedding):
    def __init__(
        self,
        max_length,
        embedding_dim,
        init_type: Literal["kaiming_uniform", "truncated_normal"],
        init_scale: float,
    ):
        super(InputWisePositionalEmbedding, self).__init__(max_length, embedding_dim, init_type, init_scale)
        
    def forward(self, x, positions):
        embeddings = self.layer(positions)
        return embeddings