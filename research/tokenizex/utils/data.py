from statistics import mean
from typing import List

import numpy as np
import torch
from attr import dataclass

from lizrd.text.data import LLMExample


@dataclass
class TokenizexExample(LLMExample):
    positions: np.ndarray  # [int]
    attention_mask: np.ndarray  # [bool]
    deftok_byte_scale: float


class TokenizexBatch:
    def __init__(self, examples: List[TokenizexExample]):
        self.input_ids = self._make_tensor([example.input_ids for example in examples])
        self.target_ids = self._make_tensor(
            [example.target_ids for example in examples]
        )
        self.should_calculate_loss = self._make_tensor(
            [example.should_calculate_loss for example in examples]
        )
        self.attention_mask = torch.from_numpy(
            np.stack([example.attention_mask for example in examples])
        )
        self.positions = self._make_tensor([example.positions for example in examples])

        self.deftok_byte_scale = mean(
            [example.deftok_byte_scale for example in examples]
        )

        assert self.input_ids.shape == self.target_ids.shape
        assert self.input_ids.shape == self.should_calculate_loss.shape

    def pin_memory(self):
        """Pin memory for faster transfer to GPU as described in https://pytorch.org/docs/stable/data.html#memory-pinning"""
        self.input_ids = self.input_ids.pin_memory()
        self.target_ids = self.target_ids.pin_memory()
        self.should_calculate_loss = self.should_calculate_loss.pin_memory()
        self.attention_mask = self.attention_mask.pin_memory()
        self.positions = self.positions.pin_memory()
        return self

    def __iter__(self):
        all_attrs = vars(self).items()
        return iter(
            [
                (attr, value)
                for attr, value in all_attrs
                if isinstance(value, torch.Tensor)
            ]
        )

    @property
    def device(self) -> torch.device:
        assert (
            self.input_ids.device
            == self.target_ids.device
            == self.should_calculate_loss.device
            == self.positions.device
            == self.attention_mask.device
        )
        return self.input_ids.device

    def to(self, device) -> "TokenizexBatch":
        self.input_ids = self.input_ids.to(device)
        self.target_ids = self.target_ids.to(device)
        self.should_calculate_loss = self.should_calculate_loss.to(device)
        self.positions = self.positions.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self

    def _make_tensor(self, list_of_lists: List[List[int]]) -> torch.Tensor:
        matrix = np.array(list_of_lists)
        return torch.from_numpy(matrix)
