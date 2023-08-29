from typing import List

from attr import dataclass
import numpy as np
import torch


@dataclass
class LLMExample(object):
    input_ids: List[int]
    target_ids: List[int]
    should_calculate_loss: List[int]


class LLMBatch:
    def __init__(self, examples: List[LLMExample]):
        self.input_ids = self._make_tensor([example.input_ids for example in examples])
        self.target_ids = self._make_tensor(
            [example.target_ids for example in examples]
        )
        self.should_calculate_loss = self._make_tensor(
            [example.should_calculate_loss for example in examples]
        )

        assert self.input_ids.shape == self.target_ids.shape
        assert self.input_ids.shape == self.should_calculate_loss.shape

    def __iter__(self):
        all_attrs = vars(self).items()
        return iter(
            [(attr, value) for attr, value in all_attrs if hasattr(value, "shape")]
        )

    @property
    def device(self):
        assert (
            self.input_ids.device
            == self.target_ids.device
            == self.should_calculate_loss.device
        )
        return self.input_ids.device

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.target_ids = self.target_ids.to(device)
        self.should_calculate_loss = self.should_calculate_loss.to(device)
        return self

    def _make_tensor(self, list_of_token_lists: List[List[int]]) -> torch.Tensor:
        matrix = np.array(list_of_token_lists)
        return torch.from_numpy(matrix)
