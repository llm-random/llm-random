from typing import List

import numpy as np
import torch
from attr import dataclass


@dataclass
class SubtokenLLMExample(object):
    input_ids: List[int]
    input_bytes: List[List[int]]
    target_ids: List[int]
    should_calculate_loss: List[
        int
    ]  # e.g. in BERT loss is not calculated over non-masked tokens


class SubtokenLLMBatch:
    def __init__(self, examples: List[SubtokenLLMExample]):
        self.input_ids = self._make_tensor([example.input_ids for example in examples])
        self.target_ids = self._make_tensor(
            [example.target_ids for example in examples]
        )
        input_bytes = []
        for example in examples:
            # example_bytes = example.input_bytes + [-1] * (
            #     max_n_bytes - len(example.input_bytes)
            # )
            example_bytes = example.input_bytes
            input_bytes.append(example_bytes)
        self.input_bytes = torch.from_numpy(np.array(input_bytes))
        self.should_calculate_loss = self._make_tensor(
            [example.should_calculate_loss for example in examples]
        )

        assert self.input_ids.shape == self.target_ids.shape
        assert self.input_ids.shape == self.should_calculate_loss.shape

    def pin_memory(self):
        """Pin memory for faster transfer to GPU as described in https://pytorch.org/docs/stable/data.html#memory-pinning"""
        self.input_ids = self.input_ids.pin_memory()
        self.target_ids = self.target_ids.pin_memory()
        self.input_bytes = self.input_bytes.pin_memory()
        self.should_calculate_loss = self.should_calculate_loss.pin_memory()
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
            == self.input_bytes.device
        )
        return self.input_ids.device

    def to(self, device) -> "LLMBatch":
        self.input_ids = self.input_ids.to(device)
        self.target_ids = self.target_ids.to(device)
        self.input_bytes = self.input_bytes.to(device)
        self.should_calculate_loss = self.should_calculate_loss.to(device)
        return self

    def _make_tensor(self, list_of_token_lists: List[List[int]]) -> torch.Tensor:
        matrix = np.array(list_of_token_lists)
        return torch.from_numpy(matrix)
