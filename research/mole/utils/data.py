from typing import List, Optional

import numpy as np
import torch
from attr import dataclass


@dataclass
class LLMMetaExample(object):
    input_ids: List[int]
    one_hot_exp_groups: Optional[torch.Tensor]
    target_ids: List[int]
    should_calculate_loss: List[
        int
    ]  # e.g. in BERT loss is not calculated over non-masked tokens


class LLMMetaBatch:
    def __init__(self, examples: List[LLMMetaExample]):
        self.input_ids = self._make_tensor([example.input_ids for example in examples])
        if examples[0].one_hot_exp_groups != None:
            inside = [example.one_hot_exp_groups.unsqueeze(0) for example in examples]
            self.one_hot_exp_groups = torch.cat(inside, dim=0)
        else:
            self.one_hot_exp_groups = None
            
        self.target_ids = self._make_tensor(
            [example.target_ids for example in examples]
        )
        self.should_calculate_loss = self._make_tensor(
            [example.should_calculate_loss for example in examples]
        )

        assert self.input_ids.shape == self.target_ids.shape
        assert self.input_ids.shape == self.should_calculate_loss.shape

    def pin_memory(self):
        """Pin memory for faster transfer to GPU as described in https://pytorch.org/docs/stable/data.html#memory-pinning"""
        self.input_ids = self.input_ids.pin_memory()
        self.target_ids = self.target_ids.pin_memory()
        self.should_calculate_loss = self.should_calculate_loss.pin_memory()
        self.one_hot_exp_groups = self.one_hot_exp_groups.pin_memory()
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
        )
        if self.one_hot_exp_groups != None:
            assert self.one_hot_exp_groups.device == self.input_ids.device
        return self.input_ids.device

    def to(self, device) -> "LLMMetaBatch":
        self.input_ids = self.input_ids.to(device)
        self.target_ids = self.target_ids.to(device)
        self.should_calculate_loss = self.should_calculate_loss.to(device)
        if self.one_hot_exp_groups != None:
            self.one_hot_exp_groups = self.one_hot_exp_groups.to(device)
        return self

    def _make_tensor(self, list_of_token_lists: List[List[int]]) -> torch.Tensor:
        matrix = np.array(list_of_token_lists)
        return torch.from_numpy(matrix)
