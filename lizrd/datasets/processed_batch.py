from abc import ABC

import numpy as np
import torch


class ProcessedBatch(ABC):
    def __init__(self, processed_examples):
        pass

    def __iter__(self):
        all_attrs = vars(self).items()
        return iter(
            [(attr, value) for attr, value in all_attrs if hasattr(value, "shape")]
        )

    def to(self, device):
        self.device = device
        for attr, tensor in self:
            setattr(self, attr, tensor.to(device))
        return self

    def _make_tensor(self, list_of_token_lists):
        matrix = np.array(list_of_token_lists)
        return torch.from_numpy(matrix)


class ProcessedGPTBatch(ProcessedBatch):
    def __init__(self, processed_examples):
        super().__init__(processed_examples)
        self.tokens = torch.tensor(
            [example["tokens"] for example in processed_examples]
        )
        self.target_tokens = torch.tensor(
            [example["target_tokens"] for example in processed_examples]
        )
        self.non_padded_mask = torch.tensor(
            [example["non_padded_mask"] for example in processed_examples]
        )


class ProcessedBERTBatch(ProcessedBatch):
    def __init__(self, processed_examples):
        super().__init__(processed_examples)
        self.tokens = self._make_tensor(
            [example.tokens for example in processed_examples]
        )
        self.mask_mask = self._make_tensor(
            [example.mask_mask for example in processed_examples]
        )
        self.masked_tokens = self._make_tensor(
            [example.masked_tokens for example in processed_examples]
        )

        assert self.tokens.shape == self.masked_tokens.shape
        assert self.tokens.shape == self.mask_mask.shape
