
from statistics import mean
from typing import Callable, List, Optional

from attr import dataclass
import numpy as np
import torch
from lizrd.text.data import LLMExample
from lizrd.text.datasets import AbstractDataset
from lizrd.text.packers import AbstractPacker, take_circular
from lizrd.text.tokenizers import AbstractTokenizer

@dataclass
class CompLLMExample(LLMExample):
    byttok_scale: float


class CompLLMBatch:
    def __init__(self, examples: List[LLMExample]):
        self.input_ids = self._make_tensor([example.input_ids for example in examples])
        self.target_ids = self._make_tensor(
            [example.target_ids for example in examples]
        )
        self.should_calculate_loss = self._make_tensor(
            [example.should_calculate_loss for example in examples]
        )
        self.byttok_scale = mean([example.byttok_scale for example in examples])

        assert self.input_ids.shape == self.target_ids.shape
        assert self.input_ids.shape == self.should_calculate_loss.shape

    def pin_memory(self):
        """Pin memory for faster transfer to GPU as described in https://pytorch.org/docs/stable/data.html#memory-pinning"""
        self.input_ids = self.input_ids.pin_memory()
        self.target_ids = self.target_ids.pin_memory()
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
        )
        return self.input_ids.device

    def to(self, device) -> "CompLLMBatch":
        self.input_ids = self.input_ids.to(device)
        self.target_ids = self.target_ids.to(device)
        self.should_calculate_loss = self.should_calculate_loss.to(device)
        return self

    def _make_tensor(self, list_of_token_lists: List[List[int]]) -> torch.Tensor:
        matrix = np.array(list_of_token_lists)
        return torch.from_numpy(matrix)


class CompGPTPacker(
    AbstractPacker,
):
    def __init__(
        self,
        sequence_length: int,
        dataset_maker: AbstractDataset,
        tokenizer_maker: Callable[[], AbstractTokenizer],
        seed: Optional[int] = None,
    ):
        super().__init__(
            sequence_length,
            dataset_maker,
            tokenizer_maker,
            seed=seed,
        )

    def get_sample(self) -> CompLLMExample:
        """
        Sample examples from the dataset until we reach the desired sequence length.
        """
        eot_id = self.tokenizer.eot_id
        assert eot_id is not None

        buffer: List[int] = []
        calculate_loss: List[int] = []
        document_lengths: List[int] = []

        while True:
            document = self.dataset.get_document() #dev can load 53k doc
            # tokens = self.tokenizer.text_to_ids(document) 
            tokens = self.tokenizer.tokenizer.encode(document) 
            buffer.extend(tokens + [eot_id])

            document_lengths.append(len(tokens) + 1)
            if (sum(document_lengths) - max(document_lengths)) > self.sequence_length:
                break

        sample_start = self.py_rng.randint(0, len(buffer) - 1)
        sample_end = sample_start + self.sequence_length

        input_ids = list(take_circular(buffer, sample_start, sample_end))
        target_ids = list(take_circular(buffer, sample_start + 1, sample_end + 1))
        calculate_loss = [1] * len(target_ids)

        target_bytes_c = len("".join(self.tokenizer.tokenizer.tokenize(self.tokenizer.tokenizer.decode(target_ids))))
        byttok_scale = target_bytes_c/self.sequence_length
        
        return CompLLMExample(input_ids, target_ids, calculate_loss, byttok_scale=byttok_scale)