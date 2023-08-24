from abc import ABC
from typing import Callable, Iterator, List, Optional
from transformers import BertTokenizer, GPT2TokenizerFast
from lizrd.datasets.datasets import AbstractDataset

from lizrd.datasets.processed_batch import GeneralExample
from lizrd.datasets.tokenizers import AbstractTokenizer
from lizrd.datasets.utils import get_random_chunk
from torch.utils.data import IterableDataset


class AbstractPacker(ABC, IterableDataset):
    def __init__(
        self,
        sequence_length: int,
        dataset: AbstractDataset,
        tokenizer_maker: Callable[[], AbstractTokenizer],
    ):
        super().__init__()

    def get_sample(self) -> GeneralExample:
        raise NotImplementedError()

    def __iter__(self) -> Iterator[GeneralExample]:
        while True:
            yield self.get_sample()


class BERTPacker(
    AbstractPacker,
):
    def __init__(self, sequence_length, dataset, tokenizer_maker):
        super().__init__(sequence_length, dataset, tokenizer_maker)
        self.dataset = dataset
        self.tokenizer_maker = tokenizer_maker
        self.sequence_length = sequence_length

    def get_sample(self) -> GeneralExample:
        if self.tokenizer is None:
            self.tokenizer = self.tokenizer_maker()
        raise NotImplementedError()


class GPTPacker(AbstractPacker):
    def __init__(
        self,
        sequence_length: int,
        dataset: AbstractDataset,
        tokenizer_maker: Callable[[], AbstractTokenizer],
    ):
        super().__init__(sequence_length, dataset, tokenizer_maker)
        self.dataset = dataset
        self.tokenizer_maker = tokenizer_maker
        self.tokenizer = None
        self.sequence_length = sequence_length

    def get_sample(self) -> GeneralExample:
        """
        Sample examples from the dataset until we reach the desired sequence length.
        """
        if self.tokenizer is None:
            self.tokenizer = self.tokenizer_maker()
        input_ids = []
        separator_mask = []
        while len(input_ids) < self.sequence_length:
            print(len(input_ids))
            example = self.dataset.get_example()
            example_ids = self.tokenizer.text_to_ids(example)
            if len(input_ids) + len(example_ids) > self.sequence_length:
                example_ids = get_random_chunk(
                    example_ids, self.sequence_length - len(input_ids)
                )
            input_ids.extend(example_ids)
            separator_mask.extend([1] * len(example_ids))
            if len(input_ids) < self.sequence_length:
                input_ids.append(self.tokenizer.sequence_separator_id)
                separator_mask.append(0)
        return GeneralExample(
            input_ids,
            input_ids[1:] + [self.tokenizer.sequence_separator_id],
            separator_mask,
        )
