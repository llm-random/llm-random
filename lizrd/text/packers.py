from abc import ABC, abstractmethod
import itertools
import random
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np
from torch.utils.data import IterableDataset
from lizrd.datasets.processor import MaskingReplacementConfig

from lizrd.text.datasets import AbstractDataset
from lizrd.datasets.processed_batch import GeneralExample
from lizrd.text.tokenization import AbstractTokenizer, BertTokenizer


def take_circular(iterable, start, stop):
    cycle = itertools.cycle(iterable)
    return itertools.islice(cycle, start, stop)


class AbstractPacker(ABC, IterableDataset):
    def __init__(
        self,
        sequence_length: int,
        dataset: AbstractDataset,
        tokenizer_maker: Callable[[], AbstractTokenizer],
        np_rng: Optional[np.random.Generator] = None,
        py_rng: Optional[random.Random] = None,
    ):
        super().__init__()
        self._tokenizer = None
        self.dataset = dataset
        self.tokenizer_maker = tokenizer_maker
        self.sequence_length = sequence_length
        if np_rng is None:
            np_rng = np.random.default_rng()
        self.np_rng = np_rng
        if py_rng is None:
            py_rng = random.Random()
        self.py_rng = py_rng

    def __iter__(self) -> Iterator[GeneralExample]:
        while True:
            yield self.get_sample()

    @abstractmethod
    def get_sample(self) -> GeneralExample:
        raise NotImplementedError()

    @property
    def tokenizer(self) -> AbstractTokenizer:
        if self._tokenizer is None:
            self._tokenizer = self.tokenizer_maker()
        return self._tokenizer


class BERTPacker(
    AbstractPacker,
):
    def __init__(
        self,
        sequence_length: int,
        dataset: AbstractDataset,
        tokenizer_maker: Callable[[], AbstractTokenizer],
        mask_replace_config: MaskingReplacementConfig = MaskingReplacementConfig(),
        np_rng: Optional[np.random.Generator] = None,
        py_rng: Optional[random.Random] = None,
    ):
        super().__init__(
            sequence_length,
            dataset,
            tokenizer_maker,
            np_rng=np_rng,
            py_rng=py_rng,
        )
        self.mask_replace_config = mask_replace_config
        # TODO: fix mask_id and sep_id
        self.mask_id = 0
        self.sep_id = 1

    def get_sample(self) -> GeneralExample:
        """
        Sample examples from the dataset until we reach the desired sequence length.
        """
        target_ids: List[int] = []
        input_ids: List[int] = []
        calculate_loss: List[int] = []
        document_lengths: List[int] = []

        while True:
            document = self.dataset.get_document()
            tokens = self.tokenizer.text_to_ids(document)
            masked_input, is_mask = self._mask_text(tokens)

            target_ids.extend(tokens + [self.sep_id])
            input_ids.extend(masked_input + [self.sep_id])
            calculate_loss.extend(is_mask + [0])

            document_lengths.append(len(tokens) + 1)
            if (
                sum(document_lengths) - max(document_lengths)
            ) > self.sequence_length and sum(
                document_lengths
            ) > 10 * self.sequence_length:
                break

        sample_start = self.py_rng.randint(0, len(target_ids) - 1)
        sample_end = sample_start + self.sequence_length

        target_ids = list(take_circular(target_ids, sample_start, sample_end))
        input_ids = list(take_circular(input_ids, sample_start, sample_end))
        calculate_loss = list(take_circular(calculate_loss, sample_start, sample_end))

        return GeneralExample(input_ids, target_ids, calculate_loss)

    def _mask_text(self, tokens: List[int]) -> Tuple[List[int], List[int]]:
        is_mask = self.np_rng.binomial(
            1, self.mask_replace_config.mask_percentage, len(tokens)
        )
        how_to_mask = self.np_rng.multinomial(
            1,
            [
                self.mask_replace_config.replace_with_mask,
                self.mask_replace_config.replace_with_random,
                self.mask_replace_config.replace_with_original,
            ],
            size=len(tokens),
        ).nonzero()[1]
        replacements = (
            (how_to_mask == 0) * self.mask_id
            + (how_to_mask == 1) * self._get_valid_random_tokens(len(tokens))
            + (how_to_mask == 2) * tokens
        )

        input_tokens = np.where(is_mask.astype(bool), replacements, tokens)

        return input_tokens.tolist(), is_mask.tolist()

    def _get_valid_random_tokens(self, tokens_count):
        assert isinstance(self.tokenizer, BertTokenizer)
        special_tokens = 999
        return (
            self.np_rng.choice(self.tokenizer.vocab_size - special_tokens, tokens_count)
            + special_tokens
        )


class GPTPacker(
    AbstractPacker,
):
    def __init__(
        self,
        sequence_length: int,
        dataset: AbstractDataset,
        tokenizer_maker: Callable[[], AbstractTokenizer],
        np_rng: Optional[np.random.Generator] = None,
        py_rng: Optional[random.Random] = None,
    ):
        super().__init__(
            sequence_length,
            dataset,
            tokenizer_maker,
            np_rng=np_rng,
            py_rng=py_rng,
        )
        # TODO: fix mask_id and sep_id
        self.eot_id = 2

    def get_sample(self) -> GeneralExample:
        """
        Sample examples from the dataset until we reach the desired sequence length.
        """
        buffer: List[int] = []
        calculate_loss: List[int] = []
        document_lengths: List[int] = []

        while True:
            document = self.dataset.get_document()
            tokens = self.tokenizer.text_to_ids(document)
            buffer.extend(tokens + [self.eot_id])

            document_lengths.append(len(tokens) + 1)
            if (
                sum(document_lengths) - max(document_lengths)
            ) > self.sequence_length and sum(
                document_lengths
            ) > 10 * self.sequence_length:
                break

        sample_start = self.py_rng.randint(0, len(target_ids) - 1)
        sample_end = sample_start + self.sequence_length

        input_ids = list(take_circular(buffer, sample_start, sample_end))
        target_ids = list(take_circular(buffer, sample_start + 1, sample_end + 1))
        calculate_loss = [1] * len(target_ids)

        return GeneralExample(input_ids, target_ids, calculate_loss)
