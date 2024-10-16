from abc import ABC, abstractmethod
import itertools
import random
from typing import Callable, Iterator, List, Optional, Tuple
from attr import define

import numpy as np
from torch.utils.data import IterableDataset

from lizrd.text.datasets import AbstractDataset
from lizrd.text.data import LLMExample
from lizrd.text.tokenizers import AbstractTokenizer, BertTokenizer


def take_circular(iterable, start, stop):
    cycle = itertools.cycle(iterable)
    return itertools.islice(cycle, start, stop)


class AbstractPacker(ABC, IterableDataset):
    def __init__(
        self,
        sequence_length: int,
        dataset_maker: Callable[[], AbstractDataset],
        tokenizer_maker: Callable[[], AbstractTokenizer],
        seed: Optional[int] = None,
    ):
        super().__init__()
        self._tokenizer = None
        self._dataset = None
        self.dataset_maker = dataset_maker
        self.tokenizer_maker = tokenizer_maker
        self.sequence_length = sequence_length
        self.np_rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)
        self.seed = seed

    def set_rng(self, seed: Optional[int] = None):
        np_rng = np.random.default_rng(seed)
        py_rng = random.Random(seed)

        self.np_rng = np_rng
        self.py_rng = py_rng

        self.dataset.set_rng(seed)

    def __iter__(self) -> Iterator[LLMExample]:
        while True:
            yield self.get_sample()

    @abstractmethod
    def get_sample(self) -> LLMExample:
        raise NotImplementedError()

    @property
    def dataset(self) -> AbstractDataset:
        if self._dataset is None:
            self._dataset = self.dataset_maker()
            self._dataset.set_rng(self.seed)
        return self._dataset

    @property
    def tokenizer(self) -> AbstractTokenizer:
        if self._tokenizer is None:
            self._tokenizer = self.tokenizer_maker()
        return self._tokenizer


@define
class MaskingReplacementConfig:
    mask_percentage: float = 0.15
    replace_with_mask: float = 0.8
    replace_with_random: float = 0.1
    replace_with_original: float = 0.1

    def __attrs_post_init__(self):
        assert (
            self.replace_with_mask
            + self.replace_with_random
            + self.replace_with_original
        ) == 1.0


class BERTPacker(
    AbstractPacker,
):
    def __init__(
        self,
        sequence_length: int,
        dataset: AbstractDataset,
        tokenizer_maker: Callable[[], AbstractTokenizer],
        mask_replace_config: MaskingReplacementConfig = MaskingReplacementConfig(),
        seed: Optional[int] = None,
    ):
        super().__init__(
            sequence_length,
            dataset,
            tokenizer_maker,
            seed=seed,
        )
        self.mask_replace_config = mask_replace_config

    def get_sample(self) -> LLMExample:
        """
        Sample examples from the dataset until we reach the desired sequence length.
        """
        target_ids: List[int] = []
        input_ids: List[int] = []
        calculate_loss: List[int] = []
        document_lengths: List[int] = []

        sep_id = self.tokenizer.sequence_separator_id
        assert sep_id is not None

        while True:
            document = self.dataset.get_document()
            tokens = self.tokenizer.text_to_ids(document)
            masked_input, is_mask = self._mask_text(tokens)

            target_ids.extend(tokens + [sep_id])
            input_ids.extend(masked_input + [sep_id])
            calculate_loss.extend(is_mask + [0])

            document_lengths.append(len(tokens) + 1)
            if (sum(document_lengths) - max(document_lengths)) > self.sequence_length:
                break

        sample_start = self.py_rng.randint(0, len(target_ids) - 1)
        sample_end = sample_start + self.sequence_length

        target_ids = list(take_circular(target_ids, sample_start, sample_end))
        input_ids = list(take_circular(input_ids, sample_start, sample_end))
        calculate_loss = list(take_circular(calculate_loss, sample_start, sample_end))

        return LLMExample(input_ids, target_ids, calculate_loss)

    def _mask_text(self, tokens: List[int]) -> Tuple[List[int], List[int]]:
        mask_id = self.tokenizer.mask_id
        assert mask_id is not None

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
            (how_to_mask == 0) * mask_id
            + (how_to_mask == 1) * self._get_valid_random_tokens(len(tokens))
            + (how_to_mask == 2) * tokens
        )

        input_tokens = np.where(is_mask.astype(bool), replacements, tokens)

        return input_tokens.tolist(), is_mask.tolist()

    def _get_valid_random_tokens(self, tokens_count):
        assert isinstance(self.tokenizer, BertTokenizer)
        NUMBER_OF_SPECIAL_TOKENS = 999
        return (
            self.np_rng.choice(
                self.tokenizer.VOCAB_SIZE - NUMBER_OF_SPECIAL_TOKENS, tokens_count
            )
            + NUMBER_OF_SPECIAL_TOKENS
        )


class LegacyGPTPacker(
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

    def get_sample(self) -> LLMExample:
        """
        Sample examples from the dataset until we reach the desired sequence length.
        """
        eot_id = self.tokenizer.eot_id
        assert eot_id is not None

        buffer: List[int] = []
        calculate_loss: List[int] = []
        document_lengths: List[int] = []

        while True:
            document = self.dataset.get_document()
            tokens = self.tokenizer.text_to_ids(document)
            buffer.extend(tokens + [eot_id])

            document_lengths.append(len(tokens) + 1)
            if (sum(document_lengths) - max(document_lengths)) > self.sequence_length:
                break

        sample_start = self.py_rng.randint(0, len(buffer) - 1)
        sample_end = sample_start + self.sequence_length

        input_ids = list(take_circular(buffer, sample_start, sample_end))
        target_ids = list(take_circular(buffer, sample_start + 1, sample_end + 1))
        calculate_loss = [1] * len(target_ids)

        return LLMExample(input_ids, target_ids, calculate_loss)


class GPTPacker(
    AbstractPacker,
):
    def __init__(
        self,
        sequence_length: int,
        dataset_maker: AbstractDataset,
        tokenizer_maker: Callable[[], AbstractTokenizer],
        seed: Optional[int] = None,
        document_buffer_size: int = 10,
        samples_buffer_size: int = 100,
    ):
        super().__init__(
            sequence_length,
            dataset_maker,
            tokenizer_maker,
            seed=seed,
        )
        self.inputs = None
        self.targets = None
        self.documents_buffer_size = 10
        self.samples_buffer_size = 10000
        self.documents_buffer = []
        self.samples_buffer = []

    def _get_buffered(self, buffer, buffer_size, get_buffer_element):
        modified = False
        while len(buffer) < buffer_size:
            buffer.append(get_buffer_element())
            modified = True
        if modified:
            self.py_rng.shuffle(buffer)

        return buffer.pop()

    def _get_sample(self) -> LLMExample:
        """
        Sample examples from the dataset until we reach the desired sequence length.
        """
        eot_id = self.tokenizer.eot_id
        assert eot_id is not None

        if self.inputs is None:
            self.inputs = [eot_id]
            self.targets = []

        while len(self.targets) < self.sequence_length:
            document = self._get_buffered(
                self.documents_buffer,
                self.documents_buffer_size,
                self.dataset.get_document,
            )
            tokens = self.tokenizer.text_to_ids(document)
            self.inputs.extend(tokens)
            self.targets.extend(tokens)

        res_inputs = self.inputs[: self.sequence_length]
        res_targets = self.targets[: self.sequence_length]
        self.inputs = self.inputs[self.sequence_length :]
        self.targets = self.targets[self.sequence_length :]

        return LLMExample(
            input_ids=res_inputs,
            target_ids=res_targets,
            should_calculate_loss=[1] * len(res_inputs),
        )

    def get_sample(self) -> LLMExample:
        return self._get_buffered(
            self.samples_buffer, self.samples_buffer_size, self._get_sample
        )
