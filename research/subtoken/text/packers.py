from abc import ABC, abstractmethod
import itertools
import random
from typing import Callable, Iterator, List, Optional, Tuple
from attr import define

import numpy as np
from torch.utils.data import IterableDataset

from lizrd.text.datasets import AbstractDataset
from lizrd.text.data import LLMExample
from lizrd.text.tokenizers import AbstractTokenizer, BertTokenizer, GPTTokenizer
from research.subtoken.text.data import SubtokenLLMExample


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


class GPTPacker(
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


def token_ids_to_strings(token_ids: List[int], tokenizer) -> List[str]:
    return tokenizer.batch_decode(
        [[tid] for tid in token_ids], skip_special_tokens=True
    )


def string_to_bytes_list(s: str) -> List[int]:
    return list(s.encode("utf-8"))


class SubtokenGPTPacker(
    AbstractPacker,
):
    FORBIDDEN_TOKENS = [45706]
    SPECIAL_TOKENS = [50256]
    TOKEN_REPLACEMENTS = {
        15171: [220, 16733, 7992],
        20727: [220, 403, 18789],
        27473: [220, 46813, 20860],
        28719: [220, 6381, 1676, 16864, 1286],
        29760: [220, 29609, 420, 28018],
        29789: [220, 25413, 4625, 5646],
        30210: [220, 48317, 13739, 3118, 18143],
        30213: [220, 22615, 2514, 27881, 10049],
        30982: [220, 6381, 1676, 16864, 378],
        32799: [220, 22769, 1634],
        34400: [220, 1783],
        36174: [220, 29531, 34832, 35992],
        36573: [220, 24588, 19541],
        36658: [220, 4770, 28],
        37389: [220, 12100, 12100],
        40586: [220, 38986, 282, 1023],
        40800: [220, 9979, 2738, 453],
        40887: [220, 70, 459, 305, 36387],
        42045: [220, 298, 7537, 72, 16607],
        43453: [220, 46933, 42202],
        43649: [220, 521, 396, 41726],
        44436: [220, 298, 10406, 1834, 1056],
        44713: [220, 4181],
        45545: [220, 26503, 44686, 42983],
        46674: [220, 24588, 27781],
        47757: [220, 259, 785, 3866, 5135, 856],
        48667: [220, 31709, 20860],
        16529: [220, 1783, 1783, 1783, 1783],
        20368: [220, 1783, 1783],
        38093: [220, 4770, 4770, 4770, 4770, 28],
        41436: [220, 35937, 35937],
        41906: [220, 8412, 8412],
        46111: [220, 4770, 4770, 28],
        30899: [21018, 30898],
        39177: [7449, 39142],
        39753: [39752, 10493],
        39755: [39714, 39655],
        39756: [24807, 31208],
        39757: [17620, 29841],
        40242: [39693, 40241],
        41380: [21353, 41215],
        30906: [30905, 21018, 30898],
        31576: [22615, 31573],
        3880: [1783, 1783],
        8864: [4181, 4181],
        10052: [4770, 4770],
        10097: [1783, 1783, 1783, 1783],
        10221: [4841, 4841],
        14950: [8184, 8184],
        17174: [8412, 8412],
        19351: [35937, 35937],
        22369: [10541, 10541],
        23193: [4181, 4181, 4181, 4181],
        14827: [9364, 9364],
        23090: [9364, 9364, 9364, 9364],
        23926: [4770, 4770, 4770, 4770],
        27006: [15243, 15243],
        27193: [4841, 4841, 4841, 4841],
        27754: [2109, 2109, 2109],
        49129: [16317, 16317, 16317],
        28542: [16068, 16068],
        29113: [14468, 14468],
        29146: [15864, 15864],
        30542: [8184, 8184, 8184, 8184],
        32941: [2602, 2602, 2602],
        49527: [20503, 20503],
        49704: [27246, 27246],
        35496: [9364, 9364, 9364, 9364, 9364, 9364, 9364, 9364],
        43801: [26171, 26171, 26171, 26171],
        47232: [1783, 1783, 1783],
        39172: [17811, 17811],
    }

    def __init__(
        self,
        sequence_length: int,
        dataset_maker: AbstractDataset,
        tokenizer_maker: Callable[[], GPTTokenizer],
        seed: Optional[int] = None,
    ):
        super().__init__(
            sequence_length,
            dataset_maker,
            tokenizer_maker,
            seed=seed,
        )

    def get_sample(self) -> SubtokenLLMExample:
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
            tokens_with_replacement = []
            skip_document = False
            for t in tokens:
                if t in self.FORBIDDEN_TOKENS:
                    # raise ValueError(
                    #     "Token cannot be in FORBIDDEN_TOKENS"
                    # )  # TODO: somehow the forbidden tokens still show up :/ debug that
                    skip_document = True
                    break
                elif t in self.TOKEN_REPLACEMENTS:
                    tokens_with_replacement.extend(self.TOKEN_REPLACEMENTS[t])
                else:
                    tokens_with_replacement.append(t)
            if skip_document:
                continue
            assert len(tokens_with_replacement) >= len(tokens)
            tokens = tokens_with_replacement

            buffer.extend(tokens + [eot_id])

            document_lengths.append(len(tokens) + 1)
            if (sum(document_lengths) - max(document_lengths)) > self.sequence_length:
                break

        sample_start = self.py_rng.randint(0, len(buffer) - 1)
        sample_end = sample_start + self.sequence_length

        input_ids = list(take_circular(buffer, sample_start, sample_end))
        input_bytes = []
        for token in input_ids:
            if token != self.tokenizer.eot_id:
                # bytes_ = list(
                #     bytes(self.tokenizer.token_id_to_text[token].encode("utf-8"))
                # )
                # bytes_ = self.tokenizer.tokenizer.convert_tokens_to_string(
                #     self.tokenizer.tokenizer.convert_ids_to_tokens(token)
                # ).encode("utf-8")
                bytes_ = list(self.tokenizer.token_id_to_text[token].encode("utf-8"))
                assert len(bytes_) <= self.tokenizer.MAX_BYTES_PER_TOKEN
                input_bytes.append(
                    bytes_ + [-1] * (self.tokenizer.MAX_BYTES_PER_TOKEN - len(bytes_))
                )
            else:
                input_bytes.append([-1] * self.tokenizer.MAX_BYTES_PER_TOKEN)
        target_ids = list(take_circular(buffer, sample_start + 1, sample_end + 1))
        calculate_loss = [1] * len(target_ids)
        # is_special_token = [t in self.SPECIAL_TOKENS for t in input_ids]

        return SubtokenLLMExample(input_ids, input_bytes, target_ids, calculate_loss)
