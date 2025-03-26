import os
import torch
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from typing import Callable, Optional, Literal, List, Iterator
from functools import partial
from datasets import load_dataset, load_from_disk
from attr import dataclass
import itertools
import numpy as np
from abc import ABC, abstractmethod
import random
from torch.utils.data import IterableDataset, DataLoader
from transformers import GPT2TokenizerFast
from abc import ABC, abstractmethod
import torch.distributed as dist
from torch.nn.attention import SDPBackend
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp import MixedPrecision
from torch.nn import (
    LayerNorm as LayerNorm,
)  # used by FSDP, but it keeps getting removed during file formatting
import sys


@dataclass
class LLMExample(object):
    input_ids: List[int]
    target_ids: List[int]
    should_calculate_loss: List[
        int
    ]  # e.g. in BERT loss is not calculated over non-masked tokens


class LLMBatch:
    def __init__(self, examples: List[LLMExample]):
        self.input_ids = self._make_tensor([example.input_ids for example in examples])
        self.target_ids = self._make_tensor(
            [example.target_ids for example in examples]
        )
        self.should_calculate_loss = self._make_tensor(
            [example.should_calculate_loss for example in examples]
        )
        self.N = len(examples)  # assuming all tensors have the first dimension as N
        self.set_n_chunks(n_chunks=1)

        assert self.input_ids.shape == self.target_ids.shape
        assert self.input_ids.shape == self.should_calculate_loss.shape

    def pin_memory(self):
        """Pin memory for faster transfer to GPU as described in https://pytorch.org/docs/stable/data.html#memory-pinning"""
        self.input_ids = self.input_ids.pin_memory()
        self.target_ids = self.target_ids.pin_memory()
        self.should_calculate_loss = self.should_calculate_loss.pin_memory()
        return self

    def set_n_chunks(self, n_chunks: int):
        assert (
            self.N % n_chunks == 0
        ), "total_batch_size must be divisible by n_chunks without remainder."

        self.chunk_size = self.N // n_chunks
        self.idx = 0

    def __iter__(self):
        return self

    @property
    def device(self) -> torch.device:
        assert (
            self.input_ids.device
            == self.target_ids.device
            == self.should_calculate_loss.device
        )
        return self.input_ids.device

    def to(self, device) -> "LLMBatch":
        self.input_ids = self.input_ids.to(device)
        self.target_ids = self.target_ids.to(device)
        self.should_calculate_loss = self.should_calculate_loss.to(device)
        return self

    def _make_tensor(self, list_of_token_lists: List[List[int]]) -> torch.Tensor:
        matrix = np.array(list_of_token_lists)
        return torch.from_numpy(matrix)

    def __next__(self):
        if self.idx < self.N:
            chunk_input_ids = self.input_ids[self.idx : self.idx + self.chunk_size]
            chunk_target_ids = self.target_ids[self.idx : self.idx + self.chunk_size]
            chunk_should_calculate_loss = self.should_calculate_loss[
                self.idx : self.idx + self.chunk_size
            ]
            self.idx += self.chunk_size
            return chunk_input_ids, chunk_target_ids, chunk_should_calculate_loss
        else:
            raise StopIteration


class AbstractDataset:
    def __init__(self, seed: Optional[int] = None):
        self.set_rng(seed)

    def set_rng(self, seed: Optional[int] = None):
        np_rng = np.random.default_rng(seed)
        py_rng = random.Random(seed)

        self.np_rng = np_rng
        self.py_rng = py_rng

    @abstractmethod
    def get_document(self) -> str:
        raise NotImplementedError()


class C4Dataset(AbstractDataset):
    total_gpt2_tokens = 173_648_052_806  # number of tokens in the C4 dataset when using GPT2TokenizerFast

    def __init__(
        self,
        seed: Optional[int] = None,
        split: str = "train",
        use_dummy_dataset: bool = False,
        dataset_path: Optional[str] = None,
    ):
        super().__init__(seed=seed)
        assert split in ["train", "validation"]
        if dataset_path is not None:
            self.dataset = load_from_disk(dataset_path)
        elif use_dummy_dataset:
            if split != "train":
                raise NameError(
                    "Dummy dataset only supports train split for C4 dataset"
                )
            self.dataset = load_dataset("stas/c4-en-10k", split=split)
        else:
            self.dataset = load_dataset("c4", "en", split=split)

    def get_document(self) -> str:
        id = self.py_rng.randint(0, len(self.dataset) - 1)
        return self.dataset[id]["text"]


def take_circular(iterable, start, stop):
    cycle = itertools.cycle(iterable)
    return itertools.islice(cycle, start, stop)


class AbstractTokenizer(ABC):
    VOCAB_SIZE: int
    sequence_separator_id: Optional[int]
    mask_id: Optional[int]
    eot_id: Optional[int]
    blanks_ids: Optional[List[int]]

    @abstractmethod
    def text_to_ids(self, text: str) -> List[int]:
        raise NotImplementedError()


def disable_tokenizer_warnings(hf_tokenizer):
    # set model max length to high number to disable warnings
    # we handle sequence length ourselves
    hf_tokenizer.model_max_length = 100_000


class GPTTokenizer(AbstractTokenizer):
    VOCAB_SIZE = 50257

    def __init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        disable_tokenizer_warnings(self.tokenizer)
        self.eot_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")

        assert isinstance(self.eot_id, int)

    def text_to_ids(self, text: str) -> List[int]:
        # TODO: encode or tokenize + convert_tokens_to_ids?
        return self.tokenizer.encode(text)


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
        # sample_start = int(self.np_rng.integers(0, len(buffer) - 1))
        sample_end = sample_start + self.sequence_length

        input_ids = list(take_circular(buffer, sample_start, sample_end))
        target_ids = list(take_circular(buffer, sample_start + 1, sample_end + 1))
        calculate_loss = [1] * len(target_ids)

        return LLMExample(input_ids, target_ids, calculate_loss)


class DataloaderWrapper:
    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.generator = dataloader
        self.device = device

    # def get_batch(self) -> LLMBatch:
    #     return next(self.generator).to(self.device)

    def __iter__(self):
        return iter(self.generator)


def worker_init_fn(seed, worker_id):
    worker_info = torch.utils.data.get_worker_info()
    packer: AbstractPacker = (
        worker_info.dataset
    )  # the dataset copy in this worker process
    packer.set_rng(seed + worker_id)


def get_dataloaders(
    total_batch_size: int,
    sequence_length: int,
    device: torch.device,
    num_workers: int,
    seed: int,
    model_type: Literal["gpt"] = "gpt",
    dataset_type: Literal["c4"] = "c4",
    use_dummy_dataset: bool = False,
    training_dataset_path: Optional[str] = None,
    eval_dataset_path: Optional[str] = None,
):
    batch_size_per_device = total_batch_size // int(os.environ["WORLD_SIZE"])
    train_dataloader = get_processed_dataset(
        batch_size=batch_size_per_device,
        sequence_length=sequence_length,
        device=device,
        num_workers=num_workers,
        seed=seed,
        model_type=model_type,
        dataset_type=dataset_type,
        use_dummy_dataset=use_dummy_dataset,
        dataset_split="train",
        dataset_path=training_dataset_path,
    )

    eval_dataloader = get_processed_dataset(
        batch_size=batch_size_per_device,
        sequence_length=sequence_length,
        device=device,
        num_workers=num_workers,
        seed=seed + 1 if use_dummy_dataset else seed,
        model_type=model_type,
        dataset_type=dataset_type,
        use_dummy_dataset=True,  # there is no validation dataset for c4
        dataset_split="train" if use_dummy_dataset else "validation",
        dataset_path=eval_dataset_path,
    )
    return train_dataloader, eval_dataloader


def get_processed_dataset(
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    num_workers: int,
    seed: int,
    model_type: Literal["gpt"] = "gpt",
    dataset_type: Literal["c4"] = "c4",
    use_dummy_dataset: bool = False,
    dataset_split: str = "train",
    dataset_path: Optional[str] = None,
):

    if dataset_type == "c4":
        dataset = partial(
            C4Dataset,
            use_dummy_dataset=use_dummy_dataset,
            split=dataset_split,
            dataset_path=dataset_path,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    if model_type == "gpt":
        packer = GPTPacker(
            sequence_length=sequence_length,
            dataset_maker=dataset,
            tokenizer_maker=GPTTokenizer,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    dataloader = DataLoader(
        packer,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=LLMBatch,
        worker_init_fn=partial(worker_init_fn, seed),
        shuffle=False,
        pin_memory=True,
    )

    return DataloaderWrapper(dataloader, device)
