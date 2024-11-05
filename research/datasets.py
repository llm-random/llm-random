from functools import partial
from typing import Literal, Optional
from dataclasses import dataclass
import copy

import torch
from torch.utils.data import DataLoader

from lizrd.text import datasets, packers, data, tokenizers
from lizrd.support.misc import get_ith_chunk


@dataclass
class BatchSizeRampupConfig:
    transition_points: list[float]
    batch_sizes: list[float]


class DataloaderWrapper:
    def __init__(
        self,
        dataloader: DataLoader,
        device: torch.device,
    ):
        self.generator = iter(dataloader)
        self.target_batch_size = dataloader.batch_size
        self.device = device

    def get_batch(
        self, current_batch_size_per_gpu=-1, num_processed_tokens_so_far=-1
    ) -> data.LLMBatch:
        """
        Returns the next batch of data, handling batch size ramp-up if specified.

        If `current_batch_size_per_gpu` is less than `self.target_batch_size`, the batch is split into
        smaller chunks, and the appropriate chunk is returned based on `num_processed_tokens_so_far`.

        Args:
            current_batch_size_per_gpu (int, optional): The current batch size.
            Defaults to -1, which uses the target batch size.
            num_processed_tokens_so_far (int, optional): Total number of tokens processed so far; used to determine the current chunk when batch size ramp-up is in effect. Defaults to -1.
        """
        if (
            current_batch_size_per_gpu == -1
            or current_batch_size_per_gpu == self.target_batch_size
        ):
            return next(self.generator).to(self.device)
        else:
            current_num_chunks = self.target_batch_size // current_batch_size_per_gpu
            current_chunk = (
                num_processed_tokens_so_far // current_batch_size_per_gpu
            ) % current_num_chunks

            if current_chunk == 0:
                self.current_batch = next(self.generator).to(self.device)

            batch = copy.deepcopy(self.current_batch)
            for _, tensor in batch:
                tensor.data = get_ith_chunk(
                    tensor.data, current_num_chunks, current_chunk
                )

            return batch


def worker_init_fn(seed, worker_id):
    worker_info = torch.utils.data.get_worker_info()
    packer: packers.AbstractPacker = (
        worker_info.dataset
    )  # the dataset copy in this worker process
    packer.set_rng(seed + worker_id)


def get_processed_dataset(
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    num_workers: int,
    seed: int,
    model_type: Literal["bert", "gpt"] = "bert",
    dataset_type: Literal["wikibook", "c4"] = "wikibook",
    use_dummy_dataset: bool = False,
    dataset_split: str = "train",
    dataset_path: Optional[str] = None,
):
    if dataset_type == "wikibook":
        dataset = partial(
            datasets.WikiBookDataset,
            use_dummy_dataset=use_dummy_dataset,
            split=dataset_split,
        )
    elif dataset_type == "c4":
        dataset = partial(
            datasets.C4Dataset,
            use_dummy_dataset=use_dummy_dataset,
            split=dataset_split,
            dataset_path=dataset_path,
        )
    elif dataset_type == "fineweb-edu":
        dataset = partial(
            datasets.FinewebEduDataset,
            use_dummy_dataset=use_dummy_dataset,
            split=dataset_split,
            dataset_path=dataset_path,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    if model_type == "bert":
        packer = packers.BERTPacker(
            sequence_length=sequence_length,
            dataset=dataset,
            tokenizer_maker=tokenizers.BertTokenizer,
        )
    elif model_type == "gpt":
        packer = packers.GPTPacker(
            sequence_length=sequence_length,
            dataset_maker=dataset,
            tokenizer_maker=tokenizers.GPTTokenizer,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    dataloader = DataLoader(
        packer,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=data.LLMBatch,
        worker_init_fn=partial(worker_init_fn, seed),
        shuffle=False,
        pin_memory=True,
    )

    return DataloaderWrapper(dataloader, device)
