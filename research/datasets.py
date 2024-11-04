from functools import partial
from typing import Literal, Optional
from dataclasses import dataclass
import copy

import torch
from torch.utils.data import DataLoader

from lizrd.text import datasets, packers, data, tokenizers
from lizrd.support.misc import calculate_current_bsz_from_rampup, get_ith_chunk


@dataclass
class BatchSizeRampupConfig:
    batch_size_rampup_transition_points: Optional[list[float]]
    batch_size_rampup_sizes: Optional[list[float]]

    def __post_init__(self):
        self.enabled = self.batch_size_rampup_sizes is not None


class DataloaderWrapper:
    def __init__(
        self,
        dataloader: DataLoader,
        batch_size_rampup_config: BatchSizeRampupConfig,
        total_n_gpus: int,
        device: torch.device,
    ):
        self.generator = iter(dataloader)
        self.target_batch_size = dataloader.batch_size
        self.batch_size_rampup_config = batch_size_rampup_config
        self.device = device
        self.num_of_last_batch_chunk = None
        self.total_n_gpus = total_n_gpus

    def get_batch(self, num_processed_tokens_so_far: int) -> data.LLMBatch:
        if self.batch_size_rampup_config.batch_size_rampup_sizes is None:
            return next(self.generator).to(self.device)
        else:
            current_batch_size = calculate_current_bsz_from_rampup(
                num_processed_tokens_so_far,
                self.batch_size_rampup_config.batch_size_rampup_transition_points,
                self.batch_size_rampup_config.batch_size_rampup_sizes,
            )
            current_num_chunks = self.batch_size // current_batch_size
            current_chunk = (
                num_processed_tokens_so_far // current_batch_size
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
    total_n_gpus: int,
    batch_size_rampup_config: Optional[BatchSizeRampupConfig] = None,
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

    return DataloaderWrapper(dataloader, batch_size_rampup_config, total_n_gpus, device)
