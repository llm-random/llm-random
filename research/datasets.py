from functools import partial
from typing import Literal, Optional

import torch
from torch.utils.data import DataLoader

from lizrd.text import datasets, packers, data, tokenizers


class DataloaderWrapper:
    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.generator = iter(dataloader)
        self.device = device
        self.dataloader = dataloader #dev 

    def get_batch(self) -> data.LLMBatch:
        return next(self.generator).to(self.device)


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
