from typing import Literal

import torch
from torch.utils.data import DataLoader

from lizrd.text import datasets, packers, data, tokenizers


class DataloaderWrapper:
    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.generator = iter(dataloader)
        self.device = device

    def get_batch(self) -> data.LLMBatch:
        return next(self.generator).to(self.device)


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
):
    if dataset_type == "wikibook":
        dataset = datasets.WikiBookDataset(
            use_dummy_dataset=use_dummy_dataset,
            split=dataset_split,
        )
    elif dataset_type == "c4":
        if use_dummy_dataset:
            print("WARNING: Dummy dataset not supported for C4 dataset")
        dataset = datasets.C4Dataset(
            split=dataset_split,
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
            dataset=dataset,
            tokenizer_maker=tokenizers.GPTTokenizer,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    dataloader = DataLoader(
        packer,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=data.LLMBatch,
        worker_init_fn=lambda worker_id: packer.set_rng(seed + worker_id),
        shuffle=False,
        pin_memory=True,
    )

    return DataloaderWrapper(dataloader, device)
