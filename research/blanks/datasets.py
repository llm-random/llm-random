from functools import partial
from typing import Callable, Literal

import torch
from torch.utils.data import DataLoader

from lizrd.text import datasets, tokenizers
from research.datasets import DataloaderWrapper, worker_init_fn
from .packer import BlankEvalPacker, BlankPacker
import research.blanks.data as data


def get_processed_dataset(
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    num_workers: int,
    seed: int,
    tokenizer_maker: Callable[[], tokenizers.AbstractTokenizer],
    dataset_type: Literal["wikibook", "c4"] = "wikibook",
    use_dummy_dataset: bool = False,
    dataset_split: str = "train",
    n_blanks: int = 0,
    use_only_last_blank_loss: bool = False,
):
    if dataset_type == "wikibook":
        dataset = datasets.WikiBookDataset(
            use_dummy_dataset=use_dummy_dataset,
            split=dataset_split,
        )
    elif dataset_type == "c4":
        dataset = datasets.C4Dataset(
            use_dummy_dataset=use_dummy_dataset,
            split=dataset_split,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    if dataset_split == "train":
        packer = BlankPacker(
            sequence_length=sequence_length,
            dataset=dataset,
            tokenizer_maker=tokenizer_maker,
            n_blanks=n_blanks,
            use_only_last_blank_loss=use_only_last_blank_loss,
        )
    elif dataset_split in ["eval", "validation"]:
        packer = BlankEvalPacker(
            sequence_length=sequence_length,
            dataset=dataset,
            tokenizer_maker=tokenizer_maker,
            n_blanks=n_blanks,
        )
    else:
        raise ValueError(f"Unknown dataset split: {dataset_split}")

    dataloader = DataLoader(
        packer,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=data.BlanxBatch,
        worker_init_fn=partial(worker_init_fn, seed),
        shuffle=False,
        pin_memory=True,
    )

    return DataloaderWrapper(dataloader, device)
