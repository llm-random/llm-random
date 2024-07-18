from functools import partial
from typing import Callable, Literal, Optional

import torch
from torch.utils.data import DataLoader

from lizrd.text import datasets
from research.datasets import DataloaderWrapper, worker_init_fn
from research.tokenizex.utils.data import TokenizexBatch
from research.tokenizex.model.tokenizer import TokenizexTokenizer
from research.tokenizex.utils.packer import TokenizexGPTPacker


def get_processed_dataset(
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    num_workers: int,
    seed: int,
    tokenizer_maker: Callable[[], TokenizexTokenizer],
    dataset_type: Literal["wikibook", "c4"] = "wikibook",
    use_dummy_dataset: bool = False,
    dataset_split: str = "train",
    dataset_path: Optional[str] = None,
) -> DataloaderWrapper:
    if dataset_type == "wikibook":
        dataset_maker = partial(
            datasets.WikiBookDataset,
            use_dummy_dataset=use_dummy_dataset,
            split=dataset_split,
        )
    elif dataset_type == "c4":
        dataset_maker = partial(
            datasets.C4Dataset,
            use_dummy_dataset=use_dummy_dataset,
            split=dataset_split,
            dataset_path=dataset_path,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    packer = TokenizexGPTPacker(
        sequence_length=sequence_length,
        dataset_maker=dataset_maker,
        tokenizer_maker=tokenizer_maker,
        seed=seed,
    )

    dataloader = DataLoader(
        packer,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=TokenizexBatch,
        worker_init_fn=partial(worker_init_fn, seed),
        shuffle=False,
        pin_memory=True,
    )

    return DataloaderWrapper(dataloader, device)
