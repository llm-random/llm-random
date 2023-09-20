from functools import partial
from typing import Callable, Literal, Optional

import torch
from torch.utils.data import DataLoader

from lizrd.text import datasets, packers, data, tokenizers


def get_tokenizer_maker(
    model_type: str, tokenizer: Optional[str] = None
) -> Callable[[], tokenizers.AbstractTokenizer]:
    if tokenizer is None:
        if model_type == "bert":
            tokenizer = "bert"
        elif model_type == "gpt":
            tokenizer = "gpt"

    if tokenizer == "bert":
        return tokenizers.BertTokenizer
    elif tokenizer == "gpt":
        return tokenizers.GPTTokenizer
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer}")


class DataloaderWrapper:
    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.generator = iter(dataloader)
        self.device = device

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
    tokenizer_maker: Callable[[], tokenizers.AbstractTokenizer],
    model_type: Literal["bert", "gpt"] = "bert",
    dataset_type: Literal["wikibook", "c4"] = "wikibook",
    use_dummy_dataset: bool = False,
    dataset_split: str = "train",
    n_blanks: int = 0,
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

    if model_type == "bert":
        packer = packers.BERTPacker(
            sequence_length=sequence_length,
            dataset=dataset,
            tokenizer_maker=tokenizer_maker,
        )
    elif model_type == "gpt":
        if n_blanks == 0:
            packer = packers.GPTPacker(
                sequence_length=sequence_length,
                dataset=dataset,
                tokenizer_maker=tokenizer_maker,
            )
        else:
            packer = packers.BlankPacker(
                sequence_length=sequence_length,
                dataset=dataset,
                tokenizer_maker=tokenizer_maker,
                n_blanks=n_blanks,
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
