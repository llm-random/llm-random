from functools import partial
from typing import Literal, Optional
import copy

import torch
from torch.utils.data import DataLoader
from lizrd.text import datasets, data, tokenizers, packers
from lizrd.support.misc import get_ith_chunk
from research.mole.utils.data import LLMMetaBatch
from research.mole.utils.packers import GPTMetaPOSPacker


class DataloaderWrapper:
    def __init__(
        self,
        dataloader: DataLoader,
        device: torch.device,
    ):
        self.generator = iter(dataloader)
        self.target_batch_size_per_gpu = dataloader.batch_size
        self.device = device
        self.previous_batch_size_per_gpu = -1
        self.chunks_iterator = 0

    def get_batch(self, current_batch_size_per_gpu=-1) -> data.LLMBatch:
        """
        Returns the next batch of data, handling batch size ramp-up if specified.

        If `current_batch_size_per_gpu` is less than `self.target_batch_size`, the batch is split into smaller chunks, and the appropriate chunk is returned.

        Args:
            current_batch_size_per_gpu (int, optional): The current batch size.
            Defaults to -1, which uses the target batch size.
        """
        if (
            current_batch_size_per_gpu == -1
            or current_batch_size_per_gpu == self.target_batch_size_per_gpu
        ):
            return next(self.generator).to(self.device)
        else:
            current_num_chunks = (
                self.target_batch_size_per_gpu // current_batch_size_per_gpu
            )
            if self.batch_size_changed(current_batch_size_per_gpu):
                self.chunks_iterator = 0
                self.previous_batch_size_per_gpu = current_batch_size_per_gpu
            current_chunk_index = self.chunks_iterator % current_num_chunks

            if current_chunk_index == 0:
                self.current_batch = next(self.generator).to(self.device)

            self.chunks_iterator += 1

            return self.get_batch_chunk(
                self.current_batch, current_num_chunks, current_chunk_index
            )

    def batch_size_changed(self, current_batch_size_per_gpu):
        return current_batch_size_per_gpu != self.previous_batch_size_per_gpu

    def get_batch_chunk(self, batch, num_chunks, chunk_index):
        batch_chunk = copy.deepcopy(batch)
        for _, tensor in batch_chunk:
            tensor.data = get_ith_chunk(tensor.data, num_chunks, chunk_index)
        return batch_chunk


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
    biased: Optional[str] = True,
    pos_grouped: Optional[dict] = None,
    n_experts: Optional[int] = None
):
    # assert (biased and pos_grouped and n_experts) or not(biased or pos_grouped or n_experts), "Have to provide n_experts and pos_grouped when using biased datapacker." #dev
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
    
    collate_fn = data.LLMBatch

    if model_type == "bert":
        packer = packers.BERTPacker(
            sequence_length=sequence_length,
            dataset=dataset,
            tokenizer_maker=tokenizers.BertTokenizer,
        )
    elif model_type == "gpt" and biased:
        packer = GPTMetaPOSPacker(
            sequence_length=sequence_length,
            dataset_maker=dataset,
            pos_grouped=pos_grouped,
            n_experts=n_experts,
            tokenizer_maker=tokenizers.GPTTokenizer,
        )
        collate_fn = LLMMetaBatch
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
        collate_fn=collate_fn,
        worker_init_fn=partial(worker_init_fn, seed),
        shuffle=False,
        pin_memory=True,
    )

    return DataloaderWrapper(dataloader, device)
