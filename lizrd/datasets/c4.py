from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import GPT2TokenizerFast
import numpy as np
import torch

from lizrd.datasets.processed_batch import ProcessedBatch


class C4Dataset(Dataset):
    """
    Explanation...
    """
    def __init__(self, seq_length: int, batch_size: int, split: str = "train"):
        self.dataset = load_dataset("c4", "en", split=split)
        self.seq_length = seq_length
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_special_tokens("<sequence_sep>")
        print(f"Using C4 dataset consisting of {len(self.dataset)} samples")
        print(
            f"One epoch with batch {batch_size} will take {len(self.dataset) // batch_size} steps"
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = dict()
        tokenized = self.get_tokenized_sample(idx)
        example["tokens"] = tokenized["input_ids"]
        example["non_padded_mask"] = tokenized["attention_mask"]
        example["target_tokens"] = example["tokens"][1:] + [self.tokenizer.pad_token_id]
        return example

    def get_tokenized_sample(self, idx):
        pass

    def get_one_example(self, idx, length):
        result = dict()
        text = self.dataset[idx]
        ids = self.tokenizer(text)["input_ids"]
        if len(ids > length):
            return self.get_random_chunk(ids, length)
        else:
            return ids

    def get_random_chunk(self, ids, length):
        pass


class ProcessedC4Batch(ProcessedBatch):
    def __init__(self, processed_examples):
        super().__init__(processed_examples)
        self.tokens = self._make_tensor(
            [example["tokens"] for example in processed_examples]
        )
        self.target_tokens = self._make_tensor(
            [example["target_tokens"] for example in processed_examples]
        )
        self.non_padded_mask = self._make_tensor(
            [example["non_padded_mask"] for example in processed_examples]
        )

    def _make_tensor(self, list_of_token_lists):
        matrix = np.array(list_of_token_lists)
        return torch.from_numpy(matrix)

    def to(self, device):
        self.tokens = self.tokens.to(device)
        self.target_tokens = self.target_tokens.to(device)
        self.non_padded_mask = self.non_padded_mask.to(device)
        return self
