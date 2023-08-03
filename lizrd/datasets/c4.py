from random import Random

from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import GPT2TokenizerFast


class C4Dataset(Dataset):
    def __init__(self, seq_length: int, split: str = "train", seed: int = 42):
        self.dataset = load_dataset("c4", "en", split=split)
        self.seq_length = seq_length
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.random = Random(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tokenized = self.tokenizer(self.dataset[idx]["text"], truncation=True)