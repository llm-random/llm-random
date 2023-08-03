from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import GPT2TokenizerFast


class C4Dataset(Dataset):
    def __init__(self, seq_length: int, batch_size: int, split: str = "train"):
        self.dataset = load_dataset("c4", "en", split=split)
        self.seq_length = seq_length
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"Using C4 dataset consisting of {len(self.dataset)} samples")
        print(f"One epoch with batch {batch_size} will take {len(self.dataset) // batch_size} steps")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.tokenizer(
            self.dataset[idx]["text"],
            max_length=self.seq_length,
            truncation=True,
            padding="max_length",
        )
