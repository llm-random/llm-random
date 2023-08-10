import random

from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import GPT2TokenizerFast


NUM_C4_TOKENS = 173_648_052_806 # number of tokens in the C4 dataset


class C4Dataset(Dataset):
    def __init__(self, seq_length: int, batch_size: int, split: str = "train"):
        self.dataset = load_dataset("c4", "en", split=split)
        self.seq_length = seq_length
        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            "gpt2", additional_special_tokens=["<sequence_sep>"]
        )

        # set model max length to high number to disable warnings
        # we handle sequence length ourselves
        self.tokenizer.model_max_length = 100_000

        self.sequence_separator_id = self.tokenizer.additional_special_tokens_ids[0]
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"Using C4 dataset consisting of {NUM_C4_TOKENS} tokens")
        print(
            f"One epoch with batch {batch_size} and sequence length {seq_length} will take {NUM_C4_TOKENS // (batch_size * seq_length)} steps"
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
        """
        Sample examples from the dataset until we reach the desired sequence length.
        """
        result = dict()
        result["input_ids"] = []
        result["attention_mask"] = []
        current_length = 0
        document_id = idx
        rand = random.Random(idx)
        while current_length < self.seq_length:
            example = self.get_one_example(
                document_id, self.seq_length - current_length
            )
            current_length += len(example)
            result["input_ids"] += example
            result["attention_mask"] += [1] * len(example)
            if current_length < self.seq_length:
                result["input_ids"] += [self.sequence_separator_id]
                result["attention_mask"] += [0]
                current_length += 1
                document_id = rand.randint(0, len(self.dataset) - 1)
        return result

    def get_one_example(self, idx, length):
        text = self.dataset[idx]["text"]
        ids = self.tokenizer(text)["input_ids"]
        if len(ids) > length:
            return self.get_random_chunk(ids, length)
        else:
            return ids

    def get_random_chunk(self, ids, length):
        beginning = random.randint(0, len(ids) - length)
        return ids[beginning : beginning + length]
