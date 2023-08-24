from abc import ABC
from typing import List, Optional
from transformers import BertTokenizer, GPT2TokenizerFast


class AbstractTokenizer(ABC):
    sequence_separator_id: Optional[int]
    mask_id: Optional[int]
    pad_id: Optional[int]
    vocab_size: int

    def text_to_ids(self, text: str) -> List[int]:
        raise NotImplementedError()


class BertTokenizer(AbstractTokenizer):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.sequence_separator_id = self.tokenizer.convert_tokens_to_ids("[SEP]")
        self.mask_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
        self.pad_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
        self.vocab_size = 30522


class GPTTokenizer(AbstractTokenizer):
    def __init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            "gpt2", additional_special_tokens=["[SEP]"]
        )
        self.sequence_separator_id = self.tokenizer.convert_tokens_to_ids("[SEP]")
        self.mask_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
        self.pad_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
        self.vocab_size = 50257

    def text_to_ids(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)
