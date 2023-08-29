from abc import ABC, abstractmethod
from typing import List, Optional
from transformers import BertTokenizer, GPT2TokenizerFast


class AbstractTokenizer(ABC):
    sequence_separator_id: Optional[int]
    mask_id: Optional[int]
    pad_id: Optional[int]
    vocab_size: int

    @abstractmethod
    def text_to_ids(self, text: str) -> List[int]:
        raise NotImplementedError()


class BertTokenizer(AbstractTokenizer):
    vocab_size = 30522

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.mask_id = self.tokenizer.convert_tokens_to_ids("[MASK]")

    def text_to_ids(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)


class GPTTokenizer(AbstractTokenizer):
    vocab_size = 50257

    def __init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.eot_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")

    def text_to_ids(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)
