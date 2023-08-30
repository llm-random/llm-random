from abc import ABC, abstractmethod
from typing import List, Optional
from transformers import BertTokenizerFast, GPT2TokenizerFast


class AbstractTokenizer(ABC):
    sequence_separator_id: Optional[int]
    mask_id: Optional[int]
    eot_id: Optional[int]
    vocab_size: int

    @abstractmethod
    def text_to_ids(self, text: str) -> List[int]:
        raise NotImplementedError()


class BertTokenizer(AbstractTokenizer):
    vocab_size = 30522

    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.mask_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
        # set model max length to high number to disable warnings
        # we handle sequence length ourselves
        self.tokenizer.model_max_length = 100_000
        self.sequence_separator_id = self.tokenizer.convert_tokens_to_ids("[SEP]")

    def text_to_ids(self, text: str) -> List[int]:
        # TODO: encode or tokenize + convert_tokens_to_ids?
        return self.tokenizer.encode(text)


class GPTTokenizer(AbstractTokenizer):
    vocab_size = 50257

    def __init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        # set model max length to high number to disable warnings
        # we handle sequence length ourselves
        self.tokenizer.model_max_length = 100_000
        self.eot_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")

    def text_to_ids(self, text: str) -> List[int]:
        # TODO: encode or tokenize + convert_tokens_to_ids?
        return self.tokenizer.encode(text)
