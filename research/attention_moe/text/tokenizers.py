from abc import ABC, abstractmethod
from typing import List, Optional
from transformers import BertTokenizerFast, GPT2TokenizerFast


class AbstractTokenizer(ABC):
    VOCAB_SIZE: int
    sequence_separator_id: Optional[int]
    mask_id: Optional[int]
    eot_id: Optional[int]
    blanks_ids: Optional[List[int]]

    @abstractmethod
    def text_to_ids(self, text: str) -> List[int]:
        raise NotImplementedError()


def disable_tokenizer_warnings(hf_tokenizer):
    # set model max length to high number to disable warnings
    # we handle sequence length ourselves
    hf_tokenizer.model_max_length = 100_000


class BertTokenizer(AbstractTokenizer):
    VOCAB_SIZE = 30522

    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        disable_tokenizer_warnings(self.tokenizer)
        self.mask_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
        assert isinstance(self.mask_id, int)
        self.sequence_separator_id = self.tokenizer.convert_tokens_to_ids("[SEP]")
        assert isinstance(self.sequence_separator_id, int)

    def text_to_ids(self, text: str) -> List[int]:
        # TODO: encode or tokenize + convert_tokens_to_ids?
        return self.tokenizer.encode(text)


class GPTTokenizer(AbstractTokenizer):
    VOCAB_SIZE = 50257

    def __init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        disable_tokenizer_warnings(self.tokenizer)
        self.eot_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")

        assert isinstance(self.eot_id, int)

    def text_to_ids(self, text: str) -> List[int]:
        # TODO: encode or tokenize + convert_tokens_to_ids?
        return self.tokenizer.encode(text)
