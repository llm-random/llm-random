from abc import ABC, abstractmethod
from typing import List, Optional
from transformers import BertTokenizerFast, GPT2TokenizerFast

from lizrd.text.tokenizers import AbstractTokenizer, disable_tokenizer_warnings

NUM_RESERVED_TOKENS = 100


class BlankTokenizer(AbstractTokenizer):
    VOCAB_SIZE = 50257 + NUM_RESERVED_TOKENS

    def __init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        disable_tokenizer_warnings(self.tokenizer)

        self.eot_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        blank_text = "<|blank|>"
        self.tokenizer.add_tokens([blank_text])
        extra_tokens = 0
        while self.tokenizer.VOCAB_SIZE < self.VOCAB_SIZE:
            self.tokenizer.add_tokens([f"<|extra_{extra_tokens}|>"])
            extra_tokens += 1

        self.blank_id = self.tokenizer(blank_text)["input_ids"][0]

        assert isinstance(self.eot_id, int)

    def text_to_ids(self, text: str) -> List[int]:
        # TODO: encode or tokenize + convert_tokens_to_ids?
        return self.tokenizer.encode(text)
