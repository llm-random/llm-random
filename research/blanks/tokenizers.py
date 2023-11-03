from typing import List

from transformers import GPT2TokenizerFast

from lizrd.text.tokenizers import AbstractTokenizer, disable_tokenizer_warnings

NUM_RESERVED_TOKENS = 100


class BlankTokenizer(AbstractTokenizer):
    VOCAB_SIZE = 50257 + NUM_RESERVED_TOKENS

    def add_tokens(self, tokens: List[str]):
        self.tokenizer.add_tokens(tokens)
        self._current_vocab_size += len(tokens)

    def __init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self._current_vocab_size = self.tokenizer.vocab_size
        disable_tokenizer_warnings(self.tokenizer)

        self.eot_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        assert isinstance(self.eot_id, int)

        blank_text = "<|blank|>"
        self.add_tokens([blank_text])
        self.blank_id = self.tokenizer(blank_text)["input_ids"][0]

        self.tokenizer.add_tokens(
            [
                f"<|extra_token_{i}|>"
                for i in range(self.VOCAB_SIZE - self._current_vocab_size)
            ]
        )

        assert isinstance(self.eot_id, int)

    def text_to_ids(self, text: str) -> List[int]:
        # TODO: encode or tokenize + convert_tokens_to_ids?
        return self.tokenizer.encode(text)
