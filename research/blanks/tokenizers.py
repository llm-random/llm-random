from typing import List

from transformers import GPT2TokenizerFast

from source.text.tokenizers import AbstractTokenizer, disable_tokenizer_warnings

NUM_RESERVED_TOKENS = 100
MAX_BLANKS = 50


class BlankTokenizer(AbstractTokenizer):
    VOCAB_SIZE = 50257 + NUM_RESERVED_TOKENS
    BLANK_IDS = list(range(50257, 50257 + MAX_BLANKS))

    def add_tokens(self, tokens: List[str]):
        self.tokenizer.add_tokens(tokens)
        self._current_vocab_size += len(tokens)

    def __init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self._current_vocab_size = self.tokenizer.vocab_size
        disable_tokenizer_warnings(self.tokenizer)

        self.eot_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        assert isinstance(self.eot_id, int)

        blank_token_template = "<|blank_{blank_id}|>"
        blank_tokens = [
            blank_token_template.format(blank_id=i) for i in range(MAX_BLANKS)
        ]
        self.add_tokens(blank_tokens)
        self.blank_ids = [
            self.tokenizer(blank_token_text)["input_ids"][0]
            for blank_token_text in blank_tokens
        ]

        assert self.blank_ids == BlankTokenizer.BLANK_IDS

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
