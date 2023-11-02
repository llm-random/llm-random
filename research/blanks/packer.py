from typing import Callable, Optional

from lizrd.text.data import LLMExample
from lizrd.text.datasets import AbstractDataset
from lizrd.text.packers import GPTPacker
from lizrd.text.tokenizers import AbstractTokenizer


class BlankPacker(GPTPacker):
    def __init__(
        self,
        sequence_length: int,
        dataset: AbstractDataset,
        tokenizer_maker: Callable[[], AbstractTokenizer],
        n_blanks: int,
        seed: Optional[int] = None,
    ):
        super().__init__(
            sequence_length,
            dataset,
            tokenizer_maker,
            seed=seed,
        )

        self.n_blanks = n_blanks

    def get_sample(self) -> LLMExample:
        blank_id = self.tokenizer.blank_id
        assert blank_id is not None

        sample = super().get_sample()

        input_tokens = sample.input_ids
        target_tokens = sample.target_ids
        seq_len = len(input_tokens)

        blank_insertion_point = self.py_rng.randint(1, seq_len - 1 - self.n_blanks + 1)
        input_tokens = (
            input_tokens[:blank_insertion_point]
            + [blank_id] * self.n_blanks
            + input_tokens[blank_insertion_point:]
        )[:seq_len]
        target_tokens = (
            target_tokens[:blank_insertion_point]
            + [target_tokens[blank_insertion_point - 1]] * self.n_blanks
            + target_tokens[blank_insertion_point:]
        )[:seq_len]
        assert sample.should_calculate_loss == [1] * seq_len

        return LLMExample(input_tokens, target_tokens, sample.should_calculate_loss)
