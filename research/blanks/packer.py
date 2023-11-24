from typing import Callable, Optional

from .data import BlanxExample
from lizrd.text.datasets import AbstractDataset
from lizrd.text.packers import GPTPacker
from lizrd.text.tokenizers import AbstractTokenizer
from research.blanks.utils import (
    can_fit_blanks,
    get_last_point_to_fit_blanks,
    insert_blanks_input,
    insert_blanks_target,
    make_blanks_attention_mask,
    make_blanks_loss_mask,
)


class BlankPacker(GPTPacker):
    def __init__(
        self,
        sequence_length: int,
        dataset: AbstractDataset,
        tokenizer_maker: Callable[[], AbstractTokenizer],
        n_blanks: int,
        seed: Optional[int] = None,
        use_only_last_blank_loss: bool = False,
    ):
        super().__init__(
            sequence_length,
            dataset,
            tokenizer_maker,
            seed=seed,
        )

        self.n_blanks = n_blanks
        self.use_only_last_blank_loss = use_only_last_blank_loss

    def get_sample(self) -> BlanxExample:
        blank_id = self.tokenizer.blank_id
        assert blank_id is not None

        sample = super().get_sample()

        input_tokens = sample.input_ids
        target_tokens = sample.target_ids
        seq_len = len(input_tokens)

        blank_insertion_point = self.py_rng.randint(
            1, get_last_point_to_fit_blanks(seq_len, self.n_blanks)
        )
        input_tokens = insert_blanks_input(
            input_tokens, blank_id, blank_insertion_point, self.n_blanks
        )
        target_tokens = insert_blanks_target(
            target_tokens, blank_insertion_point, self.n_blanks
        )
        if self.use_only_last_blank_loss:
            should_calculate_loss = make_blanks_loss_mask(
                seq_len, blank_insertion_point, self.n_blanks
            )
            assert len(should_calculate_loss) == seq_len
        else:
            should_calculate_loss = sample.should_calculate_loss
            assert should_calculate_loss == [1] * seq_len

        att_mask = make_blanks_attention_mask(
            seq_len, blank_insertion_point, self.n_blanks
        )

        return BlanxExample(
            input_tokens, target_tokens, should_calculate_loss, att_mask
        )


class BlankEvalPacker(GPTPacker):
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

    def get_sample(self) -> BlanxExample:
        blank_id = self.tokenizer.blank_id
        assert blank_id is not None

        sample = super().get_sample()

        input_tokens = sample.input_ids
        target_tokens = sample.target_ids
        seq_len = len(input_tokens)

        blank_insertion_point = self.py_rng.randint(1, seq_len - 1)
        if can_fit_blanks(seq_len, blank_insertion_point, self.n_blanks):
            input_tokens = insert_blanks_input(
                input_tokens, blank_id, blank_insertion_point, self.n_blanks
            )
            target_tokens = insert_blanks_target(
                target_tokens, blank_insertion_point, self.n_blanks
            )
            should_calculate_loss = [0] * seq_len
            should_calculate_loss[blank_insertion_point + self.n_blanks - 1] = 1
            att_mask = make_blanks_attention_mask(
                seq_len, blank_insertion_point, self.n_blanks
            )
        else:
            should_calculate_loss = [0] * seq_len
            should_calculate_loss[blank_insertion_point - 1] = 1
            att_mask = make_blanks_attention_mask(seq_len, blank_insertion_point, 0)

        return BlanxExample(
            input_tokens, target_tokens, should_calculate_loss, att_mask
        )
