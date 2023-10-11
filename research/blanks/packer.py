from typing import Callable, List, Optional


from lizrd.text.data import LLMExample as LLMExample
from lizrd.text.datasets import AbstractDataset
from lizrd.text.packers import AbstractPacker, take_circular
from lizrd.text.tokenizers import AbstractTokenizer


class BlankPacker(AbstractPacker):
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
        assert n_blanks > 0

    def get_sample(self) -> LLMExample:
        """
        Sample examples from the dataset until we reach the desired sequence length.
        """
        eot_id = self.tokenizer.eot_id
        blank_id = self.tokenizer.blank_id
        assert eot_id is not None
        assert blank_id is not None

        buffer_input: List[int] = []
        buffer_output: List[int] = []
        blank_mask_buffer: List[int] = []
        calculate_loss: List[int] = []
        document_lengths: List[int] = []

        # TODO: change it so the blank can come first in both example and sequence
        while True:
            document = self.dataset.get_document()
            tokens = self.tokenizer.text_to_ids(document)
            tokens.append(eot_id)

            blank_insertion_point = self.py_rng.randint(0, len(tokens) - 1)
            input_tokens = (
                tokens[:blank_insertion_point]
                + [blank_id] * self.n_blanks
                + tokens[blank_insertion_point:]
            )
            output_tokens = (
                tokens[:blank_insertion_point]
                + [tokens[blank_insertion_point]] * self.n_blanks
                + tokens[blank_insertion_point:]
            )
            blank_mask = (
                [0] * blank_insertion_point
                + [1] * self.n_blanks
                + [0] * (len(tokens) - blank_insertion_point)
            )

            buffer_input.extend(input_tokens)
            buffer_output.extend(output_tokens)
            blank_mask_buffer.extend(blank_mask)

            document_lengths.append(len(tokens))
            if (sum(document_lengths) - max(document_lengths)) > self.sequence_length:
                break

        assert len(buffer_input) == len(buffer_output) == len(blank_mask_buffer)

        illegal_blanks_position = True
        while illegal_blanks_position:
            sample_start = self.py_rng.randint(0, len(buffer_input) - 1)
            sample_end = sample_start + self.sequence_length
            blanks_at_beginning = blank_mask_buffer[sample_start] == 1

            blanks_at_end = (
                blank_mask_buffer[(sample_end - 1) % len(blank_mask_buffer)] == 1
                and sum(
                    take_circular(
                        blank_mask_buffer, sample_end - self.n_blanks, sample_end
                    )
                )
                != self.n_blanks
            )

            illegal_blanks_position = blanks_at_beginning or blanks_at_end

        input_ids = list(take_circular(buffer_input, sample_start, sample_end))
        target_ids = list(
            take_circular(buffer_output, sample_start + 1, sample_end + 1)
        )
        calculate_loss = [1] * len(target_ids)

        return LLMExample(input_ids, target_ids, calculate_loss)
