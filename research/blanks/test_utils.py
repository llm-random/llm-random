import torch

from lizrd.support.test_utils import GeneralTestCase
from research.blanks.utils import (
    can_fit_blanks,
    get_last_blanks_in_series,
    get_last_point_to_fit_blanks,
    insert_blanks_input,
    insert_blanks_target,
    shift_left,
    shift_right,
    get_first_blanks_in_series,
    get_preblanks,
    get_last_blanks_in_series,
    make_blanks_attention_mask,
    make_blanks_fixed_positions,
)


class TestUtils(GeneralTestCase):
    def test_shift_left_2d(self):
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        expected = torch.tensor([[2, 3, 0], [5, 6, 0]])
        result = shift_left(x)
        self.assertTrue(torch.equal(result, expected))

    def test_shift_right_2d(self):
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        expected = torch.tensor([[0, 1, 2], [0, 4, 5]])
        result = shift_right(x)
        self.assertTrue(torch.equal(result, expected))

    def test_shift_left_3d(self):
        x = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        expected = torch.tensor([[[4, 5, 6], [0, 0, 0]], [[10, 11, 12], [0, 0, 0]]])
        result = shift_left(x)
        self.assertTrue(torch.equal(result, expected))

    def test_shift_right_3d(self):
        x = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        expected = torch.tensor([[[0, 0, 0], [1, 2, 3]], [[0, 0, 0], [7, 8, 9]]])
        result = shift_right(x)
        self.assertTrue(torch.equal(result, expected))

    def test_get_first_blanks_in_series(self):
        is_blank = torch.tensor(
            [[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]]
        )
        expected = torch.tensor(
            [[0, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]]
        )
        result = get_first_blanks_in_series(is_blank)
        self.assertTrue(torch.equal(result, expected))

    def test_get_last_blanks_in_series(self):
        is_blank = torch.tensor(
            [[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]]
        )
        expected = torch.tensor(
            [[0, 0, 1, 0], [1, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
        )
        result = get_last_blanks_in_series(is_blank)
        self.assertTrue(torch.equal(result, expected))

    def test_get_preblanks(self):
        is_blank = torch.tensor([[0, 1, 1, 0], [1, 0, 1, 1]])
        expected = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]])
        result = get_preblanks(is_blank)
        self.assertTrue(torch.equal(result, expected))

    def test_get_last_blanks_in_series(self):
        is_blank = torch.tensor([[0, 1, 1, 0], [1, 0, 1, 1]])
        expected = torch.tensor([[0, 0, 1, 0], [1, 0, 0, 1]])
        result = get_last_blanks_in_series(is_blank)
        self.assertTrue(torch.equal(result, expected))

    def test_make_blanks_attention_mask(self):
        is_blank = torch.tensor([[0, 1, 1, 0], [1, 0, 1, 0]])
        expected = torch.tensor(
            [
                [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 1, 1, 0]],
                [[0, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 1], [1, 0, 1, 0]],
            ]
        ).bool()
        result = make_blanks_attention_mask(is_blank)
        self.assertTrue(torch.equal(result, expected))

    def test_insert_blanks_input(self):
        input_sequence = [1, 2, 3, 4, 5]
        blank_id = 0
        blank_insertion_point = 2
        n_blanks = 2
        expected = [1, 2, 0, 0, 3]
        result = insert_blanks_input(
            input_sequence, blank_id, blank_insertion_point, n_blanks
        )
        self.assertEqual(result, expected)

    def test_insert_blanks_target(self):
        target_sequence = [2, 3, 4, 5, 6]
        blank_insertion_point = 2
        n_blanks = 2
        expected = [2, 3, 3, 3, 4]
        result = insert_blanks_target(target_sequence, blank_insertion_point, n_blanks)
        self.assertEqual(result, expected)

    def test_get_last_point_to_fit_blanks(self):
        sequence_length = 5
        n_blanks = 2
        expected = 3
        result = get_last_point_to_fit_blanks(sequence_length, n_blanks)
        self.assertEqual(result, expected)

    def test_can_fit_blanks(self):
        sequence_length = 5
        blank_insertion_point = 2
        n_blanks = 2
        self.assertTrue(
            can_fit_blanks(sequence_length, blank_insertion_point, n_blanks)
        )

        sequence_length = 5
        blank_insertion_point = 4
        n_blanks = 2
        self.assertFalse(
            can_fit_blanks(sequence_length, blank_insertion_point, n_blanks)
        )

    def test_make_blanks_fixed_positions(self):
        tokens = torch.tensor([[0, 0, 1, 1, 0], [0, 1, 1, 0, 0], [1, 1, 0, 1, 1]])
        expected = torch.tensor([[0, 1, 2, 3, 2], [0, 1, 2, 1, 2], [0, 1, 0, 1, 2]])
        result = make_blanks_fixed_positions(tokens, 1, n_blanks_block=2)
        self.assertTrue(torch.equal(result, expected))

    def test_make_blanks_fixed_positions_throws(self):
        tokens = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 0]])
        self.assertRaises(
            ValueError, make_blanks_fixed_positions, tokens, 1, n_blanks_block=2
        )
