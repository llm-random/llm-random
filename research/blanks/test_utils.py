import torch

from lizrd.support.test_utils import GeneralTestCase
from research.blanks.utils import (
    shift_left,
    shift_right,
    get_first_blanks_in_series,
    get_preblanks,
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
        is_blank = torch.tensor([[0, 1, 1, 0], [1, 0, 1, 1]])
        expected = torch.tensor([[0, 1, 0, 0], [1, 0, 1, 0]])
        result = get_first_blanks_in_series(is_blank)
        self.assertTrue(torch.equal(result, expected))

    def test_get_preblanks(self):
        is_blank = torch.tensor([[0, 1, 1, 0], [1, 0, 1, 1]])
        expected = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]])
        result = get_preblanks(is_blank)
        self.assertTrue(torch.equal(result, expected))
