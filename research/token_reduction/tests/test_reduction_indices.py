import unittest
import torch

from research.token_reduction.layers import (
    choose_indeces_to_reduce,
)


class TestRandomIndicesOutside(unittest.TestCase):
    def test_basic_case(self):
        batch_size = 2
        seq_len = 7
        result_seq_len = 5
        n_tokens_to_reduce = 2
        indices_to_keep, indices_to_reduct = choose_indeces_to_reduce(
            batch_size, seq_len, result_seq_len, n_tokens_to_reduce
        )
        self.assertEqual(indices_to_keep.numel(), batch_size * result_seq_len)
        self.assertEqual(indices_to_reduct.numel(), batch_size * n_tokens_to_reduce)

    def test_last_token_undroppable(self):
        batch_size = 1
        seq_len = 10
        result_seq_len = 9
        n_tokens_to_reduce = 1
        indices_to_keep, _ = choose_indeces_to_reduce(
            batch_size, seq_len, result_seq_len, n_tokens_to_reduce
        )
        self.assertIn(
            seq_len - 1,
            indices_to_keep.tolist(),
            "Last token index should be in indices_to_keep",
        )

    def test_assertion_error(self):
        batch_size = 1
        seq_len = 5
        result_seq_len = 3
        n_tokens_to_reduce = 3
        with self.assertRaises(AssertionError, msg="Too many tokens to reduce"):
            _ = choose_indeces_to_reduce(
                batch_size, seq_len, result_seq_len, n_tokens_to_reduce
            )

    def test_indices_range(self):
        batch_size = 2
        seq_len = 8
        result_seq_len = 5
        n_tokens_to_reduce = 3
        indices_to_keep, indices_to_reduct = choose_indeces_to_reduce(
            batch_size, seq_len, result_seq_len, n_tokens_to_reduce
        )
        self.assertTrue(torch.all(indices_to_keep < (batch_size * seq_len)))
        self.assertTrue(torch.all(indices_to_reduct < (batch_size * seq_len)))

    def test_all_indices_managed(self):
        batch_size = 3
        seq_len = 6
        result_seq_len = 4
        n_tokens_to_reduce = 2
        indices_to_keep, indices_to_reduct = choose_indeces_to_reduce(
            batch_size, seq_len, result_seq_len, n_tokens_to_reduce
        )
        total_tokens = batch_size * seq_len
        unique_indices = set(indices_to_keep.tolist() + indices_to_reduct.tolist())
        self.assertEqual(
            len(unique_indices),
            total_tokens,
            "All indices should be managed exactly once",
        )

    def test_permutation_property(self):
        batch_size = 2
        seq_len = 9
        result_seq_len = 6
        n_tokens_to_reduce = 2
        indices_to_keep, indices_to_reduct = choose_indeces_to_reduce(
            batch_size, seq_len, result_seq_len, n_tokens_to_reduce
        )
        for i in range(batch_size):
            expected_indices = set(
                range(
                    i * seq_len,
                    i * seq_len + (result_seq_len + n_tokens_to_reduce),
                )
            )
            managed_indices = set(
                indices_to_keep[i * result_seq_len : (i + 1) * result_seq_len].tolist()
                + indices_to_reduct[
                    i * n_tokens_to_reduce : (i + 1) * n_tokens_to_reduce
                ].tolist()
            )
            self.assertEqual(
                expected_indices,
                managed_indices,
                "All indices within a batch should be managed correctly",
            )
