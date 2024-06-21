import unittest
import torch

from research.token_reduction.layers import TokenMergingLayer, choose_indeces_to_reduce


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

        self.dm = 8

class TestTokenMerging(unittest.TestCase):

    def setUp(self):
        self.batch_size = 2
        self.seq_len = 7
        self.result_seq_len = 4
        self.n_tokens_to_reduce = 2

        self.dm = 8
        self.input = torch.randn(self.batch_size, self.seq_len, self.dm)

    def test_output_shape(self):
        merging_layer = TokenMergingLayer(
            self.result_seq_len, self.dm, self.n_tokens_to_reduce
        )
        output = merging_layer(self.input)
        self.assertEqual(output.shape, (self.batch_size, self.result_seq_len, self.dm))

    def test_merged_token_exists(self):
        merging_layer = TokenMergingLayer(
            self.result_seq_len, self.dm, self.n_tokens_to_reduce
        )
        input_copy = self.input.clone()

        reduced_index = None
        while reduced_index is None:
            output = merging_layer(self.input)

            for index in merging_layer.indices_to_reduce:
                if index + 1 in merging_layer.indices_to_keep:
                    reduced_index = index
                    break

        unchanged_input = input_copy.view(-1, self.dm)
        reduced_token = unchanged_input[reduced_index]
        transformed_reduced_token = merging_layer.merge_linear_projection(reduced_token)
        merged_token = unchanged_input[reduced_index + 1] + transformed_reduced_token

        self.assertTrue(merged_token in output.view(-1, self.dm))


