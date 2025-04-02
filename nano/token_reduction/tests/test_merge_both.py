import unittest
import torch


from model import EmbeddingLayer, PositionalEmbedding, TokenEmbedding
from token_reduction.model import TokenMergingEmbeddingBothTokens


class CommonParams:
    vocab_size = 12
    dmodel = 5
    sequence_length = 10
    dropped_tokens = 1
    init_type = "truncated_normal"  # Example init type
    init_scale = 0.02


common = CommonParams()


class TestTokenMergingEmbeddingBothTokens(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

        token_embedding = TokenEmbedding(
            common.vocab_size,
            common.dmodel,
            init_type=common.init_type,
            init_scale=common.init_scale,
        )
        positional_embedding = PositionalEmbedding(
            common.sequence_length + common.dropped_tokens,
            common.dmodel,
            init_type=common.init_type,
            init_scale=common.init_scale,
        )

        self.normal_embedding = EmbeddingLayer(token_embedding, positional_embedding)

        self.model = TokenMergingEmbeddingBothTokens(
            normal_embedding=self.normal_embedding,
            dmodel=common.dmodel,
            init_type=common.init_type,
            init_scale=common.init_scale,
        )

    def test_embedding_both_tokens_training(self):
        self.model.train()
        batch_size = 4
        seq_length = common.sequence_length
        x = torch.randint(0, common.vocab_size, (batch_size, seq_length))

        keep_indexes = torch.tensor(
            [[0, 2, 4, 6, 8], [0, 2, 4, 6, 8], [0, 2, 4, 6, 8], [0, 2, 4, 6, 8]],
            dtype=torch.long,
        )
        merge_indexes = torch.tensor(
            [[1, 3, 5, 7], [1, 3, 5, 7], [1, 3, 5, 7], [1, 3, 5, 7]], dtype=torch.long
        )

        with torch.no_grad():
            output = self.model(x, keep_indexes, merge_indexes)

        expected_shape = (batch_size, keep_indexes.shape[1], common.dmodel)

        self.assertEqual(
            output.shape,
            expected_shape,
            f"Expected output shape {expected_shape}, but got {output.shape}",
        )

    def test_embedding_both_tokens_eval(self):
        self.model.eval()
        batch_size = 4
        seq_length = common.sequence_length
        x = torch.randint(0, common.vocab_size, (batch_size, seq_length))

        keep_indexes = torch.tensor(
            [[0, 2, 4, 6, 8], [0, 2, 4, 6, 8], [0, 2, 4, 6, 8], [0, 2, 4, 6, 8]],
            dtype=torch.long,
        )
        merge_indexes = torch.tensor(
            [[1, 3, 5, 7], [1, 3, 5, 7], [1, 3, 5, 7], [1, 3, 5, 7]], dtype=torch.long
        )

        with torch.no_grad():
            output = self.model(x, keep_indexes, merge_indexes)

        expected_shape = (batch_size, seq_length, common.dmodel)
        self.assertEqual(
            output.shape,
            expected_shape,
            f"Expected output shape {expected_shape}, but got {output.shape}",
        )


if __name__ == "__main__":
    unittest.main()
