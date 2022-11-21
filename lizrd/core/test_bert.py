import torch

from lizrd.core import bert
import unittest

from lizrd.support.test_utils import GeneralTestCase


class TestFeedForward(GeneralTestCase):
    def test_basic(self):
        batch, seql, dm, dff = 4, 8, 32, 64
        layer = bert.FeedForward(dm, dff)
        input = torch.normal(0.0, 1.0, (batch, seql, dm))
        output = layer(input)
        self.assertShape(output, (batch, seql, dm))


class ResidualTest(GeneralTestCase):
    def test_basic(self):
        batch, seql, dm, dff = 4, 8, 32, 64
        layer_ff = bert.FeedForward(dm, dff)
        layer_residual = bert.Residual(layer_ff)
        input = torch.normal(0.0, 1.0, (batch, seql, dm))
        output1 = layer_ff(input)
        output2 = layer_residual(input)
        self.assertShape(output2, (batch, seql, dm))
        self.assertTensorAlmostEqual(output1, output2-input)
        self.assertTensorEqual(output1+input, output2)


class AttentionTest(GeneralTestCase):
    def test_basic(self):
        batch, seql, dm, heads = 3, 7, 32, 4
        layer = bert.Attention(dm, heads)
        input = torch.normal(0.0, 1.0, (batch, seql, dm))
        out = layer(input)
        self.assertShape(out, (batch, seql, dm))

    def test_nonstandard_dhead(self):
        batch, seql, dm, heads, dhead = 3, 7, 32, 4, 100
        layer = bert.Attention(dm, heads, dhead=dhead)
        input = torch.normal(0.0, 1.0, (batch, seql, dm))
        out = layer(input)
        self.assertShape(out, (batch, seql, dm))

    def test_residual(self):
        batch, seql, dm, heads = 3, 7, 32, 4
        layer = bert.Residual(bert.Attention(dm, heads))
        input = torch.normal(0.0, 1.0, (batch, seql, dm))
        out = layer(input)
        self.assertShape(out, (batch, seql, dm))


class EncoderTowerTest(GeneralTestCase):
    def test_basic(self):
        batch, seql, dm, heads, dff = 3, 7, 32, 4, 64
        nblocks = 3
        model = bert.EncoderTower(
            nblocks, dm,
            (lambda: bert.Attention(dm, heads)),
            (lambda: bert.FeedForward(dm, dff))
        )
        input = torch.normal(0.0, 1.0, (batch, seql, dm))
        out = model(input)
        self.assertShape(out, (batch, seql, dm))


class BERTTest(GeneralTestCase):
    def test_basic(self):
        batch, seql, dm, heads, dff = 3, 7, 32, 4, 64
        vocab_size, max_length = 107, 33
        output_size = 3
        n_blocks = 2

        embedding_layer = bert.EmbeddingLayer(
            bert.PositionalEmbedding(max_length, dm),
            bert.TokenEmbedding(vocab_size, dm)
        )

        encoder_tower = bert.EncoderTower(
            n_blocks,
            dm,
            (lambda: bert.FeedForward(dm, dff)),
            (lambda: bert.Attention(dm, heads)),
        )

        head = bert.PredictionHead(dm, output_size)

        model = bert.BERT(embedding_layer, encoder_tower, head)

        input = torch.randint(0, vocab_size, (batch, seql))

        output = model(input)

        self.assertShape(output, (batch, seql, output_size))


if __name__ == '__main__':
    unittest.main()
