import torch

import research.conditional
import research.conditional.ffs
from lizrd.core import bert
from lizrd.support.test_utils import GeneralTestCase


class TestBatchedFeedForward(GeneralTestCase):
    def test_basic(self):
        batch, dm = 4, 32
        sets = 5
        experts = 7
        seql = experts * 3
        expertsize = 11
        dff = sets * experts * expertsize
        layer = research.conditional.ffs.BatchSplitFF(
            [], dm, dff, sets, experts, expertsize
        )
        input = torch.normal(0.0, 1.0, (batch, seql, dm))
        output = layer(input)
        self.assertShape(output, (batch, seql, dm))


class FactoredDenseTest(GeneralTestCase):
    def test_basic(self):
        batch, dinp, dout = 4, 32, 64
        layer = research.conditional.ffs.FactoredDense(dinp, dout, modules=4)
        input = torch.normal(0.0, 1.0, (batch, dinp))
        output = layer(input)
        self.assertShape(output, (batch, dout))

    def test_more_dims(self):
        batch, seql, dinp, dout = 4, 8, 32, 64
        layer = research.conditional.ffs.FactoredDense(dinp, dout, modules=2)
        input = torch.normal(0.0, 1.0, (batch, seql, dinp))
        output = layer(input)
        self.assertShape(output, (batch, seql, dout))


class TestGeneralizedReLU(GeneralTestCase):
    def test_basic(self):
        batch, dinp = 4, 32
        bias = False
        layer = research.conditional.ffs.GeneralizedReLU(dinp, bias)
        input = torch.normal(0.0, 1.0, (batch, dinp))
        output = layer(input)
        self.assertShape(output, (batch, dinp))


class BERTSparseTest(GeneralTestCase):
    def test_basic(self):
        batch, seql, dm, heads, dff = 3, 12, 32, 4, 64
        modules = 4

        vocab_size, max_length = 107, 33
        output_size = 3
        n_blocks = 2

        embedding_layer = bert.EmbeddingLayer(
            bert.PositionalEmbedding(max_length, dm),
            bert.TokenEmbedding(vocab_size, dm),
        )

        factored_dense_fun = lambda: research.conditional.ffs.FactoredDense(
            dm, dm, modules
        )

        encoder_tower = bert.EncoderTower(
            n_blocks,
            dm,
            (lambda: research.conditional.ffs.BatchSplitFF([], dm, dff, 4, 4, 4)),
            (lambda: bert.Attention(dm, heads, layer_fun=factored_dense_fun)),
        )

        head = bert.PredictionHead(dm, output_size)

        model = bert.BERT(embedding_layer, encoder_tower, head)

        input = torch.randint(0, vocab_size, (batch, seql))

        output = model(input)

        self.assertShape(output, (batch, seql, output_size))


class BERTSparseGradientTest(GeneralTestCase):
    def test_basic(self):
        batch, seql, dm, heads, dff = 4, 16, 32, 4, 64
        modules = 4

        vocab_size, max_length = 107, 33
        output_size = 3
        n_blocks = 1

        embedding_layer = bert.EmbeddingLayer(
            bert.PositionalEmbedding(max_length, dm),
            bert.TokenEmbedding(vocab_size, dm),
        )

        factored_dense_fun = lambda: research.conditional.ffs.FactoredDense(
            dm, dm, modules
        )

        encoder_tower = bert.EncoderTower(
            n_blocks,
            dm,
            (lambda: research.conditional.ffs.BatchSplitFF([], dm, dff, 4, 4, 4)),
            (lambda: bert.Attention(dm, heads, layer_fun=factored_dense_fun)),
        )

        head = bert.PredictionHead(dm, output_size)

        model = bert.BERT(embedding_layer, encoder_tower, head)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        input = torch.randint(0, vocab_size, (batch, seql))

        output = model(input)

        loss = torch.mean(output[:, 0] ** 2)  # TODO: add better loss function

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.assertShape(output, (batch, seql, output_size))
        self.assertShape(loss, ())
