import torch

import research.conditional.moe_layers.ffs
from source.core import llm
from source.support.test_utils import GeneralTestCase, skip_test


class TestBatchedFeedForward(GeneralTestCase):
    def test_basic(self):
        batch, dm = 4, 32
        sets = 5
        experts = 7
        seql = experts * 3
        expertsize = 11
        dff = sets * experts * expertsize
        layer = research.conditional.moe_layers.ffs.BatchSplitFF(
            register_list=[],
            dm=dm,
            dff=dff,
            expertsets=sets,
            nexperts=experts,
            expertsize=expertsize,
            init_type="kaiming_uniform",
            init_scale=1.0,
        )
        input = torch.normal(0.0, 1.0, (batch, seql, dm))
        output = layer(input)
        self.assertShape(output, (batch, seql, dm))


class FactoredDenseTest(GeneralTestCase):
    def test_basic(self):
        batch, dinp, dout = 4, 32, 64
        layer = research.conditional.moe_layers.ffs.FactoredDense(dinp, dout, modules=4)
        input = torch.normal(0.0, 1.0, (batch, dinp))
        output = layer(input)
        self.assertShape(output, (batch, dout))

    def test_more_dims(self):
        batch, seql, dinp, dout = 4, 8, 32, 64
        layer = research.conditional.moe_layers.ffs.FactoredDense(dinp, dout, modules=2)
        input = torch.normal(0.0, 1.0, (batch, seql, dinp))
        output = layer(input)
        self.assertShape(output, (batch, seql, dout))


class TestGeneralizedReLU(GeneralTestCase):
    def test_basic(self):
        batch, dinp = 4, 32
        bias = False
        layer = research.conditional.moe_layers.ffs.GeneralizedReLU(dinp, bias)
        input = torch.normal(0.0, 1.0, (batch, dinp))
        output = layer(input)
        self.assertShape(output, (batch, dinp))


class BERTSparseTest(GeneralTestCase):
    @skip_test(reason="Attention implementation changed")
    def test_basic(self):
        batch, seql, dm, heads, dff = 3, 12, 32, 4, 64
        modules = 4

        vocab_size, max_length = 107, 33
        output_size = 3
        n_blocks = 2

        embedding_layer = llm.EmbeddingLayer(
            llm.PositionalEmbedding(max_length, dm),
            llm.TokenEmbedding(vocab_size, dm),
        )

        factored_dense_fun = lambda: research.conditional.moe_layers.ffs.FactoredDense(
            dm, dm, modules
        )

        encoder_tower = llm.TransformerTower(
            n_blocks,
            dm,
            (
                lambda: research.conditional.moe_layers.ffs.BatchSplitFF(
                    [], dm, dff, 4, 4, 4
                )
            ),
            (lambda: llm.Attention(dm, heads, layer_fun=factored_dense_fun)),
        )

        head = llm.PredictionHead(dm, output_size)

        model = llm.LLM(embedding_layer, encoder_tower, head)

        input = torch.randint(0, vocab_size, (batch, seql))

        output = model(input)

        self.assertShape(output, (batch, seql, output_size))


class BERTSparseGradientTest(GeneralTestCase):
    @skip_test(reason="Attention implementation changed")
    def test_basic(self):
        batch, seql, dm, heads, dff = 4, 16, 32, 4, 64
        modules = 4

        vocab_size, max_length = 107, 33
        output_size = 3
        n_blocks = 1

        embedding_layer = llm.EmbeddingLayer(
            llm.PositionalEmbedding(max_length, dm),
            llm.TokenEmbedding(vocab_size, dm),
        )

        factored_dense_fun = lambda: research.conditional.moe_layers.ffs.FactoredDense(
            dm, dm, modules
        )

        encoder_tower = llm.TransformerTower(
            n_blocks,
            dm,
            (
                lambda: research.conditional.moe_layers.ffs.BatchSplitFF(
                    [], dm, dff, 4, 4, 4
                )
            ),
            (lambda: llm.Attention(dm, heads, layer_fun=factored_dense_fun)),
        )

        head = llm.PredictionHead(dm, output_size)

        model = llm.LLM(embedding_layer, encoder_tower, head)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        input = torch.randint(0, vocab_size, (batch, seql))

        output = model(input)

        loss = torch.mean(output[:, 0] ** 2)  # TODO: add better loss function

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.assertShape(output, (batch, seql, output_size))
        self.assertShape(loss, ())
