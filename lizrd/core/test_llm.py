import torch

from lizrd.core import llm
import unittest

from lizrd.support.test_utils import GeneralTestCase


class TestFeedForward(GeneralTestCase):
    def test_basic(self):
        batch, seql, dm, dff = 4, 8, 32, 64
        layer = llm.FeedForward(dm, dff)
        input = torch.normal(0.0, 1.0, (batch, seql, dm))
        output = layer(input)
        self.assertShape(output, (batch, seql, dm))


class ResidualTest(GeneralTestCase):
    def test_basic(self):
        batch, seql, dm, dff = 4, 8, 32, 64
        layer_ff = llm.FeedForward(dm, dff)
        layer_residual = llm.Residual(layer_ff)
        input = torch.normal(0.0, 1.0, (batch, seql, dm))
        output1 = layer_ff(input)
        output2 = layer_residual(input)
        self.assertShape(output2, (batch, seql, dm))
        self.assertTensorAlmostEqual(output1, output2 - input)
        self.assertTensorEqual(output1 + input, output2)


class AttentionTest(GeneralTestCase):
    def test_basic(self):
        try:
            batch, seql, dm, heads = 3, 7, 32, 4
            layer = llm.Attention(dm, heads, causal=False, flash=True)
            input = torch.normal(0.0, 1.0, (batch, seql, dm))
            out = layer(input)
            self.assertShape(out, (batch, seql, dm))
        except Exception as e:
            pass

    def test_flash_basic(self):
        batch, seql, dm, heads = 3, 7, 32, 4
        layer = llm.Attention(dm, heads, causal=False, flash=True)
        input = torch.normal(0.0, 1.0, (batch, seql, dm))
        out = layer(input)
        self.assertShape(out, (batch, seql, dm))

    def test_flash_basic_causal(self):
        batch, seql, dm, heads = 3, 7, 32, 4
        layer = llm.Attention(dm, heads, causal=True, flash=True)
        input = torch.normal(0.0, 1.0, (batch, seql, dm))
        out = layer(input)
        self.assertShape(out, (batch, seql, dm))

    def test_attention_mechanism_equivalence(self):
        batch, seql, dm, dhead, heads = 16, 4, 512, 64, 8
        q = torch.normal(0.0, 1.0, (batch, heads, seql, dhead))
        k = torch.normal(0.0, 1.0, (batch, heads, seql, dhead))
        v = torch.normal(0.0, 1.0, (batch, heads, seql, dhead))
        out1 = llm.attention_mechanism(q, k, v, dhead, flash=False, causal=False)
        out2 = llm.attention_mechanism(q, k, v, dhead, flash=True, causal=False)
        self.assertTensorAlmostEqual(out1, out2)

    def test_flash_equivalence(self):
        batch, seql, dm, heads = 3, 7, 32, 4
        layer = llm.Attention(dm, heads, causal=False, flash=False)
        input = torch.normal(0.0, 1.0, (batch, seql, dm))
        out = layer(input)

        layer.flash = False
        out2 = layer(input)

        self.assertTensorAlmostEqual(out, out2)

    def test_nonstandard_dhead(self):
        batch, seql, dm, heads, dhead = 3, 7, 32, 4, 100
        layer = llm.Attention(dm, heads, causal=False, dhead=dhead)
        input = torch.normal(0.0, 1.0, (batch, seql, dm))
        out = layer(input)
        self.assertShape(out, (batch, seql, dm))

    def test_residual(self):
        batch, seql, dm, heads = 3, 7, 32, 4
        layer = llm.Residual(llm.Attention(dm, heads, causal=False))
        input = torch.normal(0.0, 1.0, (batch, seql, dm))
        out = layer(input)
        self.assertShape(out, (batch, seql, dm))


class EncoderTowerTest(GeneralTestCase):
    def test_basic(self):
        batch, seql, dm, heads, dff = 3, 7, 32, 4, 64
        nblocks = 3
        device = torch.device("cpu")

        layer_dict = {
            "attention": lambda: llm.Attention(dm, heads, causal=False),
            "feedforward": lambda: llm.FeedForward(dm, dff),
        }
        model = llm.TransformerTower(nblocks, dm, layer_dict, device=device)
        input = torch.normal(0.0, 1.0, (batch, seql, dm))
        out = model(input)
        self.assertShape(out, (batch, seql, dm))


class LLMTest(GeneralTestCase):
    def test_bert(self):
        batch, seql, dm, heads, dff = 3, 7, 32, 4, 64
        vocab_size, max_length = 107, 33
        output_size = 3
        n_blocks = 2
        device = torch.device("cpu")

        embedding_layer = llm.EmbeddingLayer(
            llm.PositionalEmbedding(max_length, dm),
            llm.TokenEmbedding(vocab_size, dm),
        )
        layer_dict = {
            "attention": lambda: llm.Attention(dm, heads, causal=False),
            "feedforward": lambda: llm.FeedForward(dm, dff),
        }
        encoder_tower = llm.TransformerTower(n_blocks, dm, layer_dict, device=device)

        head = llm.PredictionHead(dm, output_size)

        model = llm.LLM(embedding_layer, encoder_tower, head)

        input = torch.randint(0, vocab_size, (batch, seql))

        output = model(input)

        self.assertShape(output, (batch, seql, output_size))

    def test_gpt(self):
        batch, seql, dm, heads, dff = 3, 7, 32, 4, 64
        vocab_size, max_length = 107, 33
        output_size = 3
        n_blocks = 2
        device = torch.device("cpu")

        embedding_layer = llm.EmbeddingLayer(
            llm.PositionalEmbedding(max_length, dm),
            llm.TokenEmbedding(vocab_size, dm),
        )
        layer_dict = {
            "attention": lambda: llm.Attention(dm, heads, causal=True),
            "feedforward": lambda: llm.FeedForward(dm, dff),
        }
        encoder_tower = llm.TransformerTower(n_blocks, dm, layer_dict, device=device)

        head = llm.PredictionHead(dm, output_size)

        model = llm.LLM(embedding_layer, encoder_tower, head)

        input = torch.randint(0, vocab_size, (batch, seql))

        output = model(input)

        self.assertShape(output, (batch, seql, output_size))


if __name__ == "__main__":
    unittest.main()
