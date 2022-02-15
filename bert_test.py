import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import time
from time import sleep
from opt_einsum import contract
import opt_einsum
from einops.layers.torch import Rearrange, Reduce
#https://einops.rocks/3-einmix-layer/

import gc
from typing import List

import bert
import unittest


class GeneralTestCase(unittest.TestCase):
    def assertShape(self, tensor, shape):
        self.assertEqual(tuple(tensor.shape), tuple(shape))

    def assertTensorEqual(self, tensor1, tensor2):
        self.assertShape(tensor1, tensor2.shape)
        list1 = list(torch.flatten(tensor1).detach().numpy())
        list2 = list(torch.flatten(tensor2).detach().numpy())
        self.assertListEqual(list1, list2)

    def assertTensorAlmostEqual(self, tensor1, tensor2):
        self.assertShape(tensor1, tensor2.shape)
        list1 = torch.flatten(tensor1).detach().numpy()
        list2 = torch.flatten(tensor2).detach().numpy()
        almostequal = np.isclose(list1, list2,
                                 rtol=1e-5, atol=1e-5)
        listA = list1 * (1-almostequal) + list2 * almostequal
        self.assertListEqual(list(listA), list(list2))


class TestDense(GeneralTestCase):
    def test_basic(self):
        batch, dinp, dout = 4, 32, 64
        layer = bert.Dense(dinp, dout)
        input = torch.normal(0.0, 1.0, (batch, dinp))
        output = layer(input)
        self.assertShape(output, (batch, dout))

    def test_more_dims(self):
        batch, seql, dinp, dout = 4, 8, 32, 64
        layer = bert.Dense(dinp, dout)
        input = torch.normal(0.0, 1.0, (batch, seql, dinp))
        output = layer(input)
        self.assertShape(output, (batch, seql, dout))


class TestEinMix(GeneralTestCase):
    def test_no_ellipsis(self):
        batch, dinp, dout = 4, 32, 64
        layer = bert.EinMix('b d -> b f',
                            weight_shape='d f', bias_shape='f',
                            d=dinp, f=dout)
        input = torch.normal(0.0, 1.0, (batch, dinp))
        output = layer(input)
        self.assertShape(output, (batch, dout))

    def test_one_ellipsis(self):
        batch, dinp, dout = 4, 32, 64
        layer = bert.EinMix('... d -> ... f',
                            weight_shape='d f', bias_shape='f',
                            d=dinp, f=dout)
        input = torch.normal(0.0, 1.0, (batch, dinp))
        output = layer(input)
        self.assertShape(output, (batch, dout))

    def test_two_ellipsis(self):
        batch, seqlen, dinp, dout = 4, 2, 32, 64
        layer = bert.EinMix('... d -> ... f',
                            weight_shape='d f', bias_shape='f',
                            d=dinp, f=dout)
        input = torch.normal(0.0, 1.0, (batch, seqlen, dinp))
        output = layer(input)
        self.assertShape(output, (batch, seqlen, dout))

    def test_two_ellipsis(self):
        batch, seqlen, whatever, dinp, dout = 5, 7, 3, 32, 64
        layer = bert.EinMix('... d -> ... f',
                            weight_shape='d f', bias_shape='f',
                            d=dinp, f=dout)
        input = torch.normal(0.0, 1.0, (batch, seqlen, whatever, dinp))
        output = layer(input)
        self.assertShape(output, (batch, seqlen, whatever, dout))


class TestFeedForward(GeneralTestCase):
    def test_basic(self):
        batch, seql, dm, dff = 4, 8, 32, 64
        layer = bert.FeedForward(dm, dff)
        input = torch.normal(0.0, 1.0, (batch, seql, dm))
        output = layer(input)
        self.assertShape(output, (batch, seql, dm))


class TestBatchedFeedForward(GeneralTestCase):
    def test_basic(self):
        batch, dm = 4, 32
        sets = 5
        experts = 7
        seql = experts * 3
        expertsize = 11
        dff = sets * experts * expertsize
        layer = bert.BatchSplitFF([], dm, dff, sets, experts, expertsize)
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


class BERTSparseTest(GeneralTestCase):
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
