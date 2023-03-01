import torch

from lizrd.support.test_utils import GeneralTestCase
from research.nonlinearities.core.research_bert import (
    FeedForwardBottleneck,
    FeedForwardMultineck,
    FeedForwardInceptionNeck,
    FeedForwardChoppedNeck,
)


class TestFFNecks(GeneralTestCase):
    def test_ff_bottleneck(self):
        batch, seqlen, dmodel, exp_rate = 3, 5, 256, 11
        input = torch.normal(0.0, 1.0, (batch, seqlen, dmodel))
        layer = FeedForwardBottleneck(dmodel, exp_rate)
        output = layer(input)
        self.assertShape(output, (batch, seqlen, dmodel))

    def test_ff_multineck_vanilla(self):
        batch, seqlen, dmodel, exprate, n_heads = 3, 5, 256, 8, 16
        input = torch.normal(0.0, 1.0, (batch, seqlen, dmodel))
        layer = FeedForwardMultineck(dmodel, exprate, n_heads, "none")
        output = layer(input)
        self.assertShape(output, (batch, seqlen, dmodel))

    def test_ff_multineck_inner_layer_parameter_sharing(self):
        batch, seqlen, dmodel, exprate, n_heads = 3, 5, 256, 8, 16
        input = torch.normal(0.0, 1.0, (batch, seqlen, dmodel))
        layer = FeedForwardMultineck(dmodel, exprate, n_heads, "neck_and_ff")
        output = layer(input)
        self.assertShape(output, (batch, seqlen, dmodel))

    def test_ff_inception_neck(self):
        batch, seqlen, dmodel, exprate, head_sizes = 3, 5, 256, 8, [0.25, 0.5, 0.25]
        input = torch.normal(0.0, 1.0, (batch, seqlen, dmodel))
        layer = FeedForwardInceptionNeck(dmodel, exprate, head_sizes)
        output = layer(input)
        self.assertShape(output, (batch, seqlen, dmodel))

    def test_ff_chopped_neck(self):
        batch, seqlen, dmodel, n_chunks = 3, 5, 256, 8
        input = torch.normal(0.0, 1.0, (batch, seqlen, dmodel))
        layer = FeedForwardChoppedNeck(dmodel, n_chunks)
        output = layer(input)
        self.assertShape(output, (batch, seqlen, dmodel))
