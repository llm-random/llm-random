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
