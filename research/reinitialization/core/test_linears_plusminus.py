import torch

from research.reinitialization.core import linears_plusminus

from lizrd.support.test_utils import GeneralTestCase


"""
@ash.check("... d -> ... d")
class MultiGroupedConv(nn.Module):
    def __init__(self, dmodel: int, dff: int):
"""


class TestMultiGroupedConv(GeneralTestCase):
    def test_smoke(self):
        dmodel = 1024
        dff = 4096
        module = linears_plusminus.MultiGroupedConv(dmodel, dff)
        x = torch.randn(4, 32, dmodel)
        y = module(x)
        self.assertEqual(y.shape, x.shape)

    def test_smoke2(self):
        dmodel = 512
        dff = 2048
        module = linears_plusminus.MultiGroupedConv(dmodel, dff)
        x = torch.randn(4, 32, dmodel)
        y = module(x)
        self.assertEqual(y.shape, x.shape)

