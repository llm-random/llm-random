import torch

from lizrd.core import bert
import unittest

from lizrd.support.test_utils import PruneLinearCase


class TestReinitLinear(PruneLinearCase):
    def smoke(self):
        shapes = [(1, 5), (10, 3), (3, 3)]
        types = [torch.float32, torch.float64, torch.double]
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))

        bert.ReinitLinear(10, 3)

        for shape, dtype, device in zip(shapes, types, devices):
            bert.ReinitLinear(*shape, dtype=dtype, device=device)

    def test_basic(self):
        layer = bert.ReinitLinear(2, 5)
        b = layer.bias.data
        t = torch.rand((10, 2))

        self._test_prune(layer, b, t)


class TestReinitFF(PruneLinearCase):
    def test_smoke(self):
        bert.ReinitFF(10, 2)
        bert.ReinitFF(10, 1)
        bert.ReinitFF(5, 5)

    def test_basic(self):
        layer = bert.ReinitFF(10, 2)
        b = layer.linears[1].bias.data
        t = torch.rand((10, 10))

        self._test_prune(layer, b, t)
        