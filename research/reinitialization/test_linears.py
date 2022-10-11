import torch

from lizrd.core import bert
import unittest

from lizrd.support.test_utils import GeneralTestCase


class PruneLinearCase(GeneralTestCase):
    def _test_prune(self, layer, bias_tensor, input_tensor):
        # prune with probability 0
        res = layer(input_tensor)
        layer.prune_unstr(0)
        res_after_prune = layer(input_tensor)
        self.assertTensorEqual(res, res_after_prune)

        # prune with probability 1
        layer.prune_unstr(1)
        res_after_prune = layer(input_tensor)
        self.assertTensorEqual(res_after_prune, bias_tensor.repeat(input_tensor.shape[0], 1))


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
