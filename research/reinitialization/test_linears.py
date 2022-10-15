import torch

from research.reinitialization import linears

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
    def test_smoke(self):
        shapes = [(1, 5), (10, 3), (3, 3)]
        types = [torch.float32, torch.float64, torch.double]
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))

        linears.ReinitLinear(10, 3)

        for shape, dtype, device in zip(shapes, types, devices):
            linears.ReinitLinear(*shape, dtype=dtype, device=device)

    def test_basic(self):
        layer = linears.ReinitLinear(2, 5)
        b = layer.bias.data
        t = torch.rand((10, 2))

        self._test_prune(layer, b, t)


class TestReinitFF(PruneLinearCase):
    def test_smoke(self):
        linears.ReinitFF(10, 2)
        linears.ReinitFF(10, 1)
        linears.ReinitFF(5, 5)

    def test_basic(self):
        layer = linears.ReinitFF(10, 2)
        b = layer.linears[2].bias.data
        t = torch.rand((10, 10))

        self._test_prune(layer, b, t)
