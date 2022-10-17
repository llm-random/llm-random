import torch

from research.reinitialization import linears

from lizrd.support.test_utils import GeneralTestCase
from research.reinitialization.pruner import Pruner


class TestReinitLinear(GeneralTestCase):
    def test_smoke(self):
        shapes = [(1, 5), (10, 3), (3, 3)]
        types = [torch.float32, torch.float64, torch.double]
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))

        linears.ReinitLinear(10, 3)

        for shape, dtype, device in zip(shapes, types, devices):
            linears.ReinitLinear(*shape, dtype=dtype, device=device)

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

    def test_basic(self):
        pruner = Pruner(1, 0.2)
        layer = linears.ReinitLinear(2, 5)
        b = layer.bias.data
        t = torch.rand((10, 2))

        self._test_prune(layer, b, t)


class TestReinitFF(GeneralTestCase):
    def test_smoke(self):
        pruner = Pruner(1, 0.5)
        linears.ReinitFF(10, 2, pruner)
        linears.ReinitFF(10, 1, pruner)
        linears.ReinitFF(5, 5, pruner)

    def test_with_pruner(self):
        pruner = Pruner(2, 0.2)
        layer = linears.ReinitFF(1000, 100, pruner)

        b = layer.linears[2].bias.data
        t = torch.rand((10, 1000))

        for _ in range(4):
            pruner.step()

        # assert that number of nonzero is approximately as expected
        self._assert_perc_nonzero(layer, 64)

        res = layer(t)

        # make sure using optimizer doesn't cause problems (like changing mask)
        optimizer = torch.optim.Adam(layer.parameters())
        loss = torch.sum(2 * res)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # assert that number of nonzero is approximately as expected
        self._assert_perc_nonzero(layer, 64)

        for _ in range(2):
            pruner.step()

        # assert that number of nonzero is approximately as expected
        self._assert_perc_nonzero(layer, 51.2)

    def _assert_perc_nonzero(self, ff_layer, perc_nonzero_exp):
        for linear_layer in [ff_layer.linears[0], ff_layer.linears[2]]:
            nonzero = torch.count_nonzero(linear_layer.mask)
            perc_nonzero = 100 * nonzero / torch.numel(linear_layer.mask)
            perc_nonzero = perc_nonzero.item()
            self.assertAlmostEqual(perc_nonzero, perc_nonzero_exp, 0)
