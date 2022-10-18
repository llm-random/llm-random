import torch

from research.reinitialization import linears

from lizrd.support.test_utils import GeneralTestCase
from research.reinitialization.pruner import Pruner


class TestPruneLinear(GeneralTestCase):
    def test_smoke(self):
        shapes = [(1, 5), (10, 3), (3, 3)]
        types = [torch.float32, torch.float64, torch.double]
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))

        linears.PruneLinear(10, 3)

        for shape, dtype, device in zip(shapes, types, devices):
            linears.PruneLinear(*shape, dtype=dtype, device=device)

    def test_basic(self):
        layer = linears.PruneLinear(2, 5)
        b = layer.bias.data
        t = torch.rand((10, 2))

        # prune with probability 0
        res = layer(t)
        layer.prune(0)
        res_after_prune = layer(t)
        self.assertTensorEqual(res, res_after_prune)

        # prune with probability 1
        layer.prune(1)
        res_after_prune = layer(t)
        self.assertTensorEqual(res_after_prune, b.repeat(t.shape[0], 1))


class PruneFFTest(GeneralTestCase):
    def _test_with_pruner(self, layer, pruner, inp_tensor):
        for _ in range(4):
            pruner.step()

        # assert that number of nonzero is approximately as expected
        self._assert_perc_nonzero(layer, 64)

        res = layer(inp_tensor)

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


class TestUnstructPruneFF(PruneFFTest):
    def test_smoke(self):
        pruner = Pruner(1, 0.5)
        linears.UnstructPruneFF(10, 2, pruner)
        linears.UnstructPruneFF(10, 1, pruner)
        linears.UnstructPruneFF(5, 5, pruner)

    def test_with_pruner(self):
        pruner = Pruner(2, 0.2)
        layer = linears.UnstructPruneFF(1000, 100, pruner)
        t = torch.rand((20, 1000))
        self._test_with_pruner(layer, pruner, t)

    def _assert_perc_nonzero(self, ff_layer, perc_nonzero_exp):
        for linear_layer in [ff_layer.lin1, ff_layer.lin2]:
            print('Checking first layer...')
            nonzero = torch.count_nonzero(linear_layer.mask)
            perc_nonzero = 100 * nonzero / torch.numel(linear_layer.mask)
            perc_nonzero = perc_nonzero.item()
            self.assertAlmostEqual(perc_nonzero, perc_nonzero_exp, 0)


class TestStructPruneFF(PruneFFTest):
    def test_smoke(self):
        pruner = Pruner(1, 0.5)
        linears.StructPruneFF(10, 2, pruner)
        linears.StructPruneFF(10, 1, pruner)
        linears.StructPruneFF(5, 5, pruner)

    def test_with_pruner(self):
        pruner = Pruner(2, 0.2)
        layer = linears.StructPruneFF(10, 100000, pruner)
        t = torch.rand((20, 10))
        self._test_with_pruner(layer, pruner, t)

    def _assert_perc_nonzero(self, ff_layer, perc_nonzero_exp):
        nonzero = torch.count_nonzero(ff_layer.mask)
        perc_nonzero = 100 * nonzero / torch.numel(ff_layer.mask)
        perc_nonzero = perc_nonzero.item()
        self.assertAlmostEqual(perc_nonzero, perc_nonzero_exp, 0)
