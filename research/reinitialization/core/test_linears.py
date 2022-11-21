import torch

from research.reinitialization.core import linears

from lizrd.support.test_utils import GeneralTestCase
from research.reinitialization.core.pruner import Pruner


class TestPruneLinear(GeneralTestCase):
    def test_smoke(self):
        shapes = [(1, 5), (10, 3), (3, 3)]
        types = [torch.float32, torch.float64, torch.double]
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))

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
    def _test_with_pruner(
        self,
        layer: torch.nn.Module,
        pruner: Pruner,
        inp_tensor: torch.Tensor,
        prob: float,
    ):
        N = 3
        assert N * prob <= 1

        for _ in range(N):
            pruner.prune(prob)

        # assert that number of nonzero is approximately as expected
        self._assert_perc_nonzero(layer, 100 - round(prob * N * 100))

        res = layer(inp_tensor)

        # make sure using optimizer doesn't cause problems (like changing mask)
        optimizer = torch.optim.Adam(layer.parameters())
        loss = torch.sum(2 * res)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # assert that number of nonzero is approximately as expected
        self._assert_perc_nonzero(layer, 100 - round(prob * N * 100))

        for _ in range(N):
            pruner.prune(prob)

        # assert that number of nonzero is approximately as expected
        self._assert_perc_nonzero(layer, 100 - round(prob * 2 * N * 100))


class TestUnstructPruneFF(PruneFFTest):
    def test_smoke(self):
        pruner = Pruner()
        linears.UnstructPruneFF(10, 2, pruner)
        linears.UnstructPruneFF(10, 1, pruner)
        linears.UnstructPruneFF(5, 5, pruner)

    def test_with_pruner(self):
        P = 0.1
        pruner = Pruner()
        layer = linears.UnstructPruneFF(1000, 100, pruner)
        t = torch.rand((20, 1000))
        self._test_with_pruner(layer, pruner, t, P)

    def _assert_perc_nonzero(self, ff_layer, perc_nonzero_exp):
        for linear_layer in [ff_layer.lin1, ff_layer.lin2]:
            print("Checking first layer...")
            nonzero = torch.count_nonzero(linear_layer.mask)
            perc_nonzero = 100 * nonzero / torch.numel(linear_layer.mask)
            perc_nonzero = perc_nonzero.item()
            self.assertAlmostEqual(perc_nonzero, perc_nonzero_exp, 0)


class TestStructPruneFF(PruneFFTest):
    def test_smoke(self):
        pruner = Pruner()
        linears.StructPruneFF(10, 2, pruner)
        linears.StructPruneFF(10, 1, pruner)
        linears.StructPruneFF(5, 5, pruner)

    def test_with_pruner(self):
        P = 0.1
        pruner = Pruner()
        layer = linears.StructPruneFF(10, 100000, pruner)
        t = torch.rand((20, 10))
        self._test_with_pruner(layer, pruner, t, P)

    def _assert_perc_nonzero(self, ff_layer, perc_nonzero_exp):
        nonzero = torch.count_nonzero(ff_layer.mask)
        perc_nonzero = 100 * nonzero / torch.numel(ff_layer.mask)
        perc_nonzero = perc_nonzero.item()
        self.assertAlmostEqual(perc_nonzero, perc_nonzero_exp, 0)


class TestUnstructMagnitudePruneFF(TestUnstructPruneFF):
    def test_smoke(self):
        pruner = Pruner()
        linears.UnstructMagnitudePruneFF(10, 2, pruner)
        linears.UnstructMagnitudePruneFF(10, 1, pruner)
        linears.UnstructMagnitudePruneFF(5, 5, pruner)

    def test_with_pruner(self):
        P = 0.1
        N = 3
        pruner = Pruner()
        layer = linears.UnstructMagnitudePruneFF(1000, 100, pruner)
        inp_tensor = torch.rand((20, 1000))

        for _ in range(N):
            pruner.prune(P)

        # assert that number of nonzero is approximately as expected
        self._assert_perc_nonzero(layer, 100 - round(P * N * 100))

        res = layer(inp_tensor)

        # make sure using optimizer doesn't cause problems (like changing mask)
        optimizer = torch.optim.Adam(layer.parameters())
        loss = torch.sum(2 * res)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # assert that number of nonzero is approximately as expected
        self._assert_perc_nonzero(layer, 100 - round(P * N * 100))

        for _ in range(N):
            pruner.prune(P)

        # assert that number of nonzero is approximately as expected
        self._assert_perc_nonzero(layer, 100 - 2 * round(P * N * 100))

    def test_magnitude(self):
        P = 0.001
        pruner = Pruner()
        layer = linears.UnstructMagnitudePruneFF(1000, 100, pruner)

        d = torch.diagonal(layer.lin1.weight.data)
        d *= 0
        r = layer.lin2.weight.data[2] = 0

        pruner.prune(P)

        d = torch.diagonal(layer.lin1.mask.data)
        assert torch.count_nonzero(d) == 0

        r = layer.lin2.mask.data[2]
        assert torch.count_nonzero(r) == 0


class TestStructMagnitudePruneFF(TestStructPruneFF):
    def test_smoke(self):
        pruner = Pruner()
        linears.StructMagnitudePruneFF(10, 2, pruner)
        linears.StructMagnitudePruneFF(10, 1, pruner)
        linears.StructMagnitudePruneFF(5, 5, pruner)

    def test_with_pruner(self):
        P = 0.1
        N = 3
        pruner = Pruner()
        layer = linears.StructMagnitudePruneFF(1000, 100, pruner)
        inp_tensor = torch.rand((20, 1000))

        for _ in range(N):
            pruner.prune(P)

        # assert that number of nonzero is approximately as expected
        self._assert_perc_nonzero(layer, 100 - round(P * N * 100))

        res = layer(inp_tensor)

        # make sure using optimizer doesn't cause problems (like changing mask)
        optimizer = torch.optim.Adam(layer.parameters())
        loss = torch.sum(2 * res)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # assert that number of nonzero is approximately as expected
        self._assert_perc_nonzero(layer, 100 - round(P * N * 100))

        for _ in range(N):
            pruner.prune(P)

        # assert that number of nonzero is approximately as expected
        self._assert_perc_nonzero(layer, 40)

    def test_magnitude(self):
        P = 0.1
        pruner = Pruner()
        layer = linears.StructMagnitudePruneFF(10, 10, pruner)

        layer.lin1.weight.data[7, :] = 0
        layer.lin2.weight.data[:, 7] = 0

        pruner.prune(P)

        assert layer.mask[7] == 0


class TestLayersPruningEquivalence(GeneralTestCase):
    """Tests if pruning the same number of times with the same rate in various pruning layers
    prunes the same percentage of weights."""

    def test_equivalence(self):
        P = 0.1
        N = 6
        DM = 64
        DFF = 128

        def get_mask_zeros_part(layer) -> float:
            return (torch.count_nonzero(layer.mask) / torch.numel(layer.mask)).item()

        classes = [
            linears.UnstructPruneFF,
            linears.StructPruneFF,
            linears.UnstructMagnitudePruneFF,
            linears.StructMagnitudePruneFF,
        ]

        registered_layers = []

        for layer_class in classes:
            pruner = Pruner()
            layer = layer_class(DM, DFF, pruner)
            for _ in range(N):
                pruner.prune(P)
            registered_layers.extend(pruner.layers)

        masks_parts = [
            get_mask_zeros_part(layer) if not isinstance(layer, str) else layer
            for layer in registered_layers
        ]

        for mask_part in masks_parts:
            self.assertAlmostEqual(masks_parts[0], mask_part, delta=0.01)
