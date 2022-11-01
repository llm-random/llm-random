import torch

from research.reinitialization import linears_recycle

from lizrd.support.test_utils import GeneralTestCase
from research.reinitialization.pruner import Pruner


class RecycleFFTest(GeneralTestCase):
    def _assert_perc_changed(self, weights_before, weights_after, perc):
        changed = torch.sum(weights_before != weights_after)
        total = torch.numel(weights_before)
        self.assertTensorAlmostEqual(changed / total, perc)


class TestRandomUnstructRecycleFF(RecycleFFTest):
    def test_smoke(self):
        pruner = Pruner(1, 0.5)
        linears_recycle.RandomUnstructRecycleFF(10, 2, pruner)
        linears_recycle.RandomUnstructRecycleFF(10, 1, pruner)
        linears_recycle.RandomUnstructRecycleFF(5, 5, pruner)

    def test_with_pruner(self):
        pruner = Pruner(2, 0.2)
        layer = linears_recycle.RandomUnstructRecycleFF(1000, 100, pruner)
        inp_tensor = torch.rand((20, 1000))

        for _ in range(4):
            pruner.step()

        # assert that number of nonzero is approximately as expected
        self._assert_perc_nonzero(layer, 60)

        res = layer(inp_tensor)

        # make sure using optimizer doesn't cause problems (like changing mask)
        optimizer = torch.optim.Adam(layer.parameters())
        loss = torch.sum(2 * res)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # assert that number of nonzero is approximately as expected
        self._assert_perc_nonzero(layer, 60)

        for _ in range(2):
            pruner.step()

        # assert that number of nonzero is approximately as expected
        self._assert_perc_nonzero(layer, 40)

    def test_magnitude(self):
        pruner = Pruner(2, 0.001)
        layer = linears.UnstructMagnitudePruneFF(1000, 100, pruner)

        d = torch.diagonal(layer.lin1.weight.data)
        d *= 0
        r = layer.lin2.weight.data[2] = 0

        for _ in range(2):
            pruner.step()

        d = torch.diagonal(layer.lin1.mask.data)
        assert torch.count_nonzero(d) == 0

        r = layer.lin2.mask.data[2]
        assert torch.count_nonzero(r) == 0
