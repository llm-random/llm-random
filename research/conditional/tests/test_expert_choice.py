import torch

from research.conditional.moe_layers.expert_choice import ExpertChoiceFF
from lizrd.support.test_utils import GeneralTestCase


class TestExpertChoice(GeneralTestCase):
    def test_permutation(self):
        batch, dm = 4, 16
        experts = 1
        dff = 32
        seql = 16
        layer = ExpertChoiceFF(dm, experts, dff, seql, batch * dm)

        # make sure weights don't change input
        # now make sure each expert is identity matrix
        out = torch.zeros_like(layer.lin1_weight).flatten(end_dim=1)
        print(out.shape)
        res = torch.eye(
            n=layer.lin1_weight.shape[0], m=layer.lin1_weight.shape[2], out=out
        ).reshape(layer.lin1_weight.shape)
        layer.lin1_weight.data = res

        out = torch.zeros_like(layer.lin2_weight).flatten(end_dim=1)
        layer.lin2_weight.data = torch.eye(
            n=layer.lin2_weight.shape[1], m=layer.lin2_weight.shape[2], out=out
        ).reshape(layer.lin2_weight.shape)

        input = torch.rand((batch, seql, dm))
        output = layer(input)
        self.assertTensorAlmostEqual(output, input)

    def test_equivalence_linear(self):
        pass
