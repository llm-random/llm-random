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
        torch.eye(
            n=layer.lin1.weight.shape[0],
            m=layer.lin1.weight.shape[1],
            out=layer.lin1.weight.data,
        )
        torch.eye(
            n=layer.lin2.weight.shape[0],
            m=layer.lin2.weight.shape[1],
            out=layer.lin2.weight.data,
        )

        input = torch.rand((batch, seql, dm))
        output = layer(input)
        self.assertTensorAlmostEqual(output, input)
