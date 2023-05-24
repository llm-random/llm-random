import torch
import torch.nn.functional as F
from fancy_einsum import einsum

from research.conditional.moe_layers.expert_choice import ExpertChoiceFF
from lizrd.support.test_utils import GeneralTestCase


class TestExpertChoice(GeneralTestCase):
    def test_permutation(self):
        batch, dm = 2, 4
        experts = 1
        exp_size = 6
        seql = 2
        topk = batch * seql
        layer = ExpertChoiceFF(dm, experts, exp_size, seql, topk)

        # make sure weights don't change input
        layer.lin1_weight.data = torch.eye(m=exp_size, n=dm).unsqueeze(0)
        layer.lin2_weight.data = torch.eye(m=exp_size, n=dm).unsqueeze(0)

        # make sure weight matrices are identity
        input = torch.rand((1, batch * seql, dm))
        x = einsum(
            "n_exp topk dmodel, n_exp dmodel exp_size -> n_exp exp_size topk",
            input,
            layer.lin1_weight,
        )
        x = F.relu(x)
        output = einsum(
            "n_exp dmodel exp_size, n_exp exp_size topk -> n_exp topk dmodel",
            layer.lin2_weight,
            x
        )
        self.assertTensorAlmostEqual(output, input)

        # make sure permutation works as intended
        input = torch.rand((batch, seql, dm))
        output = layer(input)
        self.assertTensorAlmostEqual(output, input)

    def test_equivalence_linear(self):
        pass
