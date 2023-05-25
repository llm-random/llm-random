import torch
import torch.nn.functional as F
from torch.nn import Sequential, ReLU, LayerNorm
from fancy_einsum import einsum

from research.conditional.moe_layers.expert_choice import ExpertChoiceFF
from lizrd.support.test_utils import GeneralTestCase
from lizrd.core.misc import Linear


class TestExpertChoice(GeneralTestCase):
    def test_permutation(self):
        batch, dm = 2, 2
        experts = 1
        exp_size = 6
        seql = 2
        topk_perc = 100
        layer = ExpertChoiceFF(dm, experts, exp_size, topk_perc)

        # make sure weights don't change input
        layer.lin1_weight.data = torch.eye(m=exp_size, n=dm).unsqueeze(0)
        layer.lin2_weight.data = torch.eye(m=dm, n=exp_size).unsqueeze(0)

        # make sure weight matrices are identity
        input = torch.rand((1, batch * seql, dm))
        x = einsum(
            "n_exp topk dmodel, n_exp dmodel exp_size -> n_exp topk exp_size",
            input,
            layer.lin1_weight,
        )
        x = F.relu(x)
        output = einsum(
            "n_exp topk exp_size, n_exp exp_size dmodel -> n_exp topk dmodel",
            x,
            layer.lin2_weight,
        )
        self.assertTensorAlmostEqual(output, input)

        # make sure permutation works as intended
        input = torch.rand((batch, seql, dm))
        output = layer(input)
        self.assertTensorAlmostEqual(output, input)

    def test_equivalence_linear(self, softmax_disabled=False):
        batch, dm = 2, 2
        experts = 1
        exp_size = 6
        seql = 2
        topk_perc = 100
        ln = LayerNorm(dm)
        lin = Sequential(
            Linear(dm, exp_size, bias=False), ReLU(), Linear(exp_size, dm, bias=False)
        )
        ec = ExpertChoiceFF(dm, experts, exp_size, topk_perc)
        ec.lin1_weight.data = lin[0].weight.data.transpose(0, 1).unsqueeze(0)
        ec.lin2_weight.data = lin[2].weight.data.transpose(0, 1).unsqueeze(0)

        # make sure weights act the same
        input = torch.rand((batch, seql, dm))
        output_lin = lin(input)

        input = input.flatten(0, 1).unsqueeze(0)
        input = input.reshape(1, batch * seql, dm)
        x = einsum(
            "n_exp topk dmodel, n_exp dmodel exp_size -> n_exp topk exp_size",
            input,
            ec.lin1_weight,
        )

        x = F.relu(x)

        # lin2 maps from (...) to (n_experts, topk, dmodel)
        output_ec = einsum(
            "n_exp topk exp_size, n_exp exp_size dmodel -> n_exp topk dmodel",
            x,
            ec.lin2_weight,
        )
        output_ec = output_ec.reshape(batch, seql, dm)
        self.assertTensorAlmostEqual(output_lin, output_ec)

        # create inputs and make sure both layers give the same output
        # works only when multiply by softmax is commended out
        if softmax_disabled:
            input = torch.rand((batch, seql, dm))
            output_lin = lin(input)
            output_ec = ec(input)
            self.assertTensorAlmostEqual(output_lin, output_ec)

        # backprop and make sure gradients are the same
        output_lin.sum().backward()
        output_ec.sum().backward()
        self.assertTensorAlmostEqual(
            lin[0].weight.grad, ec.lin1_weight.grad.squeeze(0).transpose(0, 1)
        )
        self.assertTensorAlmostEqual(
            lin[2].weight.grad, ec.lin2_weight.grad.squeeze(0).transpose(0, 1)
        )
