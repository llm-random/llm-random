import torch
from torch.nn import Sequential, ReLU

from research.conditional.moe_layers.token_choice import TokenChoiceFF
from lizrd.support.test_utils import GeneralTestCase
from lizrd.core.misc import Linear


def mock_topk_factory(topk_fn):
    def mock_topk(x, k, dim):
        values, indices = topk_fn(x, k=k, dim=dim)
        return torch.ones_like(values), indices

    return mock_topk


class TestTokenChoice(GeneralTestCase):
    def test_equivalence_linear(self):
        """
        Test that the TokenChoiceFF layer with one expert is equivalent to a linear layer.

        If we don't multiply by softmax, the layer is equivalent
        to a linear layer (plus LayerNorm) with regard to output and gradients.

        If we multiply by softmax the layer is equivalent with regard to gradients only.
        """
        batch, dm = 2, 2
        experts = 1
        exp_size = 6
        seql = 2
        lin = Sequential(
            Linear(dm, exp_size, bias=False), ReLU(), Linear(exp_size, dm, bias=False)
        )
        token_choice_layer = TokenChoiceFF(dm, experts, exp_size, 5.0)
        token_choice_layer.lin1_weights[0].data = lin[0].weight.data.transpose(0, 1)
        token_choice_layer.lin2_weights[0].data = lin[2].weight.data.transpose(0, 1)
        ln = token_choice_layer.ln

        # make sure weights act the same
        input = torch.rand((batch, seql, dm))
        output_lin = ln(lin(input))
        output_token_choice = token_choice_layer(input)

        self.assertTensorAlmostEqual(output_lin, output_token_choice)

        # backprop and make sure gradients are the same
        output_lin.sum().backward()
        output_token_choice.sum().backward()
        self.assertTensorAlmostEqual(
            lin[0].weight.grad,
            token_choice_layer.lin1_weights[0].grad.squeeze(0).transpose(0, 1),
        )
        self.assertTensorAlmostEqual(
            lin[2].weight.grad,
            token_choice_layer.lin2_weights[0].grad.squeeze(0).transpose(0, 1),
        )
