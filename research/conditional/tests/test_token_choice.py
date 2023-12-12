from copy import deepcopy
import torch
from torch.nn import Sequential, ReLU
from lizrd.core.misc import Linear, propagate_forward_pass_cache

from research.conditional.moe_layers.token_choice import TokenChoiceFF
from lizrd.support.test_utils import GeneralTestCase


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
        batch, dm = 2, 3
        experts = 1
        exp_size = 6
        seql = 2
        lin = Sequential(
            Linear(
                dm, exp_size, init_type="kaiming_uniform", init_scale=1.0, bias=False
            ),
            ReLU(),
            Linear(
                exp_size, dm, init_type="kaiming_uniform", init_scale=1.0, bias=False
            ),
        )
        token_choice_layer = TokenChoiceFF(
            dm,
            experts,
            exp_size,
            5.0,
            load_balancing_loss_weight=0.1,
            init_type="kaiming_uniform",
            init_scale=1.0,
        )
        propagate_forward_pass_cache(token_choice_layer)

        token_choice_layer.lin1_weight.data = (
            lin[0].weight.data.transpose(0, 1).unsqueeze(0)
        )
        token_choice_layer.lin2_weight.data = (
            lin[2].weight.data.transpose(0, 1).unsqueeze(0)
        )

        # make sure weights act the same
        input_data = torch.rand((batch, seql, dm))
        output_lin = lin(input_data)
        output_token_choice = token_choice_layer(input_data)

        self.assertTensorAlmostEqual(output_lin, output_token_choice)

        # backprop and make sure gradients are the same
        output_lin.sum().backward()
        output_token_choice.sum().backward()
        self.assertTensorAlmostEqual(
            lin[0].weight.grad,
            token_choice_layer.lin1_weight.grad.squeeze(0).transpose(0, 1),
        )
        self.assertTensorAlmostEqual(
            lin[2].weight.grad,
            token_choice_layer.lin2_weight.grad.squeeze(0).transpose(0, 1),
        )

    def test_einsum_vs_matmul(self):
        batch = 3
        dm = 5
        experts = 7
        exp_size = 11
        seq_len = 13
        x = torch.rand((batch, seq_len, dm))
        einsum_module = TokenChoiceFF(
            dm,
            experts,
            exp_size,
            5.0,
            load_balancing_loss_weight=0.1,
            init_type="kaiming_uniform",
            init_scale=1.0,
        )

        matmul_module = deepcopy(einsum_module)
        matmul_module.use_einsum = False

        propagate_forward_pass_cache(einsum_module)
        propagate_forward_pass_cache(matmul_module)
        self.assertTensorAlmostEqual(matmul_module(x), einsum_module(x))
