from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.nn import Sequential, ReLU, Identity
from fancy_einsum import einsum
from lizrd.train.checkpointing import (
    first_forward_manager,
    make_checkpoint_wrapper_function,
    second_forward_manager,
)

from research.inverted.moe_layers.expert_choice import ExpertChoiceFF
from research.inverted.moe_layers._expert_choice_old import ExpertChoiceFFOld
from research.inverted.moe_layers.expert_types import ExpertFF
from lizrd.support.test_utils import GeneralTestCase
from lizrd.core.misc import Linear
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)


def mock_topk_factory(topk_fn):
    def mock_topk(x, k, dim):
        values, indices = topk_fn(x, k=k, dim=dim)
        return torch.ones_like(values), indices

    return mock_topk


def create_expert_choice(
    dmodel,
    n_experts,
    expert_size,
    topk_fracton,
    init_type="kaiming_uniform",
    init_scale=1.0,
    *args,
    **kwargs
):
    topk = int(topk_fracton * n_experts)
    expert_inner = ExpertFF(
        dmodel,
        n_experts,
        expert_size,
        init_type,
        init_scale,
        topk=topk,
        use_topk_initialization=True,
    )
    return ExpertChoiceFF(
        dmodel,
        n_experts,
        topk_fracton,
        expert_inner_function=expert_inner,
        init_type=init_type,
        init_scale=init_scale,
        *args,
        **kwargs,
    )


class TestExpertChoice(GeneralTestCase):
    def test_permutation(self):
        """
        Test that the ExpertChoiceFF layer permutes the input as intended.
        Only the case with one expert is tested here.
        """

        batch, dm = 2, 2
        experts = 1
        exp_size = 6
        seql = 2
        topk_fraction = 1
        layer = create_expert_choice(dm, experts, exp_size, topk_fraction)
        layer.ln = Identity()

        # make sure weights don't change input
        layer.expert_inner_function.lin1_weight.data = torch.eye(
            m=exp_size, n=dm
        ).unsqueeze(0)
        layer.expert_inner_function.lin2_weight.data = torch.eye(
            m=dm, n=exp_size
        ).unsqueeze(0)

        # make sure weight matrices are identity
        input = torch.rand((1, batch * seql, dm))
        x = einsum(
            "n_exp topk dmodel, n_exp dmodel exp_size -> n_exp topk exp_size",
            input,
            layer.expert_inner_function.lin1_weight,
        )
        x = F.relu(x)
        output = einsum(
            "n_exp topk exp_size, n_exp exp_size dmodel -> n_exp topk dmodel",
            x,
            layer.expert_inner_function.lin2_weight,
        )
        self.assertTensorAlmostEqual(output, input)

        # make sure permutation works as intended
        input = torch.rand((batch, seql, dm))
        with patch(
            "torch.topk", wraps=mock_topk_factory(torch.topk)
        ):  # patch torch topk, s t. we don't multiply by softmax
            output = layer(input)
        self.assertTensorAlmostEqual(output, input)

    def test_old_new_equivalence(self):
        """
        Test that checks if the one-hot implementation of ExpertChoiceFF is equivalent to the original.
        """
        batch, dm = 2, 2
        experts = 2
        exp_size = 6
        seql = 2
        topk_fraction = 0.5
        layer = create_expert_choice(
            dm,
            experts,
            exp_size,
            topk_fraction,
            one_hot_impl=True,
            group_by_batch=True,
            use_torch_bmm=True,
        )
        layer_old = ExpertChoiceFFOld(
            dm,
            experts,
            exp_size,
            topk_fraction,
            init_type="kaiming_uniform",
            init_scale=1.0,
            one_hot_impl=True,
            group_by_batch=True,
        )
        layer_old.lin1_weight.data = layer.expert_inner_function.lin1_weight.data
        layer_old.lin2_weight.data = layer.expert_inner_function.lin2_weight.data
        layer_old.expert_gating.gate.data = layer.gating.gate.data
        layer_old.ln = layer.ln

        input = torch.rand((batch, seql, dm))

        output = layer.forward(input)
        output_onehot = layer_old.forward(input)

        self.assertTensorAlmostEqual(output, output_onehot)

    def test_onehot_bmm_equivalence(self):
        """
        Test that checks if the one-hot implementation of ExpertChoiceFF is equivalent to the original.
        """
        batch, dm = 2, 2
        experts = 2
        exp_size = 6
        seql = 2
        topk_fraction = 0.5

        layer = create_expert_choice(
            dm,
            experts,
            exp_size,
            topk_fraction,
            one_hot_impl=True,
            group_by_batch=True,
            use_torch_bmm=True,
        )
        layer_einsum = create_expert_choice(
            dm,
            experts,
            exp_size,
            topk_fraction,
            one_hot_impl=True,
            group_by_batch=True,
        )
        layer_einsum.expert_inner_function.lin1_weight.data = (
            layer.expert_inner_function.lin1_weight.data
        )
        layer_einsum.expert_inner_function.lin2_weight.data = (
            layer.expert_inner_function.lin2_weight.data
        )
        layer_einsum.gating.gate.data = layer.gating.gate.data
        layer_einsum.ln = layer.ln

        input = torch.rand((batch, seql, dm))

        output = layer.forward(input)
        output_onehot = layer_einsum.forward(input)

        self.assertTensorAlmostEqual(output, output_onehot)

    def test_onehot_equivalence(self):
        """
        Test that checks if the one-hot implementation of ExpertChoiceFF is equivalent to the original.
        """
        batch, dm = 2, 2
        experts = 2
        exp_size = 6
        seql = 2
        topk_fraction = 0.5

        layer = create_expert_choice(
            dm,
            experts,
            exp_size,
            topk_fraction,
            init_type="kaiming_uniform",
            init_scale=1.0,
            one_hot_impl=False,
            group_by_batch=True,
        )

        layer_onehot = create_expert_choice(
            dm,
            experts,
            exp_size,
            topk_fraction,
            init_type="kaiming_uniform",
            init_scale=1.0,
            one_hot_impl=True,
            group_by_batch=True,
        )
        layer_onehot.expert_inner_function.lin1_weight.data = (
            layer.expert_inner_function.lin1_weight.data
        )
        layer_onehot.expert_inner_function.lin2_weight.data = (
            layer.expert_inner_function.lin2_weight.data
        )
        layer_onehot.gating.gate.data = layer.gating.gate.data
        layer_onehot.ln = layer.ln

        input = torch.rand((batch, seql, dm))

        output = layer.forward(input)
        output_onehot = layer_onehot.forward(input)

        self.assertTensorAlmostEqual(output, output_onehot)

    def test_equivalence_linear(self):
        """
        Test that the ExpertChoiceFF layer with one expert is equivalent to a linear layer.

        If we don't multiply by softmax, the layer is equivalent
        to a linear layer (plus LayerNorm) with regard to output and gradients.

        If we multiply by softmax the layer is equivalent with regard to gradients only.
        """
        batch, dm = 2, 2
        experts = 1
        exp_size = 6
        seql = 2
        topk_fraction = 1
        lin = Sequential(
            Linear(
                dm, exp_size, init_type="kaiming_uniform", init_scale=1.0, bias=False
            ),
            ReLU(),
            Linear(
                exp_size, dm, init_type="kaiming_uniform", init_scale=1.0, bias=False
            ),
        )
        ec = create_expert_choice(
            dm,
            experts,
            exp_size,
            topk_fraction,
        )
        ec.expert_inner_function.lin1_weight.data = (
            lin[0].weight.data.transpose(0, 1).unsqueeze(0)
        )
        ec.expert_inner_function.lin2_weight.data = (
            lin[2].weight.data.transpose(0, 1).unsqueeze(0)
        )
        ln = ec.ln

        # make sure weights act the same
        input = torch.rand((batch, seql, dm))
        output_lin = lin(input)

        input = input.flatten(0, 1).unsqueeze(0)
        input = input.reshape(1, batch * seql, dm)
        x = einsum(
            "n_exp topk dmodel, n_exp dmodel exp_size -> n_exp topk exp_size",
            input,
            ec.expert_inner_function.lin1_weight,
        )

        x = F.relu(x)

        # lin2 maps from (...) to (n_experts, topk, dmodel)
        output_ec = einsum(
            "n_exp topk exp_size, n_exp exp_size dmodel -> n_exp topk dmodel",
            x,
            ec.expert_inner_function.lin2_weight,
        )
        output_ec = output_ec.reshape(batch, seql, dm)
        self.assertTensorAlmostEqual(output_lin, output_ec)

        input = torch.rand((batch, seql, dm))
        output_lin = ln(lin(input))
        with patch(
            "torch.topk", wraps=mock_topk_factory(torch.topk)
        ):  # patch torch topk, s t. we don't multiply by softmax
            output_ec = ec(input)
        self.assertTensorAlmostEqual(output_lin, output_ec)

        # backprop and make sure gradients are the same
        output_lin.sum().backward()
        output_ec.sum().backward()
        self.assertTensorAlmostEqual(
            lin[0].weight.grad,
            ec.expert_inner_function.lin1_weight.grad.squeeze(0).transpose(0, 1),
        )
        self.assertTensorAlmostEqual(
            lin[2].weight.grad,
            ec.expert_inner_function.lin2_weight.grad.squeeze(0).transpose(0, 1),
        )

    def test_checkpointing(self):
        """
        Test that checkpointing implementation is equivalent to non-checkpointed one.
        """
        batch = 2
        dm = 3
        experts = 5
        exp_size = 7
        seql = 11
        topk_fraction = 0.5

        ec = create_expert_choice(
            dm,
            experts,
            exp_size,
            topk_fraction,
        )

        x = torch.rand((batch, seql, dm))

        output_no_checkpointing = ec(x)
        with first_forward_manager():
            output_checkpointing_1st_forward = ec(x)

        with second_forward_manager():
            output_checkpointing_2nd_forward = ec(x)

        self.assertTensorAlmostEqual(
            output_no_checkpointing, output_checkpointing_1st_forward
        )
        self.assertTensorAlmostEqual(
            output_no_checkpointing, output_checkpointing_2nd_forward
        )

    def test_checkpointing_compatibility(self):
        """
        Test that checkpointing when the module is actually checkpointed.
        """
        batch = 2
        dm = 3
        experts = 5
        exp_size = 7
        seql = 11
        topk_fraction = 0.5

        non_reentrant_wrapper = make_checkpoint_wrapper_function()

        ec = torch.nn.Sequential(
            create_expert_choice(
                dm,
                experts,
                exp_size,
                topk_fraction,
            )
        )

        apply_activation_checkpointing(
            ec,
            check_fn=lambda module: isinstance(module, ExpertChoiceFF),
            checkpoint_wrapper_fn=non_reentrant_wrapper,
        )
        x = torch.rand((batch, seql, dm))
        loss = ec(x).reshape(-1).sum()
        loss.backward()
