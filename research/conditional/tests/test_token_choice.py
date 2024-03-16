from copy import deepcopy
import torch
from torch.nn import Sequential, ReLU
from lizrd.core.misc import Linear, propagate_forward_pass_cache
from lizrd.core.llm import SwiGLUFeedForward
from lizrd.train.checkpointing import (
    first_forward_manager,
    make_checkpoint_wrapper_function,
    second_forward_manager,
)

from research.conditional.moe_layers._token_choice_old import (
    TokenChoiceFFOld,
    ExpertReluOld,
)
from research.conditional.moe_layers.token_choice import TokenChoiceFF
from research.conditional.moe_layers.expert_types import ExpertGated, ExpertFF
from lizrd.support.test_utils import GeneralTestCase

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)


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
        expert_logic = ExpertFF(
            dm, experts, exp_size, "kaiming_uniform", 1.0, activation_name="relu"
        )
        token_choice_layer = TokenChoiceFF(
            dm,
            experts,
            5.0,
            load_balancing_loss_weight=0.1,
            init_type="kaiming_uniform",
            init_scale=1.0,
            expert_inner_function=expert_logic,
        )
        propagate_forward_pass_cache(token_choice_layer)

        token_choice_layer.expert_inner_function.lin1_weight.data = (
            lin[0].weight.data.transpose(0, 1).unsqueeze(0)
        )
        token_choice_layer.expert_inner_function.lin2_weight.data = (
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
            token_choice_layer.expert_inner_function.lin1_weight.grad.squeeze(
                0
            ).transpose(0, 1),
        )
        self.assertTensorAlmostEqual(
            lin[2].weight.grad,
            token_choice_layer.expert_inner_function.lin2_weight.grad.squeeze(
                0
            ).transpose(0, 1),
        )

    def test_equivalence_swi_glu(self):
        """
        Test that the TokenChoiceFFSwiGLU layer with one expert is equivalent to a SwiGluFeedForward.
        """
        batch, dm = 2, 3
        experts = 1
        exp_size = 6
        seql = 2
        lin = SwiGLUFeedForward(dm, exp_size, "kaiming_uniform", 1.0)
        expert_logic = ExpertGated(
            dm, experts, exp_size, "kaiming_uniform", 1.0, activation_name="silu"
        )
        token_choice_layer = TokenChoiceFF(
            dm,
            experts,
            5.0,
            load_balancing_loss_weight=0.1,
            init_type="kaiming_uniform",
            init_scale=1.0,
            expert_inner_function=expert_logic,
        )
        propagate_forward_pass_cache(token_choice_layer)

        token_choice_layer.expert_inner_function.lin1_weight.data = (
            lin.w1_gate.weight[exp_size:].data.transpose(0, 1).unsqueeze(0)
        )
        token_choice_layer.expert_inner_function.gate_weight.data = (
            lin.w1_gate.weight.data[0:exp_size].transpose(0, 1).unsqueeze(0)
        )
        token_choice_layer.expert_inner_function.lin2_weight.data = (
            lin.w2.weight.data.transpose(0, 1).unsqueeze(0)
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
            lin.w1_gate.weight.grad[exp_size:],
            token_choice_layer.expert_inner_function.lin1_weight.grad.squeeze(
                0
            ).transpose(0, 1),
        )
        self.assertTensorAlmostEqual(
            lin.w1_gate.weight.grad[0:exp_size],
            token_choice_layer.expert_inner_function.gate_weight.grad.squeeze(
                0
            ).transpose(0, 1),
        )
        self.assertTensorAlmostEqual(
            lin.w2.weight.grad,
            token_choice_layer.expert_inner_function.lin2_weight.grad.squeeze(
                0
            ).transpose(0, 1),
        )

    def test_einsum_vs_matmul(self):
        batch = 3
        dm = 5
        experts = 7
        exp_size = 11
        seq_len = 13
        x = torch.rand((batch, seq_len, dm))
        expert_logic = ExpertFF(
            dm, experts, exp_size, "kaiming_uniform", 1.0, use_einsum=True
        )
        einsum_module = TokenChoiceFF(
            dm,
            experts,
            5.0,
            load_balancing_loss_weight=0.1,
            init_type="kaiming_uniform",
            init_scale=1.0,
            expert_inner_function=expert_logic,
            use_einsum=True,
        )

        matmul_module = deepcopy(einsum_module)
        matmul_module.use_einsum = False
        matmul_module.expert_inner_function.use_einsum = False
        matmul_module

        propagate_forward_pass_cache(einsum_module)
        propagate_forward_pass_cache(matmul_module)
        self.assertTensorAlmostEqual(matmul_module(x), einsum_module(x))

    def test_checkpointing_values(self):
        """
        Test that checkpointing implementation is equivalent to non-checkpointed one.
        """
        batch = 2
        dm = 3
        experts = 5
        exp_size = 7
        seql = 11

        expert_logic = ExpertFF(dm, experts, exp_size, "kaiming_uniform", 1.0)
        tc = TokenChoiceFF(
            dm,
            experts,
            5.0,
            load_balancing_loss_weight=0.1,
            init_type="kaiming_uniform",
            init_scale=1.0,
            expert_inner_function=expert_logic,
        )
        propagate_forward_pass_cache(tc)

        x = torch.rand((batch, seql, dm))

        output_no_checkpointing = tc(x)
        with first_forward_manager():
            output_checkpointing_1st_forward = tc(x)

        with second_forward_manager():
            output_checkpointing_2nd_forward = tc(x)

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

        non_reentrant_wrapper = make_checkpoint_wrapper_function()

        expert_logic = ExpertFF(dm, experts, exp_size, "kaiming_uniform", 1.0)

        tc = torch.nn.Sequential(
            TokenChoiceFF(
                dm,
                experts,
                5.0,
                load_balancing_loss_weight=0.1,
                init_type="kaiming_uniform",
                init_scale=1.0,
                expert_inner_function=expert_logic,
            )
        )
        propagate_forward_pass_cache(tc)
        apply_activation_checkpointing(
            tc,
            check_fn=lambda module: isinstance(module, TokenChoiceFF),
            checkpoint_wrapper_fn=non_reentrant_wrapper,
        )
        x = torch.rand((batch, seql, dm))
        loss = tc(x).reshape(-1).sum()
        loss.backward()

    def test_new_implementation_compatibility(self):
        """
        Test that the new implementation that allows a token to select multiple experts is equivalent to the old one if topk=1.
        """
        batch = 2
        dm = 3
        experts = 5
        exp_size = 7
        seql = 11

        expert_logic = ExpertFF(dm, experts, exp_size, "kaiming_uniform", 1.0)
        expert_logic_old = ExpertReluOld(dm, experts, exp_size, "kaiming_uniform", 1.0)

        # non_reentrant_wrapper = make_checkpoint_wrapper_function()
        tc = TokenChoiceFF(
            dm,
            experts,
            1.0,
            expert_inner_function=expert_logic,
            load_balancing_loss_weight=0.1,
            routing_top_k=1,
            init_type="kaiming_uniform",
            init_scale=1.0,
        )
        old_tc = TokenChoiceFFOld(
            dm,
            experts,
            1.0,
            expert_inner_function=expert_logic_old,
            load_balancing_loss_weight=0.1,
            routing_top_k=1,
            init_type="kaiming_uniform",
            init_scale=1.0,
            vectorize=True,
        )
        with torch.no_grad():
            old_tc.expert_inner_function.lin1_weight.data = (
                tc.expert_inner_function.lin1_weight.data.clone()
            )
            old_tc.expert_inner_function.lin2_weight.data = (
                tc.expert_inner_function.lin2_weight.data.clone()
            )
            old_tc.router.gate.data = tc.gate.data.clone()

        propagate_forward_pass_cache(tc)
        propagate_forward_pass_cache(old_tc)

        x = torch.rand((batch, seql, dm))

        output_new_implementation = tc(x)
        output_old_implementation = old_tc(x)

        self.assertTensorAlmostEqual(
            output_new_implementation, output_old_implementation
        )

        (
            output_new_implementation.sum()
            + sum(tc.forward_pass_cache["load_balancing_losses"])
        ).backward()
        (
            output_old_implementation.sum()
            + sum(old_tc.forward_pass_cache["load_balancing_losses"])
        ).backward()

        self.assertTensorAlmostEqual(
            tc.expert_inner_function.lin1_weight.grad,
            old_tc.expert_inner_function.lin1_weight.grad,
        )
        self.assertTensorAlmostEqual(
            tc.expert_inner_function.lin2_weight.grad,
            old_tc.expert_inner_function.lin2_weight.grad,
        )
        self.assertTensorAlmostEqual(tc.gate.grad, old_tc.router.gate.grad)

    # TODO przepisać
    def test_topk2_equivalence_linear(self):
        """
        Test that the TokenChoiceFF layer with multiple experts is equivalent to a sum of linear layers.
        """
        batch, dm = 2, 3
        experts = 2
        exp_size = 6
        seql = 2

        def make_expert():
            return Sequential(
                Linear(
                    dm,
                    exp_size,
                    init_type="kaiming_uniform",
                    init_scale=1.0,
                    bias=False,
                ),
                ReLU(),
                Linear(
                    exp_size,
                    dm,
                    init_type="kaiming_uniform",
                    init_scale=1.0,
                    bias=False,
                ),
            )

        expert1 = make_expert()
        expert2 = make_expert()
        expert_logic = ExpertFF(dm, experts, exp_size, "kaiming_uniform", 1.0)
        token_choice_layer = TokenChoiceFFOld(
            dm,
            experts,
            100.0,
            expert_inner_function=expert_logic,
            load_balancing_loss_weight=0.1,
            init_type="kaiming_uniform",
            init_scale=1.0,
            routing_top_k=2,
            vectorize=False,
        )
        propagate_forward_pass_cache(token_choice_layer)

        with torch.no_grad():
            # make sure the gating is the same for both experts
            token_choice_layer.router.gate.data[
                :, 0
            ] = token_choice_layer.router.gate.data[:, 1]

            # copy weights from experts to layer
            token_choice_layer.expert_inner_function.lin1_weight.data[0] = (
                expert1[0].weight.data.transpose(0, 1).unsqueeze(0)
            )
            token_choice_layer.expert_inner_function.lin1_weight.data[1] = (
                expert2[0].weight.data.transpose(0, 1).unsqueeze(0)
            )
            token_choice_layer.expert_inner_function.lin2_weight.data[0] = (
                expert1[2].weight.data.transpose(0, 1).unsqueeze(0)
            )
            token_choice_layer.expert_inner_function.lin2_weight.data[1] = (
                expert2[2].weight.data.transpose(0, 1).unsqueeze(0)
            )
        # make sure weights act the same
        input_data = torch.rand((batch, seql, dm))

        # because scores are the same, the output is the average of the two experts
        output_lin = (expert1(input_data) + expert2(input_data)) / 2
        output_token_choice = token_choice_layer(input_data)

        self.assertTensorAlmostEqual(output_lin, output_token_choice)

        # backprop and make sure gradients are the same
        output_lin.sum().backward()
        output_token_choice.sum().backward()
        self.assertTensorAlmostEqual(
            expert1[0].weight.grad,
            token_choice_layer.expert_inner_function.lin1_weight.grad[0]
            .squeeze(0)
            .transpose(0, 1),
        )
        self.assertTensorAlmostEqual(
            expert2[0].weight.grad,
            token_choice_layer.expert_inner_function.lin1_weight.grad[1]
            .squeeze(0)
            .transpose(0, 1),
        )
        self.assertTensorAlmostEqual(
            expert1[2].weight.grad,
            token_choice_layer.expert_inner_function.lin2_weight.grad[0]
            .squeeze(0)
            .transpose(0, 1),
        )
        self.assertTensorAlmostEqual(
            expert2[2].weight.grad,
            token_choice_layer.expert_inner_function.lin2_weight.grad[1]
            .squeeze(0)
            .transpose(0, 1),
        )
