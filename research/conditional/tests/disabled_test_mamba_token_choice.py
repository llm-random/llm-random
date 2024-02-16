from research.conditional.moe_layers.mamba_token_choice import (
    MambaRouter,
    LinearExpertsMamba,
    MambaTokenChoice,
)
import torch
from lizrd.support.test_utils import GeneralTestCase
from lizrd.core.misc import propagate_forward_pass_cache
from mamba_ssm import Mamba


class TestMambaTokenChoice(GeneralTestCase):
    def test_mamba_router(self):
        dmodel = 10
        num_experts = [3, 3]
        batch_size = 2
        length = 5
        capacity_factor = 5.0
        load_balancing_loss_weight = 0.0
        routing_groups = [["input", "gate"], ["output"]]
        router = MambaRouter(
            dinput=dmodel,
            n_experts_per_group=num_experts,
            capacity_factor=capacity_factor,
            load_balancing_loss_weight=load_balancing_loss_weight,
            routing_groups=routing_groups,
            init_type="truncated_normal",
            init_scale=1.0,
        )
        propagate_forward_pass_cache(router)
        x = torch.randn(batch_size, length, dmodel)
        y1 = router.make_routing_params_for_module(x, "input")
        y2 = router.make_routing_params_for_module(x, "gate")
        y3 = router.make_routing_params_for_module(x, "output")
        for i, (a, b) in enumerate(zip(y1, y2)):
            if i < 3:
                self.assertEqual(a.shape, b.shape)
                self.assertTensorAlmostEqual(a, b)
            else:
                self.assertEqual(a, b)

        self.assertEqual(len(y1), 4)
        self.assertEqual(len(y3), 4)

        for i, (a, b) in enumerate(zip(y1, y3)):
            if i < 3:
                self.assertEqual(a.shape, b.shape)
                self.assertFalse(torch.allclose(a, b))
            else:
                self.assertEqual(a, b)

    def test_inner_forward_linear_mamba(self):
        experts = LinearExpertsMamba(1, 10, 5, 3, "truncated_normal", 1.0)
        expert_inputs = torch.randn(3, 5, 10)
        dropped_tokens = torch.randn(5, 10)
        experts_output, dropped_output = experts._inner_forward(
            expert_inputs, dropped_tokens
        )
        self.assertEqual(experts_output.shape, (3, 5, 5))
        self.assertEqual(dropped_output.shape, (5, 5))

    def test_one_expert_equvalence_linear_mamba(self):
        experts = LinearExpertsMamba(10, 10, 5, 1, "truncated_normal", 1.0)
        router = MambaRouter(
            dinput=10,
            n_experts_per_group=[1],
            capacity_factor=1.0,
            load_balancing_loss_weight=0.0,
            routing_groups=[["input"]],
            init_type="truncated_normal",
            init_scale=1.0,
        )
        propagate_forward_pass_cache(router)
        x = torch.randn(4, 10, 10)  # batch_size, length, dmodel
        routing_params = router.make_routing_params_for_module(x, "input")
        routed_tokens_with_all_params = router.route_according_to_params(
            x, "input", routing_params
        )

        output = experts.forward(*routed_tokens_with_all_params)
        self.assertEqual(output.shape, (4, 10, 5))
        self.assertTensorAlmostEqual(output, x @ experts.lin_experts)

    def test_mamba_equivalence(self):
        dinput = 16
        expansion_factor = 2
        n_experts_per_group = [1, 1]
        capacity_factor = 1.0
        seq_len = 8
        input_experts = LinearExpertsMamba(
            seq_len,
            dinput,
            dinput * expansion_factor,
            n_experts_per_group[0],
            "truncated_normal",
            1.0,
        )
        output_experts = LinearExpertsMamba(
            seq_len,
            dinput * expansion_factor,
            dinput,
            n_experts_per_group[0],
            "truncated_normal",
            1.0,
        )
        gate_experts = LinearExpertsMamba(
            seq_len,
            dinput,
            dinput * expansion_factor,
            n_experts_per_group[1],
            "truncated_normal",
            1.0,
        )

        router = MambaRouter(
            dinput=dinput,
            n_experts_per_group=n_experts_per_group,
            capacity_factor=capacity_factor,
            load_balancing_loss_weight=0.0,
            routing_groups=[["input", "output"], "gate"],
            init_type="truncated_normal",
            init_scale=1.0,
        )
        full_mamba = MambaTokenChoice(
            dinput, input_experts, gate_experts, output_experts, router
        )
        propagate_forward_pass_cache(full_mamba)
        reference_mamba = Mamba(
            dinput,
            use_fast_path=False,
        )
        with torch.no_grad():
            reference_mamba.in_proj.weight[
                : dinput * expansion_factor
            ] = input_experts.lin_experts[0].T
            reference_mamba.in_proj.weight[
                dinput * expansion_factor :
            ] = gate_experts.lin_experts[0].T
            reference_mamba.out_proj.weight[:] = output_experts.lin_experts[0].T
            reference_mamba.conv1d.weight = full_mamba.conv1d.weight
            reference_mamba.conv1d.bias = full_mamba.conv1d.bias
            reference_mamba.x_proj = full_mamba.x_proj
            reference_mamba.dt_proj = full_mamba.dt_proj
        propagate_forward_pass_cache(router)
        full_mamba.cuda()
        reference_mamba.cuda()
        x = torch.randn(16, seq_len, dinput).cuda()
        y1 = full_mamba(x)
        y2 = reference_mamba(x)
        y2 = torch.Tensor.cpu(y2)
        y1 = torch.Tensor.cpu(y1)
        self.assertTensorAlmostEqual(y1, y2)
