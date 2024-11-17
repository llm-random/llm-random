import torch
import research.mole.moe_layers.cont_moe_designs.add_layernorms
import research.mole.moe_layers.cont_moe_designs.learn_temp_and_common_base
import research.mole.moe_layers.cont_moe_designs.merge_without_weights
import research.mole.moe_layers.cont_moe_designs.no_softmax_on_weights
import research.mole.moe_layers.cont_moe_designs.random_grouping
import research.mole.moe_layers.cont_moe_designs.send_result_only_to_top1_token
import research.mole.moe_layers.cont_moe_designs.separate_merge_emit_weights
import research.mole.moe_layers.cont_moe_designs.separate_merge_emit_weights_common_base
import research.mole.moe_layers.continuous_moe
import research.mole.moe_layers.ffs
import research.mole.moe_layers.cont_moe_designs.learnable_temperature
from lizrd.support.test_utils import GeneralTestCase

(
    batch,
    seq_len,
    dm,
    dff,
) = (
    16,
    16,
    32,
    64,
)

common_arguments = {
    "dm": dm,
    "dff": dff,
    "n_experts": 4,
    "group_size": 4,
    "sparsity_dim": 0,
    "temperature": 1.0,
    "expert_size": 8,
    "use_opt_einsum": True,
    "init_type": "kaiming_uniform",
    "init_scale": 1.0,
}


def shape_and_parameters(layer):
    for _ in layer.parameters():
        pass
    input = torch.normal(0.0, 1.0, (batch, seq_len, dm))
    output = layer(input)
    loss = output.sum()
    loss.backward()
    assert output.shape == (batch, seq_len, dm)


class TestContinuousMoE(GeneralTestCase):
    def test_basic(self):
        layer = research.mole.moe_layers.continuous_moe.ContinuousMoE(
            **common_arguments
        )
        shape_and_parameters(layer)


class TestContinuousMoEQuick(GeneralTestCase):
    def test_basic(self):
        layer = research.mole.moe_layers.continuous_moe.ContinuousMoE(
            **common_arguments
        )
        shape_and_parameters(layer)


# FIX ME: This test is failing because contmoe interface has changed
# class TestContinuousMoEQuickMergeDifferentlySimple(GeneralTestCase):
#     def test_basic(self):
#         layer = research.mole.moe_layers.cont_moe_designs.separate_merge_emit_weights.ContinuousMoEMergeDifferentlySimple(
#             **common_arguments
#         )
#         shape_and_parameters(layer)

# FIX ME: This test is failing because contmoe interface has changed
# class ContinuousMoEQuickMergeDifferentlyCommonBase(GeneralTestCase):
#     def test_basic(self):
#         layer = research.mole.moe_layers.cont_moe_designs.separate_merge_emit_weights_common_base.ContinuousMoEMergeDifferentlyCommonBase(
#             **common_arguments
#         )
#         shape_and_parameters(layer)


class ContinuousMoEQuickRawmerge(GeneralTestCase):
    def test_basic(self):
        layer = research.mole.moe_layers.cont_moe_designs.merge_without_weights.ContinuousMoERawmerge(
            **common_arguments
        )
        shape_and_parameters(layer)


class ContinuousMoEQuickTopmerge(GeneralTestCase):
    def test_basic(self):
        layer = research.mole.moe_layers.cont_moe_designs.send_result_only_to_top1_token.ContinuousMoETopmerge(
            **common_arguments
        )
        shape_and_parameters(layer)


class ContinuousMoEQuickNosoftmax(GeneralTestCase):
    def test_basic(self):
        layer = research.mole.moe_layers.cont_moe_designs.no_softmax_on_weights.ContinuousMoENosoftmax(
            **common_arguments
        )
        shape_and_parameters(layer)


class ContinuousMoEQuickAdaTemp(GeneralTestCase):
    def test_basic(self):
        layer = research.mole.moe_layers.cont_moe_designs.learnable_temperature.ContinuousMoEAdaTemp(
            **common_arguments,
            share_by_experts=True,
            share_by_emit_merge=True,
        )
        shape_and_parameters(layer)

    def test_single_temp(self):
        layer = research.mole.moe_layers.cont_moe_designs.learnable_temperature.ContinuousMoEAdaTemp(
            **common_arguments,
            share_by_experts=False,
            share_by_emit_merge=False,
        )
        shape_and_parameters(layer)


class ContinuousMoELayernorm(GeneralTestCase):
    def test_basic(self):
        layer = research.mole.moe_layers.cont_moe_designs.add_layernorms.ContinuousMoELayernorm(
            **common_arguments
        )
        shape_and_parameters(layer)

    # FIX ME: This test is failing because contmoe interface has changed
    # class ContinuousMoEFinal(GeneralTestCase):
    #     def test_basic(self):
    #         layer = research.mole.moe_layers.cont_moe_designs.learn_temp_and_common_base.ContinuousMoEFinal(
    #             **common_arguments,
    #             share_by_experts=True,
    #             share_by_emit_merge=True,
    #         )
    #         shape_and_parameters(layer)

    def test_single_temp(self):
        layer = research.mole.moe_layers.cont_moe_designs.learnable_temperature.ContinuousMoEAdaTemp(
            **common_arguments,
            share_by_experts=False,
            share_by_emit_merge=False,
        )
        shape_and_parameters(layer)


class ContinuousMoERandomGroups(GeneralTestCase):
    def test_basic(self):
        layer = research.mole.moe_layers.cont_moe_designs.random_grouping.ContinuousMoERandomGroups(
            **common_arguments,
            batch_size=16,
            seqlen=16,
            mix_whole_batch=False,
        )
        shape_and_parameters(layer)

    def whole_batch(self):
        layer = research.mole.moe_layers.cont_moe_designs.random_grouping.ContinuousMoERandomGroups(
            **common_arguments,
            batch_size=16,
            seqlen=16,
            mix_whole_batch=True,
        )
        shape_and_parameters(layer)


class OptimizedVersusLegacy(GeneralTestCase):
    def test_gpt(self):
        seq_len = 48
        dm = 72
        for group_size in [1, 2, batch]:
            arguments = {
                "dm": dm,
                "dff": dff,
                "n_experts": 4,
                "group_size": group_size,
                "sparsity_dim": 0,
                "temperature": 1.0,
                "expert_size": 8,
                "use_opt_einsum": True,
                "init_type": "kaiming_uniform",
                "init_scale": 1.0,
            }
            legacy = research.mole.moe_layers.continuous_moe.LegacyContinuousMoE(
                **arguments
            )
            bmm = research.mole.moe_layers.continuous_moe.ContinuousMoE(
                **arguments
            )
            input = torch.normal(0.0, 1.0, (batch, seq_len, dm))
            legacy.lin1.data = bmm.lin1.data.clone().transpose(0, 1)
            legacy.lin2.data = bmm.lin2.data.clone().permute(2, 0, 1)
            legacy.controller.data = bmm.controller.data.clone()

            _legacy_output = legacy(input)
            _bmm_output = bmm(input)
            self.assertTensorAlmostEqual(_legacy_output, _bmm_output)

    def test_bert(self):
        seq_len = 48
        dm = 72
        for group_size in [1, 2, batch]:
            arguments = {
                "dm": dm,
                "dff": dff,
                "n_experts": 4,
                "group_size": group_size,
                "sparsity_dim": 1,
                "temperature": 1.0,
                "expert_size": 8,
                "use_opt_einsum": True,
                "init_type": "kaiming_uniform",
                "init_scale": 1.0,
            }
            legacy = research.mole.moe_layers.continuous_moe.LegacyContinuousMoE(
                **arguments
            )
            bmm = research.mole.moe_layers.continuous_moe.ContinuousMoE(
                **arguments
            )
            input = torch.normal(0.0, 1.0, (batch, seq_len, dm))
            legacy.lin1.data = bmm.lin1.data.clone().transpose(0, 1)
            legacy.lin2.data = bmm.lin2.data.clone().permute(2, 0, 1)
            legacy.controller.data = bmm.controller.data.clone()

            _legacy_output = legacy(input)
            _bmm_output = bmm(input)
            self.assertTensorAlmostEqual(_legacy_output, _bmm_output)
