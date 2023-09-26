import torch
import research.conditional.moe_layers.cont_moe_designs.add_layernorms
import research.conditional.moe_layers.cont_moe_designs.learn_temp_and_common_base
import research.conditional.moe_layers.cont_moe_designs.merge_without_weights
import research.conditional.moe_layers.cont_moe_designs.no_softmax_on_weights
import research.conditional.moe_layers.cont_moe_designs.random_grouping
import research.conditional.moe_layers.cont_moe_designs.send_result_only_to_top1_token
import research.conditional.moe_layers.cont_moe_designs.separate_merge_emit_weights
import research.conditional.moe_layers.cont_moe_designs.separate_merge_emit_weights_common_base
import research.conditional.moe_layers.continuous_moe
import research.conditional.moe_layers.ffs
import research.conditional.moe_layers.cont_moe_designs.learnable_temperature
import research.conditional.moe_layers.cont_moe_designs.learnable_temperature_positive
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
        layer = research.conditional.moe_layers.continuous_moe.ContinuousMoE(
            dm,
            dff,
            n_experts=4,
            group_size=4,
            sparsity_dim=0,
            temperature=1.0,
            expert_size=8,
        )
        shape_and_parameters(layer)

    def test_dim1(self):
        layer = research.conditional.moe_layers.continuous_moe.ContinuousMoE(
            dm,
            dff,
            n_experts=4,
            group_size=4,
            sparsity_dim=1,
            temperature=1.0,
            expert_size=8,
        )
        shape_and_parameters(layer)


class TestContinuousMoEQuick(GeneralTestCase):
    def test_basic(self):
        layer = research.conditional.moe_layers.continuous_moe.ContinuousMoE(
            dm,
            dff,
            n_experts=4,
            group_size=4,
            sparsity_dim=0,
            temperature=1.0,
            expert_size=8,
            use_opt_einsum=True,
        )
        shape_and_parameters(layer)

    def test_dim1(self):
        layer = research.conditional.moe_layers.continuous_moe.ContinuousMoE(
            dm,
            dff,
            n_experts=4,
            group_size=4,
            sparsity_dim=1,
            temperature=1.0,
            expert_size=8,
            use_opt_einsum=True,
        )
        shape_and_parameters(layer)


class TestContinuousMoEQuickMergeDifferentlySimple(GeneralTestCase):
    def test_basic(self):
        layer = research.conditional.moe_layers.cont_moe_designs.separate_merge_emit_weights.ContinuousMoEMergeDifferentlySimple(
            dm,
            dff,
            n_experts=4,
            group_size=4,
            sparsity_dim=0,
            temperature=1.0,
            expert_size=8,
            use_opt_einsum=True,
        )
        shape_and_parameters(layer)

    def test_dim1(self):
        layer = research.conditional.moe_layers.cont_moe_designs.separate_merge_emit_weights.ContinuousMoEMergeDifferentlySimple(
            dm,
            dff,
            n_experts=4,
            group_size=4,
            sparsity_dim=1,
            temperature=1.0,
            expert_size=8,
            use_opt_einsum=True,
        )
        shape_and_parameters(layer)


class ContinuousMoEQuickMergeDifferentlyCommonBase(GeneralTestCase):
    def test_basic(self):
        layer = research.conditional.moe_layers.cont_moe_designs.separate_merge_emit_weights_common_base.ContinuousMoEMergeDifferentlyCommonBase(
            dm,
            dff,
            n_experts=4,
            group_size=4,
            sparsity_dim=0,
            temperature=1.0,
            expert_size=8,
            use_opt_einsum=True,
        )
        shape_and_parameters(layer)

    def test_dim1(self):
        layer = research.conditional.moe_layers.cont_moe_designs.separate_merge_emit_weights_common_base.ContinuousMoEMergeDifferentlyCommonBase(
            dm,
            dff,
            n_experts=4,
            group_size=4,
            sparsity_dim=1,
            temperature=1.0,
            expert_size=8,
            use_opt_einsum=True,
        )
        shape_and_parameters(layer)


class ContinuousMoEQuickRawmerge(GeneralTestCase):
    def test_basic(self):
        layer = research.conditional.moe_layers.cont_moe_designs.merge_without_weights.ContinuousMoERawmerge(
            dm,
            dff,
            n_experts=4,
            group_size=4,
            sparsity_dim=0,
            temperature=1.0,
            expert_size=8,
            use_opt_einsum=True,
        )
        shape_and_parameters(layer)

    def test_dim1(self):
        layer = research.conditional.moe_layers.cont_moe_designs.merge_without_weights.ContinuousMoERawmerge(
            dm,
            dff,
            n_experts=4,
            group_size=4,
            sparsity_dim=1,
            temperature=1.0,
            expert_size=8,
            use_opt_einsum=True,
        )
        shape_and_parameters(layer)


class ContinuousMoEQuickTopmerge(GeneralTestCase):
    def test_basic(self):
        layer = research.conditional.moe_layers.cont_moe_designs.send_result_only_to_top1_token.ContinuousMoETopmerge(
            dm,
            dff,
            n_experts=4,
            group_size=4,
            sparsity_dim=0,
            temperature=1.0,
            expert_size=8,
            use_opt_einsum=True,
        )
        shape_and_parameters(layer)

    def test_dim1(self):
        layer = research.conditional.moe_layers.cont_moe_designs.send_result_only_to_top1_token.ContinuousMoETopmerge(
            dm,
            dff,
            n_experts=4,
            group_size=4,
            sparsity_dim=1,
            temperature=1.0,
            expert_size=8,
            use_opt_einsum=True,
        )
        shape_and_parameters(layer)


class ContinuousMoEQuickNosoftmax(GeneralTestCase):
    def test_basic(self):
        layer = research.conditional.moe_layers.cont_moe_designs.no_softmax_on_weights.ContinuousMoENosoftmax(
            dm,
            dff,
            n_experts=4,
            group_size=4,
            sparsity_dim=0,
            temperature=1.0,
            expert_size=8,
            use_opt_einsum=True,
        )
        shape_and_parameters(layer)

    def test_dim1(self):
        layer = research.conditional.moe_layers.cont_moe_designs.no_softmax_on_weights.ContinuousMoENosoftmax(
            dm,
            dff,
            n_experts=4,
            group_size=4,
            sparsity_dim=1,
            temperature=1.0,
            expert_size=8,
            use_opt_einsum=True,
        )
        shape_and_parameters(layer)


class ContinuousMoEQuickAdaTemp(GeneralTestCase):
    def test_basic(self):
        layer = research.conditional.moe_layers.cont_moe_designs.learnable_temperature.ContinuousMoEAdaTemp(
            dm,
            dff,
            n_experts=4,
            group_size=4,
            sparsity_dim=0,
            temperature=1.0,
            expert_size=8,
            use_opt_einsum=True,
            separate_temp_for_emit_merge=True,
            separate_temp_for_experts=True,
        )
        shape_and_parameters(layer)

    def test_dim1(self):
        layer = research.conditional.moe_layers.cont_moe_designs.learnable_temperature.ContinuousMoEAdaTemp(
            dm,
            dff,
            n_experts=4,
            group_size=4,
            sparsity_dim=1,
            temperature=1.0,
            expert_size=8,
            use_opt_einsum=True,
            separate_temp_for_emit_merge=False,
            separate_temp_for_experts=False,
        )
        shape_and_parameters(layer)

    def test_single_temp(self):
        layer = research.conditional.moe_layers.cont_moe_designs.learnable_temperature.ContinuousMoEAdaTemp(
            dm,
            dff,
            n_experts=16,
            group_size=4,
            sparsity_dim=1,
            temperature=1.0,
            expert_size=8,
            use_opt_einsum=True,
            separate_temp_for_emit_merge=False,
            separate_temp_for_experts=False,
        )
        shape_and_parameters(layer)


class ContinuousMoELayernorm(GeneralTestCase):
    def test_basic(self):
        layer = research.conditional.moe_layers.cont_moe_designs.add_layernorms.ContinuousMoELayernorm(
            dm,
            dff,
            n_experts=4,
            group_size=4,
            sparsity_dim=0,
            temperature=1.0,
            expert_size=8,
            use_opt_einsum=True,
        )
        shape_and_parameters(layer)

    def test_dim1(self):
        layer = research.conditional.moe_layers.cont_moe_designs.add_layernorms.ContinuousMoELayernorm(
            dm,
            dff,
            n_experts=4,
            group_size=4,
            sparsity_dim=1,
            temperature=1.0,
            expert_size=8,
            use_opt_einsum=True,
        )
        shape_and_parameters(layer)


class ContinuousMoEFinal(GeneralTestCase):
    def test_basic(self):
        layer = research.conditional.moe_layers.cont_moe_designs.learn_temp_and_common_base.ContinuousMoEFinal(
            dm,
            dff,
            n_experts=4,
            group_size=4,
            sparsity_dim=0,
            temperature=1.0,
            expert_size=8,
            use_opt_einsum=True,
            share_by_experts=True,
            share_by_emit_merge=True,
        )
        shape_and_parameters(layer)

    def test_dim1(self):
        layer = research.conditional.moe_layers.cont_moe_designs.learn_temp_and_common_base.ContinuousMoEFinal(
            dm,
            dff,
            n_experts=4,
            group_size=4,
            sparsity_dim=1,
            temperature=1.0,
            expert_size=8,
            use_opt_einsum=True,
            share_by_experts=True,
            share_by_emit_merge=True,
        )
        shape_and_parameters(layer)

    def test_single_temp(self):
        layer = research.conditional.moe_layers.cont_moe_designs.learnable_temperature.ContinuousMoEAdaTemp(
            dm,
            dff,
            n_experts=16,
            group_size=4,
            sparsity_dim=1,
            temperature=1.0,
            expert_size=8,
            use_opt_einsum=True,
            separate_temp_for_emit_merge=False,
            separate_temp_for_experts=False,
        )
        shape_and_parameters(layer)


class ContinuousMoERandomGroups(GeneralTestCase):
    def test_basic(self):
        layer = research.conditional.moe_layers.cont_moe_designs.random_grouping.ContinuousMoERandomGroups(
            dm,
            dff,
            n_experts=4,
            group_size=4,
            sparsity_dim=0,
            temperature=1.0,
            expert_size=8,
            use_opt_einsum=True,
            batch_size=16,
            seqlen=16,
            mix_whole_batch=False,
        )
        shape_and_parameters(layer)

    def whole_batch(self):
        layer = research.conditional.moe_layers.cont_moe_designs.random_grouping.ContinuousMoERandomGroups(
            dm,
            dff,
            n_experts=4,
            group_size=4,
            sparsity_dim=0,
            temperature=1.0,
            expert_size=8,
            use_opt_einsum=True,
            batch_size=16,
            seqlen=16,
            mix_whole_batch=True,
        )
        shape_and_parameters(layer)
