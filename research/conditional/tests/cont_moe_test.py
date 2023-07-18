import torch

import research.conditional.archive.continuous_moe_alternatives
import research.conditional.moe_layers.continuous_moe
import research.conditional.moe_layers.ffs
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
    print("input shape: ", input.shape)
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
        layer = research.conditional.moe_layers.continuous_moe.ContinuousMoEQuick(
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
        layer = research.conditional.moe_layers.continuous_moe.ContinuousMoEQuick(
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
        layer = research.conditional.archive.continuous_moe_alternatives.ContinuousMoEQuickMergeDifferentlySimple(
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
        layer = research.conditional.archive.continuous_moe_alternatives.ContinuousMoEQuickMergeDifferentlySimple(
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
        layer = research.conditional.archive.continuous_moe_alternatives.ContinuousMoEQuickMergeDifferentlyCommonBase(
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
        layer = research.conditional.archive.continuous_moe_alternatives.ContinuousMoEQuickMergeDifferentlyCommonBase(
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
        layer = research.conditional.archive.continuous_moe_alternatives.ContinuousMoEQuickRawmerge(
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
        layer = research.conditional.archive.continuous_moe_alternatives.ContinuousMoEQuickRawmerge(
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
        layer = research.conditional.archive.continuous_moe_alternatives.ContinuousMoEQuickTopmerge(
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
        layer = research.conditional.archive.continuous_moe_alternatives.ContinuousMoEQuickTopmerge(
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
        layer = research.conditional.archive.continuous_moe_alternatives.ContinuousMoEQuickNosoftmax(
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
        layer = research.conditional.archive.continuous_moe_alternatives.ContinuousMoEQuickNosoftmax(
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
        layer = research.conditional.archive.continuous_moe_alternatives.ContinuousMoEQuickAdaTemp(
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
        layer = research.conditional.archive.continuous_moe_alternatives.ContinuousMoEQuickAdaTemp(
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
        layer = research.conditional.archive.continuous_moe_alternatives.ContinuousMoEQuickAdaTemp(
            dm,
            dff,
            n_experts=16,
            group_size=4,
            sparsity_dim=1,
            temperature=1.0,
            expert_size=8,
            use_opt_einsum=True,
            share_by_experts=False,
            share_by_emit_merge=False,
        )
        shape_and_parameters(layer)


class ContinuousMoELayernorm(GeneralTestCase):
    def test_basic(self):
        layer = research.conditional.archive.continuous_moe_alternatives.ContinuousMoELayernorm(
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
        layer = research.conditional.archive.continuous_moe_alternatives.ContinuousMoELayernorm(
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
        layer = (
            research.conditional.archive.continuous_moe_alternatives.ContinuousMoEFinal(
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
        )
        shape_and_parameters(layer)

    def test_dim1(self):
        layer = (
            research.conditional.archive.continuous_moe_alternatives.ContinuousMoEFinal(
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
        )
        shape_and_parameters(layer)

    def test_single_temp(self):
        layer = research.conditional.archive.continuous_moe_alternatives.ContinuousMoEQuickAdaTemp(
            dm,
            dff,
            n_experts=16,
            group_size=4,
            sparsity_dim=1,
            temperature=1.0,
            expert_size=8,
            use_opt_einsum=True,
            share_by_experts=False,
            share_by_emit_merge=False,
        )
        shape_and_parameters(layer)


class ContinuousMoERandomGroups(GeneralTestCase):
    def test_basic(self):
        layer = research.conditional.archive.continuous_moe_alternatives.ContinuousMoERandomGroups(
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
        layer = research.conditional.archive.continuous_moe_alternatives.ContinuousMoERandomGroups(
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
