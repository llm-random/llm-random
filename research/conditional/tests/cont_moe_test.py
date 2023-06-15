import torch

import research.conditional.moe_layers.continuous_moe
import research.conditional.moe_layers.ffs
from lizrd.support.test_utils import GeneralTestCase


class TestContinuousMoE(GeneralTestCase):
    def test_basic(self):
        (
            batch,
            seq_len,
            dm,
            dff,
        ) = (
            4,
            10,
            32,
            64,
        )
        layer = research.conditional.moe_layers.continuous_moe.ContinuousMoE(
            dm,
            dff,
            n_experts=4,
            group_size=4,
            sparsity_dim=0,
            temperature=1.0,
            expert_size=8,
        )
        input = torch.normal(0.0, 1.0, (batch, seq_len, dm))
        output = layer(input)
        loss = output.sum()
        loss.backward()
        self.assertShape(output, (batch, seq_len, dm))

    def test_dim1(self):
        (
            batch,
            seq_len,
            dm,
            dff,
        ) = (
            5,
            12,
            32,
            64,
        )
        layer = research.conditional.moe_layers.continuous_moe.ContinuousMoE(
            dm,
            dff,
            n_experts=4,
            group_size=4,
            sparsity_dim=1,
            temperature=1.0,
            expert_size=8,
        )
        input = torch.normal(0.0, 1.0, (batch, seq_len, dm))
        output = layer(input)
        loss = output.sum()
        loss.backward()
        self.assertShape(output, (batch, seq_len, dm))


class TestContinuousMoEQuick(GeneralTestCase):
    def test_basic(self):
        (
            batch,
            seq_len,
            dm,
            dff,
        ) = (
            4,
            10,
            32,
            64,
        )
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
        input = torch.normal(0.0, 1.0, (batch, seq_len, dm))
        output = layer(input)
        loss = output.sum()
        loss.backward()
        self.assertShape(output, (batch, seq_len, dm))

    def test_dim1(self):
        (
            batch,
            seq_len,
            dm,
            dff,
        ) = (
            5,
            12,
            32,
            64,
        )
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
        input = torch.normal(0.0, 1.0, (batch, seq_len, dm))
        output = layer(input)
        loss = output.sum()
        loss.backward()
        self.assertShape(output, (batch, seq_len, dm))


class TestContinuousMoEQuickMergeDifferentlySimple(GeneralTestCase):
    def test_basic(self):
        (
            batch,
            seq_len,
            dm,
            dff,
        ) = (
            4,
            10,
            32,
            64,
        )
        layer = research.conditional.moe_layers.continuous_moe.ContinuousMoEQuickMergeDifferentlySimple(
            dm,
            dff,
            n_experts=4,
            group_size=4,
            sparsity_dim=0,
            temperature=1.0,
            expert_size=8,
            use_opt_einsum=True,
        )
        input = torch.normal(0.0, 1.0, (batch, seq_len, dm))
        output = layer(input)
        loss = output.sum()
        loss.backward()
        self.assertShape(output, (batch, seq_len, dm))

    def test_dim1(self):
        (
            batch,
            seq_len,
            dm,
            dff,
        ) = (
            5,
            12,
            32,
            64,
        )
        layer = research.conditional.moe_layers.continuous_moe.ContinuousMoEQuickMergeDifferentlySimple(
            dm,
            dff,
            n_experts=4,
            group_size=4,
            sparsity_dim=1,
            temperature=1.0,
            expert_size=8,
            use_opt_einsum=True,
        )
        input = torch.normal(0.0, 1.0, (batch, seq_len, dm))
        output = layer(input)
        loss = output.sum()
        loss.backward()
        self.assertShape(output, (batch, seq_len, dm))


class ContinuousMoEQuickMergeDifferentlyCommonBase(GeneralTestCase):
    def test_basic(self):
        (
            batch,
            seq_len,
            dm,
            dff,
        ) = (
            4,
            10,
            32,
            64,
        )
        layer = research.conditional.moe_layers.continuous_moe.ContinuousMoEQuickMergeDifferentlyCommonBase(
            dm,
            dff,
            n_experts=4,
            group_size=4,
            sparsity_dim=0,
            temperature=1.0,
            expert_size=8,
            use_opt_einsum=True,
        )
        input = torch.normal(0.0, 1.0, (batch, seq_len, dm))
        output = layer(input)
        loss = output.sum()
        loss.backward()
        self.assertShape(output, (batch, seq_len, dm))

    def test_dim1(self):
        (
            batch,
            seq_len,
            dm,
            dff,
        ) = (
            5,
            12,
            32,
            64,
        )
        layer = research.conditional.moe_layers.continuous_moe.ContinuousMoEQuickMergeDifferentlyCommonBase(
            dm,
            dff,
            n_experts=4,
            group_size=4,
            sparsity_dim=1,
            temperature=1.0,
            expert_size=8,
            use_opt_einsum=True,
        )
        input = torch.normal(0.0, 1.0, (batch, seq_len, dm))
        output = layer(input)
        loss = output.sum()
        loss.backward()
        self.assertShape(output, (batch, seq_len, dm))


class ContinuousMoEQuickRawmerge(GeneralTestCase):
    def test_basic(self):
        (
            batch,
            seq_len,
            dm,
            dff,
        ) = (
            4,
            10,
            32,
            64,
        )
        layer = (
            research.conditional.moe_layers.continuous_moe.ContinuousMoEQuickRawmerge(
                dm,
                dff,
                n_experts=4,
                group_size=4,
                sparsity_dim=0,
                temperature=1.0,
                expert_size=8,
                use_opt_einsum=True,
            )
        )
        input = torch.normal(0.0, 1.0, (batch, seq_len, dm))
        output = layer(input)
        loss = output.sum()
        loss.backward()
        self.assertShape(output, (batch, seq_len, dm))

    def test_dim1(self):
        (
            batch,
            seq_len,
            dm,
            dff,
        ) = (
            5,
            12,
            32,
            64,
        )
        layer = (
            research.conditional.moe_layers.continuous_moe.ContinuousMoEQuickRawmerge(
                dm,
                dff,
                n_experts=4,
                group_size=4,
                sparsity_dim=1,
                temperature=1.0,
                expert_size=8,
                use_opt_einsum=True,
            )
        )
        input = torch.normal(0.0, 1.0, (batch, seq_len, dm))
        output = layer(input)
        loss = output.sum()
        loss.backward()
        self.assertShape(output, (batch, seq_len, dm))


class ContinuousMoEQuickTopmerge(GeneralTestCase):
    def test_basic(self):
        (
            batch,
            seq_len,
            dm,
            dff,
        ) = (
            4,
            10,
            32,
            64,
        )
        layer = (
            research.conditional.moe_layers.continuous_moe.ContinuousMoEQuickTopmerge(
                dm,
                dff,
                n_experts=4,
                group_size=4,
                sparsity_dim=0,
                temperature=1.0,
                expert_size=8,
                use_opt_einsum=True,
            )
        )
        input = torch.normal(0.0, 1.0, (batch, seq_len, dm))
        output = layer(input)
        loss = output.sum()
        loss.backward()
        self.assertShape(output, (batch, seq_len, dm))

    def test_dim1(self):
        (
            batch,
            seq_len,
            dm,
            dff,
        ) = (
            5,
            12,
            32,
            64,
        )
        layer = (
            research.conditional.moe_layers.continuous_moe.ContinuousMoEQuickTopmerge(
                dm,
                dff,
                n_experts=4,
                group_size=4,
                sparsity_dim=1,
                temperature=1.0,
                expert_size=8,
                use_opt_einsum=True,
            )
        )
        input = torch.normal(0.0, 1.0, (batch, seq_len, dm))
        output = layer(input)
        loss = output.sum()
        loss.backward()
        self.assertShape(output, (batch, seq_len, dm))


class ContinuousMoEQuickNosoftmax(GeneralTestCase):
    def test_basic(self):
        (
            batch,
            seq_len,
            dm,
            dff,
        ) = (
            4,
            10,
            32,
            64,
        )
        layer = (
            research.conditional.moe_layers.continuous_moe.ContinuousMoEQuickNosoftmax(
                dm,
                dff,
                n_experts=4,
                group_size=4,
                sparsity_dim=0,
                temperature=1.0,
                expert_size=8,
                use_opt_einsum=True,
            )
        )
        input = torch.normal(0.0, 1.0, (batch, seq_len, dm))
        output = layer(input)
        loss = output.sum()
        loss.backward()
        self.assertShape(output, (batch, seq_len, dm))

    def test_dim1(self):
        (
            batch,
            seq_len,
            dm,
            dff,
        ) = (
            5,
            12,
            32,
            64,
        )
        layer = (
            research.conditional.moe_layers.continuous_moe.ContinuousMoEQuickNosoftmax(
                dm,
                dff,
                n_experts=4,
                group_size=4,
                sparsity_dim=1,
                temperature=1.0,
                expert_size=8,
                use_opt_einsum=True,
            )
        )
        input = torch.normal(0.0, 1.0, (batch, seq_len, dm))
        output = layer(input)
        loss = output.sum()
        loss.backward()
        self.assertShape(output, (batch, seq_len, dm))


class ContinuousMoEQuickAdaTemp(GeneralTestCase):
    def test_basic(self):
        (
            batch,
            seq_len,
            dm,
            dff,
        ) = (
            4,
            10,
            32,
            64,
        )
        layer = (
            research.conditional.moe_layers.continuous_moe.ContinuousMoEQuickAdaTemp(
                dm,
                dff,
                n_experts=4,
                group_size=4,
                sparsity_dim=0,
                temperature=1.0,
                expert_size=8,
                use_opt_einsum=True,
            )
        )
        input = torch.normal(0.0, 1.0, (batch, seq_len, dm))
        output = layer(input)
        loss = output.sum()
        loss.backward()
        self.assertShape(output, (batch, seq_len, dm))

    def test_dim1(self):
        (
            batch,
            seq_len,
            dm,
            dff,
        ) = (
            5,
            12,
            32,
            64,
        )
        layer = (
            research.conditional.moe_layers.continuous_moe.ContinuousMoEQuickAdaTemp(
                dm,
                dff,
                n_experts=4,
                group_size=4,
                sparsity_dim=1,
                temperature=1.0,
                expert_size=8,
                use_opt_einsum=True,
            )
        )
        input = torch.normal(0.0, 1.0, (batch, seq_len, dm))
        output = layer(input)
        loss = output.sum()
        loss.backward()
        self.assertShape(output, (batch, seq_len, dm))
