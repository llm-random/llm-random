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
            expertsize=8,
        )
        input = torch.normal(0.0, 1.0, (batch, seq_len, dm))
        output = layer(input)
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
            expertsize=8,
        )
        input = torch.normal(0.0, 1.0, (batch, seq_len, dm))
        output = layer(input)
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
            expertsize=8,
            use_opt_einsum=True,
        )
        input = torch.normal(0.0, 1.0, (batch, seq_len, dm))
        output = layer(input)
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
            expertsize=8,
            use_opt_einsum=True,
        )
        input = torch.normal(0.0, 1.0, (batch, seq_len, dm))
        output = layer(input)
        self.assertShape(output, (batch, seq_len, dm))
