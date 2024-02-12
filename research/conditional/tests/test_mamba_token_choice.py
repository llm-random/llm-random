from research.conditional.moe_layers.mamba_token_choice import (
    MambaRouter,
    LinearExpertsMamba,
    NoExpertsMamba,
)
import torch
from lizrd.support.test_utils import GeneralTestCase
from lizrd.core.misc import propagate_forward_pass_cache


class TestMambaTokenChoice(GeneralTestCase):
    def test_mamba_router(self):
        dmodel = 10
        num_experts = 3
        batch_size = 2
        length = 5
        capacity_factor = 5.0
        load_balancing_loss_weight = 0.0
        routing_groups = [["input", "gate"], ["output"]]
        router = MambaRouter(
            dmodel,
            num_experts,
            capacity_factor,
            load_balancing_loss_weight,
            routing_groups,
            init_type="truncated_normal",
            init_scale=1.0,
        )
        propagate_forward_pass_cache(router)
        x = torch.randn(batch_size, length, dmodel)
        y1, y2, y3 = router(x)
        self.assertEqual(y1, y2)

        self.assertEqual(len(y1), 4)
        self.assertEqual(len(y3), 4)
        print(y1[0])

        for i, (a, b) in enumerate(zip(y1, y3)):
            self.assertEqual(a.shape, b.shape)
            if i < 2:
                self.assertFalse(torch.allclose(a, b))
            else:
                self.assertTrue(torch.allclose(a, b))

    def test_linear_experts_mamba(self):
        experts = LinearExpertsMamba(10, 5, 3, "truncated_normal", 1.0)
        x1 = torch.randn(3, 5, 10)
        x2 = torch.randn(5, 10)
        y1, y2 = experts._inner_forward(x1, x2)
        self.assertEqual(y1.shape, (3, 5, 5))
        self.assertEqual(y2.shape, (5, 5))

    # def test_no_experts_mamba(self):
    #     experts = NoExpertsMamba(10, 5, 3, "truncated_normal", 1.0)
    #     x = torch.randn(2, 10, 5)
    #     y = experts(x)
    #     self.assertEqual(y.shape, (2, 10, 3))
