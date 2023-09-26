from collections import OrderedDict

from torch import nn

from lizrd.support.test_utils import GeneralTestCase
from research.conditional.moe_layers.cont_moe_designs.learnable_temperature_positive import (
    ContinuousMoEAdaTempPositive,
)
from research.conditional.utils.layer_manager import LayerManager


class TestLearningStartAdatemp(GeneralTestCase):
    module_list = OrderedDict(
        [
            (
                f"block_{i}_feedforward",
                ContinuousMoEAdaTempPositive(1, 1, 1, 1, 1, 1.0, None),
            )
            for i in range(10)
        ]
    )
    model = nn.Sequential(module_list)

    layer_manager = LayerManager(model, 0, 0)

    extra_info = {"steps_until_start_temperature_learn": 10}

    for step in range(100):
        layer_manager.manage_misc(step, **extra_info)
        if step == 50:
            for name, module in model.named_children():
                if name.endswith("feedforward"):
                    assert module.temperature_emit.requires_grad
                    assert module.temperature_merge.requires_grad
