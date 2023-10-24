from collections import OrderedDict

from torch import nn

from lizrd.support.test_utils import GeneralTestCase
from research.conditional.moe_layers.cont_moe_designs.learnable_temperature import (
    ContinuousMoEAdaTemp,
)
from research.conditional.moe_layers.cont_moe_designs.learnable_temperature_positive import (
    ContinuousMoEAdaTempPositive,
)
from research.conditional.utils.layer_manager import LayerManager


class TestLearningStartAdatemp(GeneralTestCase):
    module_list = OrderedDict(
        [
            (
                f"block_{i}_feedforward",
                ContinuousMoEAdaTemp(1, 1, 1, 1, 1, 1.0, None),
            )
            for i in range(10)
        ]
    )
    model = nn.Sequential(module_list)

    steps_until_start_temperature_learn = 10

    layer_manager = LayerManager(
        model,
        0,
        0,
        steps_until_start_temperature_learn=steps_until_start_temperature_learn,
    )

    for step in range(100):
        layer_manager.manage_learnable_temperature(step)
        for name, module in model.named_children():
            if name.endswith("feedforward"):
                if step == 0 or step == 9:
                    assert not module.temperature_emit.requires_grad
                    assert not module.temperature_merge.requires_grad
                elif step == 11 or step == 99:
                    assert module.temperature_emit.requires_grad
                    assert module.temperature_merge.requires_grad


class TestLearningStartAdatempPositive(GeneralTestCase):
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

    steps_until_start_temperature_learn = 10

    layer_manager = LayerManager(
        model,
        0,
        0,
        steps_until_start_temperature_learn=steps_until_start_temperature_learn,
    )

    for step in range(100):
        layer_manager.manage_learnable_temperature(step)
        for name, module in model.named_children():
            if name.endswith("feedforward"):
                if step == 0 or step == 9:
                    assert not module.temperature_emit.requires_grad
                    assert not module.temperature_merge.requires_grad
                elif step == 11 or step == 99:
                    assert module.temperature_emit.requires_grad
                    assert module.temperature_merge.requires_grad

    model = nn.Sequential(module_list)

    steps_until_start_temperature_learn = 10

    layer_manager = LayerManager(
        model,
        0,
        0,
        steps_until_start_temperature_learn=steps_until_start_temperature_learn,
    )

    for step in range(100):
        layer_manager.manage_learnable_temperature(step)
        for name, module in model.named_children():
            if name.endswith("feedforward"):
                if step == 0 or step == 9:
                    assert not module.temperature_emit.requires_grad
                    assert not module.temperature_merge.requires_grad
                elif step == 11 or step == 99:
                    assert module.temperature_emit.requires_grad
                    assert module.temperature_merge.requires_grad
