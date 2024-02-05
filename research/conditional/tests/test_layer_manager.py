from collections import OrderedDict

from torch import nn

from src.core.llm import Residual
from src.support.test_utils import GeneralTestCase
from research.conditional.moe_layers.cont_moe_designs.learnable_temperature import (
    ContinuousMoEAdaTemp,
)
from research.conditional.moe_layers.cont_moe_designs.learnable_temperature_positive import (
    ContinuousMoEAdaTempPositive,
)
from research.conditional.utils.layer_manager import LayerManager


class TestLearningStartAdatemp(GeneralTestCase):
    layers = []
    for i in range(10):
        mot = ContinuousMoEAdaTemp(1, 1, 1, 1, 1, 1.0, "kaiming_uniform", 1.0, None)
        residual = Residual(
            nn.Sequential(
                OrderedDict(
                    [
                        ("pre_norm", nn.LayerNorm(1)),
                        ("feedforward", mot),
                    ]
                )
            )
        )
        layers.append((f"block_{i}", residual))
    model = nn.Sequential(OrderedDict(layers))

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
            if isinstance(module, ContinuousMoEAdaTemp):
                if step == 0 or step == 9:
                    assert not module.temperature_emit.requires_grad
                    assert not module.temperature_merge.requires_grad
                elif step == 11 or step == 99:
                    assert module.temperature_emit.requires_grad
                    assert module.temperature_merge.requires_grad


class TestLearningStartAdatempPositive(GeneralTestCase):
    layers = []
    for i in range(10):
        mot = ContinuousMoEAdaTemp(1, 1, 1, 1, 1, 1.0, "kaiming_uniform", 1.0, None)
        residual = Residual(
            nn.Sequential(
                OrderedDict(
                    [
                        ("pre_norm", nn.LayerNorm(1)),
                        ("feedforward", mot),
                    ]
                )
            )
        )
        layers.append((f"block_{i}", residual))
    model = nn.Sequential(OrderedDict(layers))

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
            if isinstance(module, ContinuousMoEAdaTempPositive):
                if step == 0 or step == 9:
                    assert not module.temperature_emit.requires_grad
                    assert not module.temperature_merge.requires_grad
                elif step == 11 or step == 99:
                    assert module.temperature_emit.requires_grad
                    assert module.temperature_merge.requires_grad
