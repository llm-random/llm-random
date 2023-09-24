import math

import numpy as np
from torch.optim import AdamW
from torch.nn import Linear

from lizrd.support.test_utils import GeneralTestCase
from lizrd.train.scheduler import (
    ConstantScheduler,
    CosineScheduler,
    CosineTempScheduler,
    LinearTempScheduler,
)
from research.conditional.moe_layers.continuous_moe import ContinuousMoE


class TestSchedulers(GeneralTestCase):
    def test_constant_scheduler(self):
        scheduler = ConstantScheduler(lr_warmup_steps=10, lr=0.1)
        model = Linear(1, 1)
        optim = AdamW(model.parameters())

        for step in range(100):
            scheduler.set_lr(optim, step)

            if step < 10:
                for param_group in optim.param_groups:
                    assert math.isclose(
                        optim.param_groups[0]["lr"], 0.01 * (step + 1), abs_tol=1e-6
                    )

            if step == 0:
                assert math.isclose(optim.param_groups[0]["lr"], 0.1 / 10, abs_tol=1e-6)
            elif step == 9:
                assert math.isclose(optim.param_groups[0]["lr"], 0.1, abs_tol=1e-6)
            elif step == 55:
                assert math.isclose(optim.param_groups[0]["lr"], 0.1, abs_tol=1e-6)
            elif step == 99:
                assert math.isclose(optim.param_groups[0]["lr"], 0.1, abs_tol=1e-6)

    def test_cosine_scheduler(self):
        scheduler = CosineScheduler(
            lr_warmup_steps=10, lr=0.1, final_lr_step=90, final_lr_fraction=0.1
        )
        model = Linear(1, 1)
        optim = AdamW(model.parameters())

        for step in range(100):
            scheduler.set_lr(optim, step)

            for param_group in optim.param_groups:
                assert param_group["lr"] == scheduler.get_lr(step)

            if step == 0:
                assert math.isclose(optim.param_groups[0]["lr"], 0.1 / 10, abs_tol=1e-6)
            elif step == 9:
                assert math.isclose(optim.param_groups[0]["lr"], 0.1, abs_tol=1e-6)
            elif step == 50:
                assert math.isclose(
                    optim.param_groups[0]["lr"], 0.01 + (0.1 - 0.01) / 2, abs_tol=1e-6
                )  # this is half of the distance between high and low lr
            elif step == 90:
                assert math.isclose(optim.param_groups[0]["lr"], 0.01, abs_tol=1e-6)
            elif step == 99:
                assert math.isclose(optim.param_groups[0]["lr"], 0.01, abs_tol=1e-6)

    def test_linear_temp_scheduler(self):
        model = ContinuousMoE(1, 1, 1, 1, 1, 1.0, None)

        scheduler = LinearTempScheduler(
            model,
            steps_until_start_temperature_anneal=10,
            steps_until_finish_temperature_anneal=90,
        )

        # Test model temperature before annealing begins
        assert np.isclose(model.temperature, 1.0)
        for i in range(100):
            scheduler.step(i)
            if i < 10:
                assert np.isclose(model.temperature, 1.0)
            elif i == 50:
                assert np.isclose(model.temperature, 0.5)
            elif i == 90:
                assert np.isclose(model.temperature, 0.0)
            elif i == 99:
                assert np.isclose(model.temperature, 0.0)

    def test_cosine_temp_scheduler(self):
        model = ContinuousMoE(1, 1, 1, 1, 1, 1.0, None)

        scheduler = CosineTempScheduler(
            model,
            steps_until_start_temperature_anneal=10,
            steps_until_finish_temperature_anneal=90,
        )

        # Test model temperature before annealing begins
        assert np.isclose(model.temperature, 1.0)
        for i in range(100):
            scheduler.step(i)
            if i < 10:
                assert np.isclose(model.temperature, 1.0)
            elif i == 50:
                assert np.isclose(model.temperature, 0.5)
            elif i == 70:
                assert np.isclose(
                    model.temperature, 0.1464466094067263
                ), model.temperature
            elif i == 90:
                assert np.isclose(model.temperature, 0.0)
            elif i == 99:
                assert np.isclose(model.temperature, 0.0)
