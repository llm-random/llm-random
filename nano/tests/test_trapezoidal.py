import unittest
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

target_lrs = [
    1e-05,
    3.9999999999999996e-05,
    7e-05,
    0.0001,
    0.0001,
    0.0001,
    0.0001,
    0.0001,
    0.0001,
    7.500000000000001e-05,
    5.000000000000001e-05,
    2.5000000000000005e-05,
]
# config_yaml = """
# training:
#     learning_rate: 1e-4
#     scheduler:
#         _target_: model.TrapezoidalSchedulerConfig
#         type: "trapezoidal"
#         warmup_steps: 3
#         constant_steps: 5
#         decay_steps: 4
# """
config_yaml = """
training:
    learning_rate: 1e-4
    scheduler: 
        _target_: model.TrapezoidalLR
        _partial_: true       
        warmup_steps: 3
        constant_steps: 5
        decay_steps: 4
"""

faster_decay_yaml_end = """
training:
    learning_rate: 1e-4
    scheduler:
        _target_: model.TrapezoidalLR
        _partial_: true       
        warmup_steps: 3
        constant_steps: 2
        decay_steps: 3
"""

faster_decay_yaml_middle = """
training:
    learning_rate: 1e-4
    scheduler:
        _target_: model.TrapezoidalLR
        _partial_: true       
        warmup_steps: 3
        constant_steps: 3
        decay_steps: 3
"""


class TestTrapezoidal(unittest.TestCase):

    def test_trapezoidal_sheduler(self):
        cfg = OmegaConf.create(config_yaml)

        optimizer = torch.optim.AdamW(
            torch.nn.Linear(1, 1).parameters(),
            lr=cfg.training.learning_rate,
        )

        scheduler = instantiate(cfg.training.scheduler)(optimizer=optimizer)

        lrs = []
        for _ in range(
            scheduler.warmup_steps + scheduler.constant_steps + scheduler.decay_steps
        ):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        self.assertListEqual(target_lrs, lrs)

    def test_checkpointing(self):
        target_fast_lrs_end = [
            0.0001,
            6.666666666666668e-05,
            3.333333333333334e-05,
        ]

        target_fast_lrs_middle = [
            0.0001,
            0.0001,
            6.666666666666668e-05,
            3.333333333333334e-05,
        ]

        cfg = OmegaConf.create(config_yaml)

        optimizer = torch.optim.AdamW(
            torch.nn.Linear(1, 1).parameters(),
            lr=cfg.training.learning_rate,
        )

        scheduler = instantiate(cfg.training.scheduler)(optimizer=optimizer)
        lrs = []
        for step in range(
            scheduler.warmup_steps + scheduler.constant_steps + scheduler.decay_steps
        ):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()
            if step == 4:
                checkpoint = scheduler.state_dict()

        def test_checkpointing(
            tested_cfg,
            checkpoint,
        ):
            optimizer_for_test = torch.optim.AdamW(
                torch.nn.Linear(1, 1).parameters(),
                lr=tested_cfg.training.learning_rate,
            )

            scheduler_fast = instantiate(tested_cfg.training.scheduler)(
                optimizer=optimizer_for_test
            )
            scheduler_fast.load_state_dict(checkpoint)

            result_lrs = []
            for _ in range(
                scheduler_fast.decay_steps
                + (
                    scheduler_fast.warmup_steps
                    + scheduler_fast.constant_steps
                    - checkpoint["last_epoch"]
                )
            ):
                optimizer_for_test.step()
                result_lrs.append(scheduler_fast.get_last_lr()[0])
                scheduler_fast.step()

            return result_lrs

        cfg_end = OmegaConf.create(faster_decay_yaml_end)
        cfg_middle = OmegaConf.create(faster_decay_yaml_middle)

        lrs_end = test_checkpointing(cfg_end, checkpoint)
        lrs_middle = test_checkpointing(cfg_middle, checkpoint)

        self.assertListEqual(target_lrs, lrs)
        self.assertListEqual(lrs_end, target_fast_lrs_end)
        self.assertListEqual(lrs_middle, target_fast_lrs_middle)


if __name__ == "__main__":
    unittest.main()
