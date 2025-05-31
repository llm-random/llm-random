import unittest
from unittest.mock import patch
import torch

from hydra import initialize, compose
from hydra.utils import instantiate


from token_reduction.model import (
    LLM_MTP,
    TrainerMTP,
)

from model import (
    RecorderLogger,
    get_metric_logger,
    load_training_state,
    run,
)


class TestSimpleRun(unittest.TestCase):
    @patch("model.get_metric_logger", return_value=RecorderLogger())
    def test_simple_mtp(self, get_metric_logger):
        TOLERANCE = 1e-5
        target_losses_dropping = [
            (10.940040588378906, 0),
            (10.923925399780273, 1),
            (10.917631149291992, 2),
            (10.922800064086914, 3),
            (10.90736198425293, 4),
            (10.912927627563477, 5),
            (10.859469413757324, 6),
            (10.839786529541016, 7),
            (10.807689666748047, 8),
            (10.827632904052734, 9),
        ]

        target_eval_losses = [
            (10.889080047607422, 3),
            (10.872688293457031, 6),
            (10.832010269165039, 9),
        ]
        with initialize(version_base=None, config_path="configs"):
            cfg = compose(config_name="test_mtp", overrides=[])
            training_state = load_training_state(cfg.checkpoint_config)
            metric_logger = get_metric_logger(
                metric_logger_config=instantiate(cfg.metric_logger, _convert_="all"),
                neptune_run_id=training_state["run_id"],
            )
            run(cfg, metric_logger)

            for (expected_loss, expected_step), (actual_loss, actual_step) in zip(
                target_losses_dropping, metric_logger.data["steps/train/loss"]
            ):
                self.assertEqual(expected_step, actual_step)
                self.assertAlmostEqual(expected_loss, actual_loss, delta=TOLERANCE)

            for (expected_loss, expected_step), (actual_loss, actual_step) in zip(
                target_eval_losses, metric_logger.data["steps/eval/loss"]
            ):
                self.assertEqual(expected_step, actual_step)
                self.assertAlmostEqual(expected_loss, actual_loss, delta=TOLERANCE)


class TestMTPMerging(unittest.TestCase):
    target_losses = [
        (11.838484764099121, 0),
        (11.887506484985352, 1),
        (11.834625244140625, 2),
        (12.026954650878906, 3),
        (11.736723899841309, 4),
        (11.873994827270508, 5),
        (11.842313766479492, 6),
        (11.829595565795898, 7),
        (11.854312896728516, 8),
        (11.883036613464355, 9),
        (11.865650177001953, 10),
    ]
    grad_norms = [
        (2.3808863162994385, 0),
        (1.968237042427063, 1),
        (1.90666925907135, 2),
        (2.2426211833953857, 3),
        (1.9466161727905273, 4),
        (2.0401883125305176, 5),
        (2.138519763946533, 6),
        (2.235278844833374, 7),
        (1.9879982471466064, 8),
        (2.01497220993042, 9),
        (2.011291742324829, 10),
    ]
    eval_losses = [(11.717543601989746, 5), (11.743537902832031, 10)]

    def test_ultimate(self):
        TOLERANCE = 1e-5

        with initialize(version_base=None, config_path="configs"):
            cfg = compose(config_name="mtp_merge", overrides=[])

        training_state = {"next_step": 0, "run_id": None, "processed_tokens": 0}
        metric_logger = get_metric_logger(
            metric_logger_config=instantiate(cfg.metric_logger, _convert_="all"),
            neptune_run_id=training_state["run_id"],
        )
        metric_logger.clear()

        run(cfg, metric_logger)

        def compare_lists(list1, list2, key):
            self.assertEqual(len(list1), len(list2), f"Mismatch in {key} length")
            for (v1, s1), (v2, s2) in zip(list1, list2):
                self.assertAlmostEqual(
                    v1, v2, delta=TOLERANCE, msg=f"Mismatch in {key} at step {s1}"
                )
                self.assertEqual(s1, s2, f"Step mismatch in {key}")

        compare_lists(
            self.target_losses, metric_logger.data["steps/train/loss"], "train/loss"
        )
        compare_lists(
            self.grad_norms, metric_logger.data["steps/train/grad_norm"], "grad_norm"
        )
        compare_lists(
            self.eval_losses, metric_logger.data["steps/eval/loss"], "eval/loss"
        )


if __name__ == "__main__":
    unittest.main()
