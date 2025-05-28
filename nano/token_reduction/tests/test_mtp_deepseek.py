import unittest
from unittest.mock import patch

from hydra import initialize, compose
from hydra.utils import instantiate


from model import (
    RecorderLogger,
    load_training_state,
    run,
)


class TestSimpleRun(unittest.TestCase):
    @patch("model.get_metric_logger", return_value=RecorderLogger())
    def test_simple_mtp_deepseek(self, get_metric_logger):
        TOLERANCE = 1e-5
        target_losses_dropping = [
            (10.914034843444824, 0),
            (10.883499145507812, 1),
            (10.891767501831055, 2),
            (10.889894485473633, 3),
            (10.874717712402344, 4),
            (10.853045463562012, 5),
            (10.860404968261719, 6),
            (10.825983047485352, 7),
            (10.828263282775879, 8),
            (10.803018569946289, 9),
        ]
        target_losses_tokens_eval = [
            (10.850308418273926, 2176),
            (10.820022583007812, 3808),
            (10.793370246887207, 5440),
        ]
        with initialize(version_base=None, config_path="configs"):
            cfg = compose(config_name="test_mtp_deepseek", overrides=[])
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
                target_losses_tokens_eval, metric_logger.data["tokens/eval/loss"]
            ):
                self.assertEqual(expected_step, actual_step)
                self.assertAlmostEqual(expected_loss, actual_loss, delta=TOLERANCE)

    @patch("model.get_metric_logger", return_value=RecorderLogger())
    def test_mtp_deepseek_merge(self, get_metric_logger):
        TOLERANCE = 1e-5
        target_train_losses_dropping = [
            (10.896793365478516, 0),
            (10.923377990722656, 1),
            (10.917985916137695, 2),
            (10.898540496826172, 3),
            (10.909551620483398, 4),
            (10.886929512023926, 5),
            (10.878108978271484, 6),
            (10.83626937866211, 7),
            (10.835912704467773, 8),
            (10.82840633392334, 9),
        ]
        target_mtp_loss_1 = [
            (10.885116577148438, 768),
            (10.922760009765625, 1536),
            (10.914349555969238, 2304),
            (10.87898063659668, 3072),
            (10.903907775878906, 3840),
            (10.882705688476562, 4608),
            (10.849594116210938, 5376),
            (10.858087539672852, 6144),
            (10.88386344909668, 6912),
            (10.83377456665039, 7680),
        ]
        target_eval_loss = [
            (10.860756874084473, 3072),
            (10.880765914916992, 5376),
            (10.82032585144043, 7680),
        ]
        with initialize(version_base=None, config_path="configs"):
            cfg = compose(config_name="test_mtp_deepseek_merge", overrides=[])
            training_state = load_training_state(cfg.checkpoint_config)
            metric_logger = get_metric_logger(
                metric_logger_config=instantiate(cfg.metric_logger, _convert_="all"),
                neptune_run_id=training_state["run_id"],
            )
            run(cfg, metric_logger)

            for (expected_loss, expected_step), (actual_loss, actual_step) in zip(
                target_train_losses_dropping, metric_logger.data["steps/train/loss"]
            ):
                self.assertEqual(expected_step, actual_step)
                self.assertAlmostEqual(expected_loss, actual_loss, delta=TOLERANCE)

            for (expected_loss, expected_tokens), (actual_loss, actual_step) in zip(
                target_mtp_loss_1,
                metric_logger.data["tokens/train/mtp_loss_1"],
            ):
                self.assertEqual(expected_tokens, actual_step)
                self.assertAlmostEqual(expected_loss, actual_loss, delta=TOLERANCE)

            for (expected_loss, expected_tokens), (actual_loss, actual_step) in zip(
                target_eval_loss, metric_logger.data["tokens/eval/loss"]
            ):
                self.assertEqual(expected_tokens, actual_step)
                self.assertAlmostEqual(expected_loss, actual_loss, delta=TOLERANCE)


if __name__ == "__main__":
    unittest.main()
