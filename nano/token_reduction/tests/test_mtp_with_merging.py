import unittest
from unittest.mock import patch
from hydra import initialize, compose
from hydra.utils import instantiate

from model import (
    get_metric_logger,
    run,
)

TOLERANCE = 1e-5

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
    (11.831990242004395, 11),
]
target_lrs = [
    (1e-05, 0),
    (2.8e-05, 1),
    (4.6e-05, 2),
    (6.4e-05, 3),
    (8.2e-05, 4),
    (0.0001, 5),
    (0.0001, 6),
    (9.140576474687264e-05, 7),
    (6.890576474687263e-05, 8),
    (4.1094235253127366e-05, 9),
    (1.8594235253127368e-05, 10),
    (1e-05, 11),
]
grad_norms = [
    (2.3808863162994385, 0),
    (1.968237042427063, 1),
    (1.90666925907135, 2),
    (2.2426211833953857, 3),
    (1.9466161727905273, 4),
    (2.0401883125305176, 5),
    (2.1385042667388916, 6),
    (2.235278844833374, 7),
    (1.9879982471466064, 8),
    (2.01497220993042, 9),
    (2.011291742324829, 10),
    (1.9894421100616455, 11),
]
eval_losses = [(11.717543601989746, 5), (11.743537902832031, 10)]


class TestMTPWithMerge(unittest.TestCase):
    def patch_randint_in_get_document(self, dataset):
        original_get_document = dataset.get_document

        def randint_generator():
            for i in range(1000):
                yield i

        randint_gen = randint_generator()

        def patched_get_document():
            with patch.object(
                dataset.py_rng, "randint", side_effect=lambda a, b: next(randint_gen)
            ):
                return original_get_document()

        dataset.get_document = patched_get_document

    def test_mtp_with_merge(self):

        with initialize(version_base=None, config_path="configs"):
            cfg = compose(config_name="test_mtp_with_merging", overrides=[])

        training_state = {"next_step": 0, "run_id": None, "processed_tokens": 0}
        metric_logger = get_metric_logger(
            metric_logger_config=instantiate(cfg.metric_logger, _convert_="all"),
            neptune_run_id=training_state["run_id"],
        )
        run(cfg, metric_logger)

        def compare_lists(list1, list2, key):
            self.assertEqual(len(list1), len(list2), f"Mismatch in {key} length")
            for (v1, s1), (v2, s2) in zip(list1, list2):
                self.assertAlmostEqual(
                    v1, v2, delta=TOLERANCE, msg=f"Mismatch in {key} at step {s1}"
                )
                self.assertEqual(s1, s2, f"Step mismatch in {key}")

        compare_lists(
            target_losses, metric_logger.data["steps/train/loss"], "train/loss"
        )
        compare_lists(target_lrs, metric_logger.data["steps/train/lr"], "train/lr")
        compare_lists(
            grad_norms, metric_logger.data["steps/train/grad_norm"], "grad_norm"
        )
        compare_lists(eval_losses, metric_logger.data["steps/eval/loss"], "eval/loss")


class TestMTPWithMergeDifferentDataloader(unittest.TestCase):
    def patch_randint_in_get_document(self, dataset):
        original_get_document = dataset.get_document

        def randint_generator():
            for i in range(1000):
                yield i

        randint_gen = randint_generator()

        def patched_get_document():
            with patch.object(
                dataset.py_rng, "randint", side_effect=lambda a, b: next(randint_gen)
            ):
                return original_get_document()

        dataset.get_document = patched_get_document

    def test_mtp_with_merge_different_dataloader(self):
        TOLERANCE = 1e-5

        with initialize(version_base=None, config_path="configs"):
            cfg = compose(config_name="mtp_merge_different_dataloader", overrides=[])

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
            target_losses, metric_logger.data["steps/train/loss"], "train/loss"
        )
        compare_lists(target_lrs, metric_logger.data["steps/train/lr"], "train/lr")
        compare_lists(
            grad_norms, metric_logger.data["steps/train/grad_norm"], "grad_norm"
        )
        compare_lists(eval_losses, metric_logger.data["steps/eval/loss"], "eval/loss")


if __name__ == "__main__":
    unittest.main()
