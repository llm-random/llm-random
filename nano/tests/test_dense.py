import unittest
from unittest.mock import patch
from hydra import initialize, compose
from hydra.utils import instantiate
import torch
import sys

from model import (
    get_metric_logger,
    load_training_state,
    run,
)

TOLERANCE = 1e-6
class TestDense(unittest.TestCase):
 
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

    def test_dense(self):
        TOLERANCE = 1e-6

        target_losses = [
            (11.692767143249512, 0),
            (11.700878143310547, 1),
            (11.628955841064453, 2),
            (11.853792190551758, 3),
            (11.681392669677734, 4),
            (11.572286605834961, 5),
            (11.835830688476562, 6),
            (11.695648193359375, 7),
            (11.682708740234375, 8),
            (11.542861938476562, 9),
        ]
        target_lrs = [
            1e-05,
            2.8e-05,
            4.6e-05,
            6.4e-05,
            8.2e-05,
            0.0001,
            0.0001,
            8.681980515339464e-05,
            5.5e-05,
            2.3180194846605367e-05,
            1e-05,
        ]
        grad_norms = [
            (1.1134647130966187, 0),
            (1.172190546989441, 1),
            (1.123613715171814, 2),
            (1.20794677734375, 3),
            (1.1908608675003052, 4),
            (1.1530029773712158, 5),
            (1.1273497343063354, 6),
            (1.2080551385879517, 7),
            (1.07181715965271, 8),
            (1.0252130031585693, 9),
        ]
        eval_losses = [
            (11.650517463684082, 2),
            (11.65742301940918, 4),
            (11.603936195373535, 6),
            (11.593389511108398, 8),
        ]

        with initialize(version_base=None, config_path="configs"):
            cfg = compose(config_name="dense_minuscule", overrides=[])

        training_state = load_training_state(cfg.checkpoint_config)
        metric_logger = get_metric_logger(
            metric_logger_config=instantiate(cfg.metric_logger, _convert_="all"),
            neptune_run_id=training_state["run_id"],
        )
        run(cfg)

        target_tuple_lrs = list(zip(target_lrs, range(cfg.training.n_steps)))

        current_version = sys.version_info
        if current_version.major == 3 and current_version.minor == 10 and current_version.micro == 12:
            # if the version is 3.10.12 we know the exact values
            self.assertListEqual(target_losses, metric_logger.data["steps/train/loss"])
            self.assertListEqual(target_tuple_lrs, metric_logger.data["steps/train/lr"])
            self.assertListEqual(grad_norms, metric_logger.data["steps/train/grad_norm"])
            self.assertListEqual(eval_losses, metric_logger.data["steps/eval/loss"])
        else:
            def compare_lists(list1, list2, key):
                self.assertEqual(len(list1), len(list2), f"Mismatch in {key} length")
                for (v1, s1), (v2, s2) in zip(list1, list2):
                    self.assertAlmostEqual(v1, v2, delta=TOLERANCE, 
                                        msg=f"Mismatch in {key} at step {s1}")
                    self.assertEqual(s1, s2, f"Step mismatch in {key}")

            compare_lists(target_losses, metric_logger.data["steps/train/loss"], "train/loss")
            self.assertListEqual(target_tuple_lrs, metric_logger.data["steps/train/lr"])
            compare_lists(grad_norms, metric_logger.data["steps/train/grad_norm"], "grad_norm")
            compare_lists(eval_losses, metric_logger.data["steps/eval/loss"], "eval/loss")


if __name__ == "__main__":
    unittest.main()


