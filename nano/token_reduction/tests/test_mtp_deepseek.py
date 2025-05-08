import unittest
from unittest.mock import patch
import torch

from hydra import initialize, compose
from hydra.utils import instantiate


from token_reduction.model import (
    LLM_DeepSeekMTP,
    TrainerDeepSeekMTP,
    get_deepseek_embedding,
)

from model import (
    RecorderLogger,
    load_training_state,
    run,
)

class TestSimpleRun(unittest.TestCase):
    @patch("model.get_metric_logger", return_value=RecorderLogger())
    def test_simple_mtp_deepseek(self, get_metric_logger):
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
            (10.803018569946289, 9)
        ]
        with initialize(version_base=None, config_path="configs"):
            cfg = compose(config_name="test_mtp_deepseek", overrides=[])
            training_state = load_training_state(cfg.checkpoint_config)
            metric_logger = get_metric_logger(
                metric_logger_config=instantiate(cfg.metric_logger, _convert_="all"),
                neptune_run_id=training_state["run_id"],
            )
            run(cfg)

            self.assertListEqual(
                target_losses_dropping, metric_logger.data["steps/train/loss"]
            )


if __name__ == "__main__":
    unittest.main()