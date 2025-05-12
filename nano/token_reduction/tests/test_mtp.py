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
    load_training_state,
    run,
)


class TestSimpleRun(unittest.TestCase):
    @patch("model.get_metric_logger", return_value=RecorderLogger())
    def test_simple_mtp_deepseek(self, get_metric_logger):
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
        with initialize(version_base=None, config_path="configs"):
            cfg = compose(config_name="test_mtp", overrides=[])
            training_state = load_training_state(cfg.checkpoint_config)
            metric_logger = get_metric_logger(
                metric_logger_config=instantiate(cfg.metric_logger, _convert_="all"),
                neptune_run_id=training_state["run_id"],
            )
            run(cfg)

        setup_enviroment()
        metric_logger = get_metric_logger(
            metric_logger_config=instantiate(cfg.metric_logger, _convert_="all"),
        )
        torch.manual_seed(cfg.trainer_factory.train_dataloader.seed)
        device = torch.device("cpu")

        model = instantiate(cfg.model, _convert_="all").to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

        scheduler = instantiate(cfg.training.scheduler)(optimizer=optimizer)

        training_state = {"next_step": 0, "run_id": None, "processed_tokens": 0}

        trainer_factory = instantiate(cfg.trainer_factory)
        mtp_trainer = trainer_factory(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
            metric_logger=metric_logger,
        )
        mtp_trainer.train()


if __name__ == "__main__":
    unittest.main()
