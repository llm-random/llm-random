import unittest
import hydra
from unittest.mock import patch
import torch

from hydra import initialize, compose
from hydra.utils import instantiate

from token_reduction.model import (
    TrainerMTP,
    LLM_MTP,
)
from model import (
    StdoutLogger,
    setup_enviroment,
    get_metric_logger,
)


class TestTrainerMTP(unittest.TestCase):
    def test_mtp(self):
        with initialize(version_base=None, config_path="configs"):
            cfg = compose(config_name="test_mtp", overrides=[])

        setup_enviroment()
        metric_logger = get_metric_logger(
            metric_logger_config=instantiate(cfg.metric_logger, _convert_="all"),
        )
        torch.manual_seed(cfg.training.seed)
        device = torch.device("cpu")

        model = instantiate(cfg.model, _convert_="all").to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

        scheduler = instantiate(cfg.training.scheduler)(optimizer=optimizer)

        dataloaders_factory = instantiate(cfg.dataloaders_factory)
        train_dataloader, eval_dataloader = dataloaders_factory()

        training_state = {"next_step": 0, "run_id": None, "processed_tokens": 0}

        trainer_factory = instantiate(cfg.trainer_factory)
        mtp_trainer = trainer_factory(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            metric_logger=metric_logger,
        )
        mtp_trainer.train()


if __name__ == "__main__":
    unittest.main()
