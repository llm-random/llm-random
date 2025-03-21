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


# class TestTrainerMTP(unittest.TestCase):
#     def test_mtp(self):
#         with initialize(version_base=None, config_path="configs"):
#             cfg = compose(config_name="test_mtp", overrides=[])
@hydra.main(version_base=None, config_path="configs", config_name="test_mtp")
def tmp(cfg):
    setup_enviroment()
    metric_logger = get_metric_logger(
        metric_logger_config=instantiate(cfg.metric_logger, _convert_="all"),
    )

    model = instantiate(cfg.model, _convert_="all")

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
    # print(f'cfg.trainer_factory:\n{cfg.trainer_factory}')
    # print(f'trainer_factory:\n{type(trainer_factory)}')
    # print(f'model: {type(model)}')
    mtp_trainer = trainer_factory(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        metric_logger=metric_logger,
    )
    # print(f'mtp_trainer: {type(mtp_trainer)}')
    print("ok")
    mtp_trainer.train()


def test_preprocess_input_mtp(self):
    test_batch = torch.rand(size=(3, 7))


# t = TestTrainerMTP()
# t.test_mtp()

tmp()
