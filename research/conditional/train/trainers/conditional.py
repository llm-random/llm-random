from functools import partial
from typing import Optional

import torch
from attr import define

from lizrd.datasets import wikibookdata
from lizrd.support.logging import AbstractLogger
from research.conditional.train.trainers.utils import (
    calculate_gpt_loss,
    calculate_bert_loss,
)
from research.conditional.utils.layer_manager import LayerManager


@define
class ConditionalTrainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    train_dataloader: wikibookdata.ProcessedDatasetWrapper
    batch_size: int
    vocab_size: int
    mask_percent: float
    mixed_precision: bool
    logger: AbstractLogger
    model_type: str
    scaler: Optional[torch.cuda.amp.GradScaler] = None
    hack_for_batch_size: bool = False
    logging_interval: int = 1000
    verbosity_level: int = 0

    def __attrs_post_init__(self):
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        if self.model_type == "gpt":
            self._calculate_loss = calculate_gpt_loss
        elif self.model_type == "bert":
            self._calculate_loss = partial(
                calculate_bert_loss, mask_percent=self.mask_percent
            )
        self.layer_manager = LayerManager(self.model)

    def train(self, n_steps: int):
        for step in range(n_steps):
            if self.hack_for_batch_size:
                self._hack_for_batch_size(step)
            else:
                self._train_step(step)
            if step % 1000 == 0:
                print(f"Step {step}")

    def _optimize(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _train_step(
        self,
        step,
    ):
        self.model.train()
        processed_batch = self.train_dataloader.get_batch()
        assert isinstance(processed_batch, wikibookdata.ProcessedBatch)

        loss = self._calculate_loss(
            processed_batch, self.model, self.mixed_precision, self.vocab_size
        )
        self._optimize(loss)
        self.logger.report_scalar(
            title="loss", value=loss.item(), iteration=step, series="train"
        )

    def log(self, step):
        if self.verbosity_level == 0:
            return
        if step and step % self.logging_interval == 0:
            self.layer_manager.log(step, verbosity_level=self.verbosity_level)

    def _hack_for_batch_size(
        self,
        step,
    ):
        """
        This is a hack to easily determine the maximal batch size that can be used with given GPU memory and model size.
        """
        self.model.train()
        processed_batch = self.train_dataloader.get_batch()
        assert isinstance(processed_batch, wikibookdata.ProcessedBatch)
        for tensor in dir(processed_batch):
            if isinstance(tensor, torch.Tensor):
                tensor.data = tensor[:1].repeat(step + 1, 1).data
        loss = self._calculate_loss(
            processed_batch, self.model, self.mixed_precision, self.vocab_size
        )
        self._optimize(loss)
        print(f"Batch size {step} still fits!")
