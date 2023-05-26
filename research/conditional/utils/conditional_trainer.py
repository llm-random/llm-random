from functools import partial
from typing import Optional

import torch
from attr import define

from lizrd.datasets import wikibookdata
from lizrd.support.logging import AbstractLogger
from research.conditional.utils.layer_manager import LayerManager
from research.conditional.utils.model_utils import (
    calculate_gpt_loss,
    calculate_bert_loss,
)


@define(slots=False)
class ConditionalTrainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    train_dataloader: wikibookdata.ProcessedDatasetWrapper
    batch_size: int
    vocab_size: int
    mixed_precision: bool
    logger: AbstractLogger
    model_type: str
    logging_interval_light: int
    logging_interval_heavy: int
    _calculate_loss: Optional[callable] = None
    mask_percent: Optional[float] = None
    scaler: Optional[torch.cuda.amp.GradScaler] = None
    layer_manager: Optional[LayerManager] = None
    loss_accumulator: Optional[float] = None
    hack_for_batch_size: bool = False

    def __attrs_post_init__(self):
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.loss_accumulator = 0.0

        if self.model_type == "gpt":
            self._calculate_loss = calculate_gpt_loss
        elif self.model_type == "bert":
            self._calculate_loss = partial(
                calculate_bert_loss, mask_percent=self.mask_percent
            )
        self.layer_manager = LayerManager(
            self.model, self.logging_interval_light, self.logging_interval_heavy
        )

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
        self.layer_manager.prepare_for_logging(step)
        processed_batch = self.train_dataloader.get_batch()
        assert isinstance(processed_batch, wikibookdata.ProcessedBatch)

        loss = self._calculate_loss(
            processed_batch, self.model, self.mixed_precision, self.vocab_size
        )
        self._optimize(loss)
        self._log_loss(loss, step)
        self.layer_manager.log(step)

    def _log_loss(self, loss, step):
        self.loss_accumulator += loss.item()
        if step % 250 == 0 and step > 0:
            self.logger.report_scalar(
                title="loss", value=self.loss_accumulator / 250, iteration=step
            )
            self.loss_accumulator = 0.0

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
        for tensor in vars(processed_batch).values():
            if hasattr(tensor, "shape"):
                tensor.data = tensor[:1].repeat(step + 1, 1).data
        loss = self._calculate_loss(
            processed_batch, self.model, self.mixed_precision, self.vocab_size
        )
        self._optimize(loss)
        self.logger.report_scalar(title="max batch size", value=step, iteration=step)
