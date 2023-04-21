from typing import Optional

import torch
import torch.nn.functional as F
from attr import define

from lizrd.datasets import wikibookdata
from lizrd.support.logging import AbstractLogger


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
    scaler: Optional[torch.cuda.amp.GradScaler] = None
    hack_for_batch_size: bool = False

    def __attrs_post_init__(self):
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

    def train(self, n_steps: int):
        for step in range(n_steps):
            if self.hack_for_batch_size:
                self._hack_for_batch_size(step)
                print(f"Step {step}")
            else:
                self._train_step(step)
                if step % 1000 == 0:
                    print(f"Step {step}")

    def _optimize(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _calculate_loss(self, x_set, y_token_set, y_mask_set):
        if self.mixed_precision:
            with torch.autocast(
                device_type="cuda", enabled=self.mixed_precision, dtype=torch.float16
            ):
                model_output = self.model(x_set)
        else:
            model_output = self.model(x_set)

        mask_loss = F.cross_entropy(
            model_output.reshape(-1, self.vocab_size),
            y_token_set.reshape(-1).long(),
            reduction="none",
        )
        mask_loss *= y_mask_set.reshape(-1)
        loss = mask_loss.mean() / self.mask_percent
        return loss

    def _train_step(
        self,
        step,
    ):
        self.model.train()
        processed_batch = self.train_dataloader.get_batch()
        assert isinstance(processed_batch, wikibookdata.ProcessedBatch)
        x_set = processed_batch.masked_tokens
        y_token_set = processed_batch.tokens
        y_mask_set = processed_batch.mask_mask
        loss = self._calculate_loss(x_set, y_token_set, y_mask_set)
        self._optimize(loss)
        self.logger.report_scalar(
            title="loss", value=loss.item(), iteration=step, series="train"
        )

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
        x_set = processed_batch.masked_tokens
        y_token_set = processed_batch.tokens
        y_mask_set = processed_batch.mask_mask
        for tensor in [x_set, y_token_set, y_mask_set]:
            tensor.data = tensor[:1].repeat(step + 1, 1).data
        loss = self._calculate_loss(x_set, y_token_set, y_mask_set)
        self._optimize(loss)
