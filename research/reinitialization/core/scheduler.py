from abc import ABC, abstractmethod

import torch.nn as nn
import torch
import torch.nn.functional as F
from attr import define

from research.reinitialization.core.pruner import BasePruner, RetrainPruner
from lizrd.datasets import wikibookdata


class BaseScheduler(ABC):
    @abstractmethod
    def step(self):
        ...


class DelayedConstScheduler(BaseScheduler):
    def __init__(
        self, pruner: BasePruner, n_steps_prune: int, prob: float, delay: int = 0
    ):
        self.pruner = pruner
        self.n_steps_prune = n_steps_prune
        self.prob = prob
        self.delay = delay
        self.current_step = 0

    def step(self):
        if (
            self.current_step % self.n_steps_prune == 0
            and self.current_step >= self.delay
        ):
            self.pruner.prune(self.prob)
        self.current_step += 1


@define
class RetrainScheduler(BaseScheduler):
    pruner: RetrainPruner
    n_steps_prune: int
    prob: float
    delay: int
    model: nn.Module
    optimizer: torch.optim.Optimizer
    pdataset: wikibookdata.ProcessedDataset
    batch_size: int
    vocab_size: int
    mask_percent: float
    mask_loss_weight: float
    n_steps_retrain: int = 1000

    def __attrs_post_init__(self):
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        self.current_step = 0
        self.full_step = 0

    def _retrain(self):
        self.pruner.prepare_new(self.prob)

        # freeze model
        self.model.requires_grad_(False)

        # unfreeze new
        self.pruner.pre_retrain()

        # retrain
        for _ in range(self.n_steps_retrain):
            self._train_step()

        # unfreeze model
        self.model.requires_grad_(True)
        self.pruner.post_retrain()

    def step(self):
        if (
            self.current_step % self.n_steps_prune == 0
            and self.current_step >= self.delay
        ):
            self._retrain()
            self.current_step += 1
            return

        self.current_step += 1
        self.full_step += 1

    def _optimize(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _train_step(self):
        self.full_step += 1
        self.model.train()
        processed_batch = self.pdataset.get_batch(self.batch_size)
        assert isinstance(processed_batch, wikibookdata.ProcessedBatch)
        x_set = processed_batch.masked_tokens
        y_token_set = processed_batch.tokens
        y_mask_set = processed_batch.mask_mask

        with torch.autocast(device_type="cuda", enabled=False, dtype=torch.float16):
            model_output = self.model(x_set)
            mask_loss = F.cross_entropy(
                model_output.reshape(-1, self.vocab_size),
                y_token_set.reshape(-1).long(),
                reduction="none",
            )
            mask_loss *= y_mask_set.reshape(-1)  # only check masked words
            mask_loss = mask_loss.mean() / self.mask_percent
            scaled_mask_loss = mask_loss * self.mask_loss_weight
            total_loss = scaled_mask_loss

        self.writer.add_scalar(
            "full_loss/train_total", total_loss.item(), self.full_step
        )
        self.writer.add_scalar("full_loss/train_mask", mask_loss.item(), self.full_step)

        self._optimize(total_loss)
