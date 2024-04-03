from typing import Literal

import torch
from attr import define
import torch.nn.functional as F
from lizrd.text.data import LLMBatch
from lizrd.train.scheduler import AbstractLRScheduler
from research.datasets import DataloaderWrapper


@define(slots=False)
class BaseTrainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    train_dataloader: DataloaderWrapper
    lr_scheduler: AbstractLRScheduler
    dataset_type: Literal["wikibook", "c4"]

    def train(self, n_steps: int):
        for step in range(n_steps):
            self.model.train()
            self.lr_scheduler.set_lr(step=step, optimizer=self.optimizer)
            batch = next(self.train_dataloader)
            loss, aux_info = self.calculate_loss(batch)
            print("LOSS: ", loss.item())
            loss.backward()
            self._apply_gradient()

    def calculate_loss(self, batch: LLMBatch):
        input_tokens = batch.input_ids
        gt_tokens = batch.target_ids
        mask = batch.should_calculate_loss

        model_output = self.model(input_tokens)

        gt_tokens = gt_tokens.to(model_output.device)
        mask = mask.to(model_output.device)

        flattened_loss = F.cross_entropy(
            model_output.flatten(0, -2),
            gt_tokens.reshape(-1).long(),
            reduction="none",
        )
        mask_loss = flattened_loss[mask.reshape(-1) == 1]
        loss = mask_loss.mean()

        correct_tokens = gt_tokens.long() == model_output.argmax(dim=-1)
        masked_correct_tokens = correct_tokens.long().reshape(-1) * mask.reshape(-1)

        aux_info = {
            "correct_tokens": masked_correct_tokens,
            "total_tokens": mask.sum(),
        }
        return loss, aux_info

    def _apply_gradient(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
