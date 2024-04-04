from collections import namedtuple
from typing import Literal
from dataclasses import dataclass

import torch
from attr import define
import torch.nn.functional as F
from lizrd.text.data import LLMBatch
from lizrd.train.scheduler import AbstractLRScheduler
from research.datasets import DataloaderWrapper


@dataclass
class StepMetric:
    loss: float
    norm: float

class TrainingMetricHolder:
    def __init__(self):
        self.metrics = {}

    def set_metrics(self, metrics):
        self.metrics = metrics

    def append_metrics(self, step, loss, model):
        weights = [param.detach().flatten() for param in model.parameters()]
        norm = torch.cat(weights).norm().item()

        self.metrics[step] = StepMetric(loss, norm)

    def __eq__(self, other: "TrainingMetricHolder"):
        return self.metrics == other.metrics

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):

        metrics_str = "\n".join(
            f"{step}: {metric.loss} {metric.norm}"
            for step, metric in self.metrics.items()
        )
        return f"TrainingMetricHolder:\nStep  | Loss  | Norm\n{metrics_str}"


@define(slots=False)
class BaseTrainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    train_dataloader: DataloaderWrapper
    lr_scheduler: AbstractLRScheduler
    dataset_type: Literal["wikibook", "c4"]
    hold_metrics: bool = False

    def __attrs_post_init__(self):
        self.metric_holder = TrainingMetricHolder()
        self.hold_metrics = True  # TODO remove

    def train(self, n_steps: int):
        for step in range(n_steps):
            self.model.train()
            self.lr_scheduler.set_lr(step=step, optimizer=self.optimizer)
            batch = next(self.train_dataloader)
            loss, aux_info = self.calculate_loss(batch)
            loss.backward()
            self._apply_gradient()

            # Metrics for testing purposes only
            if self.hold_metrics:
                self.metric_holder.append_metrics(step, loss.item(), self.model)

        return self.metric_holder

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
