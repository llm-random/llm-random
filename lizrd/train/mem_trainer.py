from collections import defaultdict
from typing import List, Optional

import torch
import torch.nn.functional as F
from attr import define
import torch.nn.functional as F

from lizrd.datasets import wikibookdata
from lizrd.support.logging import AbstractLogger
from lizrd.support.loss import (
    LossDict,
    RunningLossDict,
    LossWeightDict,
)
from research.reinitialization.core.pruner import BasePruner


@define(slots=False)
class MemorizationTrainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    pdataset: wikibookdata.ProcessedDatasetWrapper
    pdataset_eval: wikibookdata.ProcessedDatasetWrapper
    batch_size: int
    vocab_size: int
    modelpath: str
    pruner: BasePruner
    logger: AbstractLogger
    mixed_precision: bool = False
    scaler: Optional[torch.cuda.amp.GradScaler] = None
    step: int = 0
    n_log_light_steps: Optional[int] = None
    n_log_heavy_steps: Optional[int] = None
    log_acc_steps: int = 100
    mask_percent: float = 0.15
    running_total_loss: float = 0.0
    running_mask_loss: float = 0.0
    running_loss_steps: int = 0
    losses_weights: LossWeightDict = defaultdict(lambda: 0.0)
    running_losses: RunningLossDict = defaultdict(lambda: 0.0)
    lr_warmup_steps: int = 10_000
    weight_decay: float = 0.01
    batches_schedulers: list = []
    test_mem_batches: List[wikibookdata.ProcessedBatch] = []
    baseline_mem_batches: List[wikibookdata.ProcessedBatch] = []
    eval_on_test_batches_n_steps: int = -1

    def __attrs_post_init__(self):
        assert len(self.batches_schedulers) == len(self.test_mem_batches)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        self.sgd_optimizer = torch.optim.SGD(
            params=self.model.parameters(), lr=self.optimizer.param_groups[0]["lr"]
        )
        self.sgd_scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        self.reset_loss_stats()

    def after_backprop(self, step: int):
        self.pruner.after_backprop(step)

    def optimize(
        self,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.grad_scaler.GradScaler],
        loss: torch.Tensor,
        step: int,
        run_after_backprop: bool,
    ):
        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        if run_after_backprop:
            self.after_backprop(step)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

    # def _pruning_step(self, step):
    #     if self.scheduler and self.scheduler.is_time_to_prune(step):
    #         self.pruner.prune(self.scheduler.prob)

    def update_loss_stats(self, losses: LossDict):
        for k, v in losses.items():
            self.running_losses[k] += v.item()

    def reset_loss_stats(self):
        for k in self.running_losses.keys():
            self.running_losses[k] = 0.0
        self.running_loss_steps = 0

    def log_loss_stats(self, step):
        total_loss = 0.0
        for k, v in self.running_losses.items():
            self.logger.report_scalar(
                title="loss",
                series=f"{k} (before scaling)",
                value=v / self.running_loss_steps,
                iteration=step,
            )
            scaled_loss = self.losses_weights[k] * v
            self.logger.report_scalar(
                title="loss",
                series=f"{k} (after scaling)",
                value=scaled_loss / self.running_loss_steps,
                iteration=step,
            )
            total_loss += scaled_loss
        self.logger.report_scalar(
            title="loss",
            series=f"total loss (after scaling)",
            value=total_loss / self.running_loss_steps,
            iteration=step,
        )

    def scale_losses(self, losses: dict) -> dict:
        scaled_losses = dict()
        for k, v in losses.items():
            scaled_losses[k] = v * self.losses_weights[k]
        return scaled_losses

    def _get_mask_loss(
        self,
        x_set: torch.Tensor,
        y_token_set: torch.Tensor,
        y_mask_set: torch.Tensor,
    ) -> torch.Tensor:
        model_output = self.model(x_set)
        mask_loss = F.cross_entropy(
            model_output.reshape(-1, self.vocab_size),
            y_token_set.reshape(-1).long(),
            reduction="none",
        )
        mask_loss *= y_mask_set.reshape(-1)  # only check masked words
        mask_loss = mask_loss.mean() / self.mask_percent
        return mask_loss

    def _task_train_step(
        self,
        dataset: wikibookdata.ProcessedDataset,
        step: int,
    ):
        self.model.train()
        processed_batch = dataset.get_batch()
        assert isinstance(processed_batch, wikibookdata.ProcessedBatch)
        x_set = processed_batch.masked_tokens
        y_token_set = processed_batch.tokens
        y_mask_set = processed_batch.mask_mask

        with torch.autocast(
            device_type="cuda", enabled=self.mixed_precision, dtype=torch.float16
        ):
            losses = {"mask": self._get_mask_loss(x_set, y_token_set, y_mask_set)}
            self.update_loss_stats(losses)
            scaled_losses = self.scale_losses(losses)
            loss = sum(scaled_losses.values())

        self.optimize(
            optimizer=self.optimizer,
            scaler=self.scaler,
            loss=loss,
            step=step,
            run_after_backprop=True,
        )

    def _model_train_step(self, step: int):
        self.model.train()
        losses = self.pruner.get_auxiliary_loss()
        self.update_loss_stats(losses)
        scaled_losses = self.scale_losses(losses)
        loss = sum(scaled_losses.values())

        if len(losses) == 0:
            print("No model auxiliary losses, skipping model training step")
            return

        self.optimize(
            optimizer=self.sgd_optimizer,
            scaler=None,
            loss=loss,
            step=step,
            run_after_backprop=False,
        )

    def _train_step(
        self,
        dataset: wikibookdata.ProcessedDataset,
        step: int,
    ):
        self._task_train_step(dataset, step)
        self._model_train_step(step)

    def _log_train_stats(self, step: int):
        if self.n_log_light_steps and step % self.n_log_light_steps == 0:
            self.pruner.log_light(step)
        if step and (step % self.log_acc_steps == 0):
            self.log_loss_stats(step)
            self.reset_loss_stats()

    def _eval_step(
        self,
        step: int,
        sample: int = 10,
        log_values: bool = True,
    ):
        self.model.eval()

        with torch.no_grad():
            total_mask_loss = 0.0
            for _ in range(sample):
                processed_batch = self.pdataset_eval.get_batch()
                assert isinstance(processed_batch, wikibookdata.ProcessedBatch)
                mask_loss = self._get_mask_loss(
                    x_set=processed_batch.masked_tokens,
                    y_token_set=processed_batch.tokens,
                    y_mask_set=processed_batch.mask_mask,
                )
                total_mask_loss += mask_loss.item()
            total_mask_loss /= sample

            if log_values:
                self.logger.report_scalar(
                    title="loss",
                    series="eval_mask",
                    value=total_mask_loss,
                    iteration=step,
                )

            return total_mask_loss

    def set_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        for param_group in self.sgd_optimizer.param_groups:
            param_group["lr"] = lr

    def train_on_batches(
        self, batches: List[wikibookdata.ProcessedBatch], step: int
    ) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in batches:
            x_set = batch.masked_tokens
            y_token_set = batch.tokens
            y_mask_set = batch.mask_mask

            with torch.autocast(
                device_type="cuda", enabled=self.mixed_precision, dtype=torch.float16
            ):
                loss = self._get_mask_loss(x_set, y_token_set, y_mask_set)

            self.optimize(
                optimizer=self.optimizer,
                scaler=self.scaler,
                loss=loss,
                step=step,
                run_after_backprop=False,
            )
            total_loss += loss.item()
        return total_loss

    def eval_on_batches(self, batches: List[wikibookdata.ProcessedBatch]) -> float:
        self.model.eval()
        total_loss = 0.0
        for batch in batches:
            x_set = batch.masked_tokens
            y_token_set = batch.tokens
            y_mask_set = batch.mask_mask

            with torch.autocast(
                device_type="cuda", enabled=self.mixed_precision, dtype=torch.float16
            ):
                loss = self._get_mask_loss(x_set, y_token_set, y_mask_set)

            total_loss += loss.item()
        return total_loss

    def train(self, n_steps: int, n_steps_eval: int):
        # params for lr warmup
        target_lr = self.optimizer.param_groups[0]["lr"]
        if self.lr_warmup_steps > n_steps:
            print(
                f"Warning: lr_warmup_steps ({self.lr_warmup_steps}) is larger than n_steps ({n_steps})."
            )

        for step in range(n_steps):
            # lr warmup in the beginning
            if step <= self.lr_warmup_steps and self.lr_warmup_steps > 0:
                lr = target_lr * step / self.lr_warmup_steps
                self.set_lr(lr)

            # tell the model to save activation stats if necessary:
            if self.n_log_heavy_steps and step % self.n_log_heavy_steps == 0:
                self.pruner.set_saving_stats()

            self._train_step(dataset=self.pdataset, step=step)
            self.running_loss_steps += 1
            self._log_train_stats(step)

            for scheduler, batch in zip(self.batches_schedulers, self.test_mem_batches):
                if scheduler(step):
                    self.train_on_batches([batch], step)

            should_test_mem = (
                any(
                    (scheduler(step + 1) or scheduler(step))
                    for scheduler in self.batches_schedulers
                )
                or (
                    self.eval_on_test_batches_n_steps > 0
                    and step % self.eval_on_test_batches_n_steps == 0
                )
                or step == n_steps - 1
            )
            if should_test_mem:
                test_mem_baseline = self.eval_on_batches(
                    self.baseline_mem_batches
                ) / len(self.baseline_mem_batches)
                self.logger.report_scalar(
                    title="mem_baseline_loss",
                    value=test_mem_baseline,
                    iteration=step,
                )
                for batch_no, batch in enumerate(self.test_mem_batches):
                    test_loss = self.eval_on_batches([batch])
                    self.logger.report_scalar(
                        title="test_mem_batch",
                        series=self.logger.args["mem_batches_schedule"][batch_no],
                        value=test_loss,
                        iteration=step,
                    )
                    if len(self.baseline_mem_batches) > 0:
                        self.logger.report_scalar(
                            title="test_mem_batch scaled (baseline / loss, bigger is better)",
                            series=self.logger.args["mem_batches_schedule"][batch_no],
                            value=test_mem_baseline / test_loss,
                            iteration=step,
                        )

            if step % self.log_acc_steps == 0:
                self.logger.report_scalar(title="step", value=step, iteration=step)
            if step % n_steps_eval == 0:
                eval_loss = self._eval_step(step)
                print(f"Eval loss:", eval_loss)
                torch.save(self.model.state_dict(), f"{self.modelpath}/model.pt")
            if (
                self.n_log_heavy_steps
                and step > 0
                and step % self.n_log_heavy_steps == 0
            ):
                print(f"Running heavy log at step {step}")
                self.pruner.log_heavy(step)
            print(f"Step {step}")
