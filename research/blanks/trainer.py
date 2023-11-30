from collections import defaultdict
import os.path
import copy
from types import SimpleNamespace as SN
import time
from typing import Callable, List, Optional, Literal

import torch
from attr import define
from lizrd.support.logging import AbstractLogger
from lizrd.support.misc import get_ith_chunk
from research.blanks.model import BlankDiffPredictionHead

from .data import BlanxBatch
from lizrd.train.scheduler import AbstractLRScheduler

from .loss import make_loss_function
from research.datasets import DataloaderWrapper
from lizrd.text.datasets import C4Dataset


@define(slots=False)
class BlankTrainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    train_dataloader: DataloaderWrapper
    eval_dataloader: DataloaderWrapper
    vocab_size: int
    mixed_precision: bool
    logger: Optional[AbstractLogger]
    dataset_type: Literal["wikibook", "c4"]
    logging_interval_loss: int
    logging_interval_light: int
    logging_interval_heavy: int
    n_eval_steps: int
    n_eval_batches: int
    max_sequence_length: int
    batch_size: int
    lr_scheduler: AbstractLRScheduler
    blanks_ids: List[int]
    _calculate_loss: Optional[Callable] = None
    mask_percent: Optional[float] = None
    scaler: Optional[torch.cuda.amp.GradScaler] = None
    # layer_manager: Optional[LayerManager] = None
    loss_accumulator: Optional[float] = None
    n_gpus: int = 1
    hack_name: str = None
    save_weights_path: str = None
    save_weights_interval: int = 1000
    load_weights_path: str = None
    gradient_clipping: float = None
    loss_checkpoint_chungs: int = 0
    gradient_accumulation_steps: int = 1
    log_gradients_and_weights: bool = False
    loss_log_intervals: tuple[int] = (1, 10, 100, 1000)
    decoding_logging_steps: int = 5_000
    total_time_trainsteps: float = 0.0
    total_time_decoding: float = 0.0
    total_time_afterstep: float = 0.0
    is_process_logging: bool = True
    n_blanks: int = 0
    use_only_last_blank_loss: bool = False

    def __attrs_post_init__(self):
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.loss_accumulators = {
            f"loss_interval/{i}": SN(acc=0.0, interval=i)
            for i in self.loss_log_intervals
        }
        self.loss_accumulators["loss"] = SN(
            acc=0.0, interval=self.logging_interval_loss
        )
        self.correct_tokens_accumulator = 0.0
        self.total_tokens_accumulator = 0.0
        self.auxiliary_losses_accumulator = dict()
        self._calculate_loss = make_loss_function(
            loss_checkpoint_chungs=self.loss_checkpoint_chungs,
            n_blanks=self.n_blanks,
            blanks_ids=self.blanks_ids,
        )

    def _restore_weights(self):
        if self.load_weights_path is not None:
            if os.path.exists(self.load_weights_path):
                print(f"Loading weights from {self.load_weights_path}")
                checkpoint = torch.load(self.load_weights_path)
                self.model.load_state_dict(checkpoint["model"], strict=False)
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.scaler.load_state_dict(checkpoint["scaler"])
            else:
                print(
                    f"No weights found at {self.load_weights_path}, training from scratch"
                )

    def _save_weights(self, step):
        if (
            self.save_weights_interval > 0
            and self.save_weights_path is not None
            and step % self.save_weights_interval == 0
        ):
            print("Saving weights... ")
            checkpoint = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(self.save_weights_path, f"{step}.pth"))
            print(f"Weights saved to {self.save_weights_path} (step {step})")

    def _before_train_operations(self):
        pass

    def _after_step_operations(self):
        pass

    def train(self, n_steps: int):
        """
        Train the model for n_steps steps.
        """
        self._before_train_operations()
        self._restore_weights()
        for step in range(n_steps + 1):
            t0 = time.time()

            if self.hack_name is not None:
                self._hack(self.hack_name, step)
            else:
                self._train_step(step)

            t1 = time.time()
            if step % 1000 == 0:
                print(f"Step {step}")

            if step % self.n_eval_steps == 0:
                self._eval_step(step)

            t2 = time.time()
            self._after_step_operations()

            t3 = time.time()

            self.total_time_trainsteps += t1 - t0
            self.total_time_decoding += t2 - t1
            self.total_time_afterstep += t3 - t2

            if step % 1000 == 0 and self.is_process_logging:
                total_time = (
                    self.total_time_trainsteps
                    + self.total_time_decoding
                    + self.total_time_afterstep
                )
                self.logger.report_scalar(
                    title="time/trainstep_fraction",
                    value=self.total_time_trainsteps / total_time,
                    iteration=step,
                )
                self.logger.report_scalar(
                    title="time/decoding_fraction",
                    value=self.total_time_decoding / total_time,
                    iteration=step,
                )
                self.logger.report_scalar(
                    title="time/afterstep_fraction",
                    value=self.total_time_afterstep / total_time,
                    iteration=step,
                )

    def _optimize(self, loss, should_apply_gradient=False):
        # since we sum gradients averaged over multiple smaller batches, we need to normalize here
        loss /= self.gradient_accumulation_steps
        if self.gradient_accumulation_steps == 1:
            self.optimizer.zero_grad()
        # clear computation graph, store gradients
        self.scaler.scale(loss).backward()
        if should_apply_gradient:
            if self.gradient_clipping is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clipping
                )
            self.scaler.step(self.optimizer)
            if self.gradient_accumulation_steps > 1:
                self.optimizer.zero_grad()
            self.scaler.update()

    def _train_step(
        self,
        step,
    ):
        self.model.train()
        processed_batch = self.train_dataloader.get_batch()

        self.lr_scheduler.set_lr(step=step, optimizer=self.optimizer)
        loss, aux_info = self.calculate_loss_and_maybe_optimize(
            processed_batch, should_optimize=True
        )
        if self.is_process_logging:
            self._log_train_stats(loss, step, aux_info)
            self._log_accuracy(aux_info, step)
            self._log_heavy(step)
            self._log_auxiliary_losses(aux_info["losses"], step)
        self._save_weights(step)

    def _eval_step(self, step):
        self.model.eval()
        total_loss = 0.0
        total_correct_tokens = 0
        total_masked_tokens = 0
        extra_losses = defaultdict(float)
        for _ in range(self.n_eval_batches):
            processed_batch = self.eval_dataloader.get_batch()
            with torch.no_grad():
                loss, aux_info = self.calculate_loss_and_maybe_optimize(
                    processed_batch, should_optimize=False
                )
            total_loss += loss
            total_correct_tokens += aux_info["correct_tokens"]
            total_masked_tokens += aux_info["total_masked_tokens"]
            for name, loss_value in aux_info["losses"].items():
                extra_losses[name] += loss_value
        if self.is_process_logging:
            self.logger.report_scalar(
                title="eval/total_loss",
                value=total_loss / self.n_eval_batches,
                iteration=step,
            )
            self.logger.report_scalar(
                title="eval/accuracy",
                value=total_correct_tokens / total_masked_tokens,
                iteration=step,
            )
            for name, loss_value in extra_losses:
                self.logger.report_scalar(
                    title=f"eval/{name}",
                    value=loss_value / self.n_eval_batches,
                    iteration=step,
                )

    def calculate_loss_and_maybe_optimize(
        self, processed_batch: BlanxBatch, should_optimize: bool
    ):
        """gradient accumulation: slice the batch into minibatches, get gradients from each, then average and apply them"""
        total_cross_entropy_loss = 0.0
        correct_tokens_value = 0
        total_masked_tokens_value = 0
        losses = {}

        for i in range(self.gradient_accumulation_steps):
            batch_copy = copy.deepcopy(processed_batch)
            for _, tensor in batch_copy:
                tensor.data = get_ith_chunk(
                    tensor.data, self.gradient_accumulation_steps, i
                )

            cross_entropy_loss, aux_info = self._calculate_loss(
                batch=batch_copy,
                model=self.model,
                mixed_precision=self.mixed_precision,
                vocab_size=self.vocab_size,
            )

            # clear computation graph, store gradients, only apply gradients at the end
            should_apply_gradient = i == self.gradient_accumulation_steps - 1

            loss_to_optimize = cross_entropy_loss
            for key, value in aux_info["losses"].items():
                loss_to_optimize += value

            if should_optimize:
                self._optimize(
                    loss_to_optimize, should_apply_gradient=should_apply_gradient
                )
            total_cross_entropy_loss += cross_entropy_loss.item()
            correct_tokens_value += aux_info["correct_tokens"]
            total_masked_tokens_value += aux_info["total_masked_tokens"]

            for key, value in aux_info["losses"].items():
                losses[key] = losses.get(key, 0) + value

        return total_cross_entropy_loss, {
            "correct_tokens": correct_tokens_value,
            "total_masked_tokens": total_masked_tokens_value,
            "losses": losses,
            "blanks_losses": aux_info["blanks_losses"],
        }

    def _log_train_stats(self, loss_value, step, aux_info):
        self.logger.report_scalar(title="step", value=step, iteration=step)
        self.logger.report_scalar(
            title="lr", value=self.lr_scheduler.get_lr(step=step), iteration=step
        )
        if self.dataset_type == "c4":
            self._log_fraction_dataset_processed(step)
        for name, stats in self.loss_accumulators.items():
            stats.acc += loss_value
            if step % stats.interval == 0 and step > 0:
                self.logger.report_scalar(
                    title=name,
                    value=stats.acc / stats.interval,
                    iteration=step,
                )
                stats.acc = 0.0
        if self.n_blanks > 0 and len(aux_info["blanks_losses"]) > 0:
            if getattr(self.model.head, "preblank_weight", None) is not None:
                self.logger.report_scalar(
                    title=f"blank_head/preblank_weight",
                    value=abs(self.model.head.preblank_weight.item()),
                    iteration=step,
                )
            if getattr(self.model.head, "blank_weight", None) is not None:
                self.logger.report_scalar(
                    title=f"blank_head/blank_weight",
                    value=abs(self.model.head.blank_weight.item()),
                    iteration=step,
                )

            self.logger.report_scalar(
                title=f"sanity/blank_0_loss - loss (should be around 0 or slightly positive due to blanks)",
                value=(aux_info["blanks_losses"]["blank_0_loss"] - loss_value),
                iteration=step,
            )

            for name, value in aux_info["blanks_losses"].items():
                self.logger.report_scalar(
                    title=name,
                    value=value,
                    iteration=step,
                )
            self.logger.report_scalar(
                title="blank_last_loss - blank_0_loss",
                value=(
                    aux_info["blanks_losses"][f"blank_{self.n_blanks}_loss"]
                    - aux_info["blanks_losses"]["blank_0_loss"]
                ),
                iteration=step,
            )
            if not self.use_only_last_blank_loss:
                for x in range(1, self.n_blanks + 1):
                    # log diff
                    name = f"blank_{x}_loss - blank_{x-1}_loss"
                    self.logger.report_scalar(
                        title=name,
                        value=(
                            aux_info["blanks_losses"][f"blank_{x}_loss"]
                            - aux_info["blanks_losses"][f"blank_{x-1}_loss"]
                        ),
                        iteration=step,
                    )

    def _log_heavy(self, step):
        g_metrics, w_metrics = {}, {}
        if (
            step % self.logging_interval_heavy == 0
            and step > 0
            and self.log_gradients_and_weights
        ):
            for name, value in self.model.named_parameters():
                if value.grad is not None:
                    norm = torch.linalg.norm(value.grad)
                    g_metrics[f"weight_norms/{name.replace('.', '/')}/grad"] = norm
                if value.requires_grad:
                    norm = torch.linalg.norm(value)
                    w_metrics[f"weight_norms/{name.replace('.', '/')}/weight"] = norm
            g_metrics[f"weight_norms/grad_norm_total"] = torch.linalg.norm(
                torch.tensor(list(g_metrics.values()))
            )
            w_metrics[f"weight_norms/weight_norm_total"] = torch.linalg.norm(
                torch.tensor(list(w_metrics.values()))
            )
            self._log_dict({**g_metrics, **w_metrics}, step)

    def _log_dict(self, metrics, step):
        for k, v in metrics.items():
            self.logger.report_scalar(title=k, value=v, iteration=step)

    def _log_fraction_dataset_processed(self, step):
        processed = step * self.batch_size * self.max_sequence_length
        total = C4Dataset.total_gpt2_tokens
        self.logger.report_scalar(
            title="Fraction of dataset that is processed (assumuing no DDP)",
            value=processed / total,
            iteration=step,
        )

    def _log_accuracy(self, aux_info, step):
        self.correct_tokens_accumulator += aux_info["correct_tokens"]
        self.total_tokens_accumulator += aux_info["total_masked_tokens"]
        if step % self.logging_interval_loss == 0 and step > 0:
            self.logger.report_scalar(
                title="accuracy",
                value=self.correct_tokens_accumulator / self.total_tokens_accumulator,
                iteration=step,
            )
            self.correct_tokens_accumulator = 0.0
            self.total_tokens_accumulator = 0.0

    def _log_auxiliary_losses(self, losses, step):
        for name, loss in losses.items():
            self.auxiliary_losses_accumulator[name] = (
                self.auxiliary_losses_accumulator.get(name, 0) + loss
            )

        if step % self.logging_interval_loss == 0 and step > 0:
            for name, loss in losses.items():
                self.logger.report_scalar(
                    title=f"{name}",
                    value=loss / self.logging_interval_loss,
                    iteration=step,
                )
            self.auxiliary_losses_accumulator.clear()

    def _hack(self, hack_name, step):
        if hack_name == "batch_size":
            self._hack_for_batch_size(step)
        else:
            raise ValueError(f"Unknown hack {hack_name}")

    def _hack_for_batch_size(
        self,
        step,
    ):
        """
        This is a hack to easily determine the maximal batch size that can be used with given GPU memory and model size.
        """
        self.model.train()
        processed_batch = self.train_dataloader.get_batch()
        for name, tensor in processed_batch:
            tensor.data = tensor[:1].repeat(step + 1, 1).data
        loss, _aux_info = self._calculate_loss(
            batch=processed_batch,
            model=self.model,
            mixed_precision=self.mixed_precision,
            vocab_size=self.vocab_size,
        )
        self._optimize(loss, should_apply_gradient=True)
        if self.is_process_logging:
            self.logger.report_scalar(
                title="max batch size", value=step * self.n_gpus, iteration=step
            )
