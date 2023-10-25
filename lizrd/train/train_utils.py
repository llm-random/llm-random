import copy
from collections import defaultdict
from typing import Callable, Optional
import os

import numpy as np
import plotly.express as px
import torch
import torch.nn.functional as F
from attr import define
from torch.utils.tensorboard import SummaryWriter

from lizrd.core import llm
from lizrd.core.misc import are_state_dicts_the_same
from lizrd.datasets import wikibookdata
import lizrd.datasets.processed_batch
from lizrd.support.logging import AbstractLogger
from lizrd.support.logging import get_current_logger
from lizrd.support.loss import (
    LossDict,
    RunningLossDict,
    LossWeightDict,
)
from research.reinitialization.core.pruner import BasePruner
from research.reinitialization.core.scheduler import BaseScheduler


def get_model(
    max_length: int,
    vocab_size: int,
    ff_layer_fun: Callable[[], torch.nn.Module],
    attention_layer_fun: Callable[[], torch.nn.Module],
    dm: int,
    n_blocks: int,
    device: torch.device,
    init_type,
    init_scale,
    gradient_checkpointing: bool = False,
    model_fragmentation: Optional[list[int]] = None,
    residual_fn: Callable[[], torch.nn.Module] = None,
):
    if model_fragmentation is None or device == torch.device("cpu"):
        first_gpu = device
        last_gpu = device
    else:
        first_gpu = torch.device("cuda:0")
        last_gpu = torch.device(f"cuda:{len(model_fragmentation)}")

    embedding_layer = llm.EmbeddingLayer(
        llm.PositionalEmbedding(
            max_length, dm, init_type=init_type, init_scale=init_scale
        ).to(first_gpu),
        llm.TokenEmbedding(
            vocab_size, dm, init_type=init_type, init_scale=init_scale
        ).to(first_gpu),
    )

    layer_dict = {"attention": attention_layer_fun, "feedforward": ff_layer_fun}
    # Python officially preserves dict order since 3.7, so we pass the layer dict
    encoder_tower = llm.TransformerTower(
        n_blocks,
        dm,
        layer_dict,
        gradient_checkpointing,
        device,
        model_fragmentation=model_fragmentation,
        residual_fn=residual_fn,
    )

    head = llm.PredictionHead(
        dm, vocab_size, init_type=init_type, init_scale=init_scale
    ).to(last_gpu)

    model = llm.LLM(embedding_layer, encoder_tower, head)

    return model


@define(slots=False)
class Trainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    pdataset: wikibookdata.ProcessedDatasetWrapper
    pdataset_eval: wikibookdata.ProcessedDatasetWrapper
    batch_size: int
    vocab_size: int
    mask_percent: float
    modelpath: str
    pruner: BasePruner
    logger: AbstractLogger
    scheduler: Optional[BaseScheduler] = None
    mixed_precision: bool = False
    scaler: Optional[torch.cuda.amp.GradScaler] = None
    step: int = 0
    n_log_light_steps: Optional[int] = None
    n_log_heavy_steps: Optional[int] = None
    log_acc_steps: int = 100
    running_total_loss: float = 0.0
    running_mask_loss: float = 0.0
    running_loss_steps: int = 0
    losses_weights: LossWeightDict = defaultdict(lambda: 0.0)
    running_losses: RunningLossDict = defaultdict(lambda: 0.0)
    neuron_diff_dataset: Optional[wikibookdata.ProcessedDatasetWrapper] = None
    neuron_diff_sample_size: int = 1
    neuron_diff_n_samples: int = 100
    neuron_diff_n_batches: int = 10
    lr_warmup_steps: int = 10_000
    noise_interpolation_delay: int = 0
    model_type: str = "bert"

    def __attrs_post_init__(self):
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

    def _pruning_step(self, step):
        if self.scheduler and self.scheduler.is_time_to_prune(step):
            self.pruner.prune(self.scheduler.prob)

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
        self.mask_loss = F.cross_entropy(
            model_output.reshape(-1, self.vocab_size),
            y_token_set.reshape(-1).long(),
            reduction="none",
        )
        self.mask_loss *= y_mask_set.reshape(-1)  # only check masked words
        self.token_losses = self.mask_loss[y_mask_set.reshape(-1).bool()]
        mask_loss = self.mask_loss.mean() / self.mask_percent
        return mask_loss

    def _get_lm_loss(self, input, target, non_padded_mask):
        output = self.model(input)
        lm_loss = F.cross_entropy(
            output.reshape(-1, self.vocab_size),
            target.reshape(-1).long(),
            reduction="none",
        )
        lm_loss *= non_padded_mask.reshape(-1)
        lm_loss = lm_loss.mean()
        return lm_loss

    def _task_train_step(
        self,
        dataset: wikibookdata.ProcessedDataset,
        step: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        if self.model_type == "bert":
            self.model.train()
            processed_batch = dataset.get_batch()
            assert isinstance(
                processed_batch, lizrd.datasets.processed_batch.ProcessedBatch
            )
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
                optimizer=self.optimizer if optimizer is None else optimizer,
                scaler=self.scaler,
                loss=loss,
                step=step,
                run_after_backprop=True,
            )
        else:
            self.model.train()
            processed_batch = dataset.get_batch()
            assert isinstance(
                processed_batch, lizrd.datasets.processed_batch.ProcessedBatch
            )
            input = processed_batch.tokens
            target = processed_batch.target_tokens
            non_padded_mask = processed_batch.non_padded_mask

            with torch.autocast(
                device_type="cuda", enabled=self.mixed_precision, dtype=torch.float16
            ):
                losses = {"mask": self._get_lm_loss(input, target, non_padded_mask)}
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

        if self.n_log_heavy_steps and step > 0 and step % self.n_log_heavy_steps == 0:
            self.token_losses_before = self.token_losses.clone()
            with torch.no_grad():
                self._get_mask_loss(x_set, y_token_set, y_mask_set)
            self.token_losses_after = self.token_losses.clone()
            assert self.token_losses_before.shape == self.token_losses_after.shape
            torch.save(processed_batch, os.path.join(self.modelpath, "checked_batch"))
            torch.save(
                self.token_losses_before,
                os.path.join(self.modelpath, "token_losses_before"),
            )

        if (
            self.n_log_heavy_steps
            and step > 0
            and step > self.n_log_heavy_steps
            and step % self.n_log_heavy_steps in [10, 100, 500, 1000]
        ):
            del processed_batch
            historical_batch = torch.load(os.path.join(self.modelpath, "checked_batch"))
            historical_losses = torch.load(
                os.path.join(self.modelpath, "token_losses_before")
            )

            x_set = historical_batch.masked_tokens
            y_token_set = historical_batch.tokens
            y_mask_set = historical_batch.mask_mask
            with torch.no_grad():
                self._get_mask_loss(x_set, y_token_set, y_mask_set)
            self.token_losses_after = self.token_losses.clone()
            self.token_losses_before = historical_losses

            self.log_token_losses(step)

    def _model_train_step(self, step: int):
        self.model.train()
        with torch.autocast(
            device_type="cuda", enabled=self.mixed_precision, dtype=torch.float16
        ):
            losses = self.pruner.get_auxiliary_loss()

            # return if there are no auxiliary losses to optimize
            if len(losses) == 0:
                return

            self.update_loss_stats(losses)
            scaled_losses = self.scale_losses(losses)
            loss = sum(scaled_losses.values())

        self.optimize(
            optimizer=self.sgd_optimizer,
            scaler=self.sgd_scaler,
            loss=loss,
            step=step,
            run_after_backprop=False,
        )

    def _train_step(
        self,
        dataset: wikibookdata.ProcessedDataset,
        step: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self._task_train_step(dataset, step, optimizer)
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
        sample: int = 100,
        log_values: bool = True,
    ):
        self.model.eval()

        if self.model_type == "bert":
            with torch.no_grad():
                total_mask_loss = 0.0
                for _ in range(sample):
                    processed_batch = self.pdataset_eval.get_batch()
                    assert isinstance(
                        processed_batch, lizrd.datasets.processed_batch.ProcessedBatch
                    )
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
        else:
            with torch.no_grad():
                total_loss = 0.0
                for _ in range(sample):
                    processed_batch = self.pdataset_eval.get_batch()
                    assert isinstance(
                        processed_batch, lizrd.datasets.processed_batch.ProcessedBatch
                    )
                    input = processed_batch.tokens
                    target = processed_batch.target_tokens
                    non_padded_mask = processed_batch.non_padded_mask
                    mask_loss = self._get_lm_loss(input, target, non_padded_mask)
                    total_loss += mask_loss.item()
                total_loss /= sample

                if log_values:
                    self.logger.report_scalar(
                        title="loss",
                        series="eval_mask",
                        value=total_loss,
                        iteration=step,
                    )

                return total_loss

    def check_neuron_diff(self, step: int):
        """
        Check the neuron diff for each layer:
            1. For each batch, compute the loss for the batch
            2. For each sample, mask neurons from the sample
            3. Compute the loss for the batch with the chosen neurons masked
            4. Compute the difference between the two losses
            5. Log histogram of the results
            6. Repeat 2-5 for all layers
        """
        print("Beginning of check_neuron_diff...")
        with torch.no_grad():
            for i in range(len(self.pruner.layers)):
                results = np.zeros(self.neuron_diff_n_samples)
                activation_ratios = np.zeros(self.neuron_diff_n_samples)
                magnitudes = np.zeros(self.neuron_diff_n_samples)

                for _ in range(self.neuron_diff_n_batches):
                    processed_batch = self.neuron_diff_dataset.get_batch()
                    assert isinstance(
                        processed_batch, lizrd.datasets.processed_batch.ProcessedBatch
                    )

                    baseline = self._compute_loss(processed_batch).detach().cpu().item()

                    for j in range(self.neuron_diff_n_samples):
                        self.pruner.enable_neuron_diff(ff_layer_num=i, sample_number=j)
                        total_mask_loss = (
                            self._compute_loss(processed_batch).detach().cpu().item()
                        )
                        results[j] += total_mask_loss - baseline
                        activation_ratios[
                            j
                        ] = self.pruner.get_activation_ratios_of_masked_neurons(i)
                        magnitudes[j] = self.pruner.get_magnitudes_of_masked_neurons(i)

                results /= self.neuron_diff_n_batches
                activation_ratios /= self.neuron_diff_n_batches
                magnitudes /= self.neuron_diff_n_batches
                mean = results.mean()
                results = results.tolist()
                activation_ratios = activation_ratios.tolist()
                magnitudes = magnitudes.tolist()
                self.pruner.disable_neuron_diff()

                # log neuron diffs
                fig = px.histogram(results)
                fig.add_vline(
                    x=mean,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="mean",
                    annotation=dict(font_size=20),
                    annotation_position="top right",
                )
                get_current_logger().report_plotly(
                    title="Neuron quality (higher is better)",
                    series=f"Layer {i+1}",
                    iteration=step,
                    figure=fig,
                )

                # log scatter of neuron diff/activation
                if self.neuron_diff_sample_size == 1:
                    fig = px.scatter(
                        x=results,
                        y=activation_ratios,
                    )
                    fig.update_layout(
                        xaxis_title="Quality (Higher is better)",
                        yaxis_title="Activation ratio",
                    )
                    get_current_logger().report_plotly(
                        title="Quality vs activation",
                        series=f"Layer {i+1}",
                        iteration=step,
                        figure=fig,
                    )

                # Log scatter of neurn diff/magnitudes
                if self.neuron_diff_sample_size == 1:
                    fig = px.scatter(
                        x=results,
                        y=magnitudes,
                    )
                    fig.update_layout(
                        xaxis_title="Quality (higher is better)",
                        yaxis_title="Magnitude",
                    )
                    get_current_logger().report_plotly(
                        title="Quality vs magnitude",
                        series=f"Layer {i+1}",
                        iteration=step,
                        figure=fig,
                    )

        print("Neuron diff logged.")

    def log_token_losses(self, step: int):
        print(f"Logging token losses at step {step}...")
        values = self.token_losses_after - self.token_losses_before
        values = values.detach().cpu().numpy().tolist()
        fig = px.histogram(values)
        # log_plot(
        #     title="Token losses difference",
        #     series="Token losses difference",
        #     iteration=step,
        #     figure=fig,
        # )

    def set_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        for param_group in self.sgd_optimizer.param_groups:
            param_group["lr"] = lr

    def train(self, n_steps: int, n_steps_eval: int):
        # params for lr warmup
        target_lr = self.optimizer.param_groups[0]["lr"]
        if self.lr_warmup_steps > n_steps:
            print(
                f"Warning: lr_warmup_steps ({self.lr_warmup_steps}) is larger than n_steps ({n_steps})."
            )

        if self.neuron_diff_dataset is not None:
            self.pruner.prepare_neuron_diff_idx(
                n_samples=self.neuron_diff_n_samples,
                sample_size=self.neuron_diff_sample_size,
            )

        for step in range(n_steps):
            # lr warmup in the beginning
            if step <= self.lr_warmup_steps and self.lr_warmup_steps > 0:
                lr = target_lr * step / self.lr_warmup_steps
                self.set_lr(lr)

            # tell the model to save activation stats if necessary:
            if self.n_log_heavy_steps and step % self.n_log_heavy_steps == 0:
                self.pruner.set_saving_stats()

            # log neuron difference stats if necessary:
            if (
                self.neuron_diff_dataset is not None
                and step % self.n_log_heavy_steps == 0
            ):
                self.check_neuron_diff(step)

            # tell the model that noise interpolation delay is finished (if necessary)
            if step == self.noise_interpolation_delay:
                self.pruner.enable_noise_interpolation()

            self._pruning_step(step)
            self._train_step(dataset=self.pdataset, step=step)
            self.running_loss_steps += 1
            self._log_train_stats(step)
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
                self.pruner.log_heavy(step, self.modelpath)
                self.log_token_losses(step)
            print(f"Step {step}")


class SetLRTemporarily:
    """
    Context manager to temporarily set the learning rate of an optimizer (like in lr warmup).
    Use as follows:
    with SetLRTemporarily(optimizer, lr):
        # do something
    """

    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.lr = lr
        self.original_lrs = []

    def __enter__(self):
        for param_group in self.optimizer.param_groups:
            self.original_lrs.append(param_group["lr"])
            param_group["lr"] = self.lr

    def __exit__(self, *args):
        for param_group, lr in zip(self.optimizer.param_groups, self.original_lrs):
            param_group["lr"] = lr


@define
class RetrainTrainer(Trainer):
    # TODO: totally split from Trainer
    pdataset_retrain: Optional[wikibookdata.ProcessedDataset] = None
    retrain_warmup_steps: Optional[int] = None
    retrain_count: int = 0
    statistics_reset_steps: int = None

    def full_step(self, step: int):
        return step + self.retrain_count

    # def _log_train_stats(self, total_loss: float, mask_loss: float, step: int):
    #     full_step = step + self.retrain_count
    #     if full_step:
    #         self.logger.report_scalar(
    #             title="loss/train_total",
    #             value=total_loss,
    #             iteration=step,
    #         )
    #         self.logger.report_scalar(
    #             title="loss/train_mask",
    #             value=mask_loss,
    #             iteration=step,
    #         )
    #         self.logger.report_scalar(
    #             title="full_loss/train_total",
    #             value=total_loss,
    #             iteration=full_step,
    #         )
    #         self.logger.report_scalar(
    #             title="full_loss/train_mask",
    #             value=mask_loss,
    #             iteration=full_step,
    #         )
    #         print(f'Reporting lr: {self.optimizer.param_groups[0]["lr"]}')
    #         self.logger.report_scalar(
    #             title="full_steps/lr",
    #             value=self.optimizer.param_groups[0]["lr"],
    #             iteration=full_step,
    #         )
    #         self.logger.report_scalar(
    #             title="is_retraining",
    #             value=0,
    #             iteration=full_step,
    #         )
    #         if full_step % self.n_log_light_steps == 0:
    #             self.pruner.log_light(full_step)

    def _log_retrain_stats(
        self,
        total_loss: float,
        mask_loss: float,
        step: int,
        optimizer: torch.optim.Optimizer,
    ):
        full_step = self.full_step(step)
        self.logger.report_scalar(
            title="full_loss/train_total",
            value=total_loss,
            iteration=full_step,
        )
        self.logger.report_scalar(
            title="full_loss/train_mask",
            value=mask_loss,
            iteration=full_step,
        )
        print(f'Reporting lr: {self.optimizer.param_groups[0]["lr"]}')
        self.logger.report_scalar(
            title="full_steps/lr",
            value=optimizer.param_groups[0]["lr"],
            iteration=full_step,
        )
        self.logger.report_scalar(
            title="is_retraining",
            value=1,
            iteration=full_step,
        )
        if full_step % self.n_log_light_steps == 0:
            self.pruner.log_light(full_step)

    def _pruning_step(self, step):
        if self.scheduler.is_time_to_prune(step):
            self._retrain(step)

    def _task_train_step(
        self,
        dataset: wikibookdata.ProcessedDataset,
        step: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        if self.model_type == "bert":
            self.model.train()
            processed_batch = dataset.get_batch()
            assert isinstance(
                processed_batch, lizrd.datasets.processed_batch.ProcessedBatch
            )
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
                optimizer=self.optimizer if optimizer is None else optimizer,
                scaler=self.scaler,
                loss=loss,
                step=step,
                run_after_backprop=True,
            )
        else:
            self.model.train()
            processed_batch = dataset.get_batch()
            assert isinstance(
                processed_batch, lizrd.datasets.processed_batch.ProcessedBatch
            )
            input = processed_batch.tokens
            target = processed_batch.target_tokens
            non_padded_mask = processed_batch.non_padded_mask

            with torch.autocast(
                device_type="cuda", enabled=self.mixed_precision, dtype=torch.float16
            ):
                losses = {"mask": self._get_lm_loss(input, target, non_padded_mask)}
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

        if self.n_log_heavy_steps and step > 0 and step % self.n_log_heavy_steps == 0:
            self.token_losses_before = self.token_losses.clone()
            with torch.no_grad():
                self._get_mask_loss(x_set, y_token_set, y_mask_set)
            self.token_losses_after = self.token_losses.clone()
            assert self.token_losses_before.shape == self.token_losses_after.shape
            torch.save(processed_batch, os.path.join(self.modelpath, "checked_batch"))
            torch.save(
                self.token_losses_before,
                os.path.join(self.modelpath, "token_losses_before"),
            )

        if (
            self.n_log_heavy_steps
            and step > 0
            and step > self.n_log_heavy_steps
            and step % self.n_log_heavy_steps in [10, 100, 500, 1000]
        ):
            del processed_batch
            historical_batch = torch.load(os.path.join(self.modelpath, "checked_batch"))
            historical_losses = torch.load(
                os.path.join(self.modelpath, "token_losses_before")
            )

            x_set = historical_batch.masked_tokens
            y_token_set = historical_batch.tokens
            y_mask_set = historical_batch.mask_mask
            with torch.no_grad():
                self._get_mask_loss(x_set, y_token_set, y_mask_set)
            self.token_losses_after = self.token_losses.clone()
            self.token_losses_before = historical_losses

            self.log_token_losses(step)

        return loss, loss

    def _train_step(
        self,
        dataset: wikibookdata.ProcessedDataset,
        step: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        total_loss, mask_loss = self._task_train_step(dataset, step, optimizer)
        self._model_train_step(step)
        return total_loss, mask_loss

    def _retrain(self, step):
        loss_before_recycle = self._eval_step(step)
        self.logger.report_scalar(
            title="loss/eval_just_before_recycle",
            value=loss_before_recycle,
            iteration=step,
        )
        print(f"Eval loss before recycle:", loss_before_recycle)

        self.pruner.prepare_new(self.scheduler.prob)

        # freeze model
        self.model.requires_grad_(False)

        # unfreeze new
        self.pruner.pre_retrain()

        # create retrain optimizer (without old stats)
        retrain_optim = torch.optim.Adam(
            self.model.parameters(),
            self.optimizer.param_groups[0]["lr"],
            self.optimizer.param_groups[0]["betas"],
            self.optimizer.param_groups[0]["eps"],
        )
        target_lr = self.optimizer.param_groups[0]["lr"]
        if not self.retrain_warmup_steps:
            self.retrain_warmup_steps = int(self.scheduler.n_steps_retrain / 2)

        # reset optimizer stats
        print("Resetting optimizer stats...")
        if self.statistics_reset_steps is None:
            self.statistics_reset_steps = self.retrain_count
        with SetLRTemporarily(self.optimizer, 0.0):
            for _ in range(self.statistics_reset_steps):
                self._train_step(
                    retrain_optim, self.pdataset_retrain, log_auxiliary_loss=False
                )
        print("Optimizer stats reset.")

        # retrain
        for i in range(self.scheduler.n_steps_retrain):
            if i < 5:
                loss_after_recycle = self._eval_step(step, log_values=False)
                # self.logger.report_scalar(
                #     title="loss/eval_just_after_recycle",
                #     value=loss_after_recycle,
                #     iteration=step + i,
                # )
                print(f"Eval loss after recycle:", loss_after_recycle)
            # lr warmup
            lr_coeff = min(1.0, i / self.retrain_warmup_steps)
            retrain_optim.param_groups[0]["lr"] = lr_coeff * target_lr

            total_loss, mask_loss = self._train_step(
                optimizer=retrain_optim,
                dataset=self.pdataset_retrain,
                step=self.full_step(step),
            )
            self._log_retrain_stats(total_loss, mask_loss, step, retrain_optim)
            self.retrain_count += 1

        # unfreeze model
        self.model.requires_grad_(True)

        self.pruner.apply_new_weights()
        self.pruner.post_retrain()


@define
class LTHTrainer:
    model: torch.nn.Module
    optimizer_creator: Callable[[torch.nn.Module], torch.optim.Optimizer]
    pdataset_creator: Callable[[], wikibookdata.ProcessedDataset]
    pruner: BasePruner
    batch_size: int
    vocab_size: int
    mask_percent: float
    mask_loss_weight: float
    modelpath: str
    n_steps_per_run: int
    n_steps_eval: int
    writer: SummaryWriter
    pruning_rate: float
    target_params: float
    initial_model_path: Optional[str] = None

    def _save_model_params(self):
        self.initial_model_path = f"{self.modelpath}/init.pt"
        print(f'Saving initial model to "{self.initial_model_path}"')
        torch.save(self.model.state_dict(), self.initial_model_path)

    def _save_checkpoint(self, step):
        model_path = f"{self.modelpath}/{step}.pt"
        print(f'Saving checkpoint@{step} to "{model_path}"')
        torch.save(self.model.state_dict(), model_path)

    def _reinitialize_model(self):
        """Reinitialize the model to its original state without losing track of masks."""
        with torch.no_grad():
            masks = copy.deepcopy([layer.mask for layer in self.pruner.layers])
            model_state_dict = torch.load(self.initial_model_path)
            assert not are_state_dicts_the_same(
                self.model.state_dict(), model_state_dict
            )
            self.model.load_state_dict(model_state_dict)
            assert are_state_dicts_the_same(self.model.state_dict(), model_state_dict)
            for layer, mask in zip(self.pruner.layers, masks):
                layer.mask = mask
            assert not are_state_dicts_the_same(
                self.model.state_dict(), model_state_dict
            )

    def _log_masks_percentage(self, step):
        zeros = 0
        total = 0
        for layer in self.pruner.layers:
            zeros += torch.sum(layer.mask == 0).item()
            total += layer.mask.numel()
        self.writer.add_scalar("mask_percentage", zeros / total, step)

    def _train_step(
        self,
        optimizer: torch.optim.Optimizer,
        pdataset: wikibookdata.ProcessedDataset,
        step=0,
    ):
        self.model.train()
        processed_batch = pdataset.get_batch()
        x_set = processed_batch.masked_tokens
        y_token_set = processed_batch.tokens
        y_mask_set = processed_batch.mask_mask

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

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step and self.writer:
            self.writer.add_scalar("loss/train_total", total_loss.item(), step)
            self.writer.add_scalar("loss/train_mask", mask_loss.item(), step)

    def _eval_step(
        self,
        pdataset: wikibookdata.ProcessedDataset,
        step: int = 0,
        sample: int = 10,
    ):
        self.model.eval()

        with torch.no_grad():
            total_mask_loss = 0.0
            for _ in range(sample):
                processed_batch = pdataset.get_batch()
                x_set = processed_batch.masked_tokens
                y_token_set = processed_batch.tokens
                y_mask_set = processed_batch.mask_mask
                model_output = self.model(x_set)
                mask_loss = F.cross_entropy(
                    model_output.reshape(-1, self.vocab_size),
                    y_token_set.reshape(-1).long(),
                    reduction="none",
                )
                mask_loss *= y_mask_set.reshape(-1)  # only check masked words
                mask_loss = mask_loss.mean() / self.mask_percent
                scaled_mask_loss = mask_loss * self.mask_loss_weight
                total_mask_loss += scaled_mask_loss.item()
            total_mask_loss /= sample

            self.writer.add_scalar("loss/eval_mask", total_mask_loss, step)
            print(f"Eval loss:", total_mask_loss)
            return total_mask_loss

    def train(self):
        self._save_model_params()
        parameters_left = 1.0
        total_step = 0
        while True:
            optimizer = self.optimizer_creator(self.model)
            pdataset = self.pdataset_creator()
            self.writer.add_scalar("parameters_left", parameters_left, total_step)
            for step in range(self.n_steps_per_run):
                self._train_step(optimizer, pdataset, total_step)
                if step % self.n_steps_eval == 0:
                    self._eval_step(
                        pdataset, step=total_step, sample=self.n_steps_eval // 2
                    )
                self.writer.add_scalar("total_step", total_step, total_step)
                print(f"Run step {step}; Total step {total_step}")
                total_step += 1
            # just in case parameters left is not exact
            self._save_checkpoint(total_step)
            self._log_masks_percentage(total_step)
            self.pruner.step(parameters_left * self.pruning_rate)
            if parameters_left < self.target_params:
                break
            parameters_left *= 1 - self.pruning_rate
            self._reinitialize_model()
