from collections import defaultdict
import copy
from time import time
from types import SimpleNamespace as SN
from typing import Callable, Iterable, Optional, Literal

import torch
from torch.profiler import profile, ProfilerActivity
from attr import define
from lizrd.core.misc import propagate_forward_pass_cache
from lizrd.support.logging import AbstractLogger
from lizrd.support.misc import get_ith_chunk
from lizrd.text.data import LLMBatch
from lizrd.train.scheduler import AbstractLRScheduler
from research.tokenizex.model.tokenizer import ReversedGPT2Tokenizer
from research.tokenizex_comp.utils.layer_manager import LayerManager
from research.tokenizex_comp.utils.model_utils import (
    make_loss_and_gradient_function,
    update_model_fit_gpu_info,
)
from research.datasets import DataloaderWrapper
from lizrd.text.datasets import C4Dataset
from transformers import GPT2Tokenizer
from lizrd.train.load_and_save_model import load_scaler_state, save_checkpoint


@define(slots=False)
class TemplateTrainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    train_dataloader: DataloaderWrapper
    eval_dataloader: DataloaderWrapper
    vocab_size: int
    mixed_precision: bool
    mixed_precision_dtype: torch.dtype
    logger: Optional[AbstractLogger]
    model_type: Literal["bert", "gpt"]
    dataset_type: Literal["wikibook", "c4"]
    logging_interval_loss: int
    logging_interval_light: int
    logging_interval_heavy: int
    eval_interval: int
    n_eval_batches: int
    max_sequence_length: int
    batch_size: int
    lr_scheduler: AbstractLRScheduler
    _calculate_loss_and_gradient: Optional[Callable] = None
    mask_percent: Optional[float] = None
    scaler: Optional[torch.cuda.amp.GradScaler] = None
    layer_manager: Optional[LayerManager] = None
    loss_accumulator: Optional[float] = None
    n_gpus: int = 1
    save_weights_path: str = None
    save_weights_interval: int = 1000
    gradient_clipping: float = None
    loss_checkpoint_chungs: int = 0
    gradient_accumulation_steps: int = 1
    log_gradients_and_weights: bool = False
    loss_log_intervals: tuple[int] = (1, 10, 100, 1000)
    decoding_interval: int = 5_000
    total_time_trainsteps: float = 0.0
    total_time_decoding: float = 0.0
    total_time_afterstep: float = 0.0
    eval_min_group_size_logfactor: int = 0
    eval_max_group_size_logfactor: int = 0
    eval_discrete_mot: bool = False
    is_logging_process: bool = True
    eval_dynamic_groupsize: bool = False
    steps_until_start_temperature_learn: int = -1
    model_fit_gpu_info_database_path: str = None
    model_fit_gpu_info_params: list[str] = None
    profiler_enabled: bool = False
    profiler_trace_path: str = None
    profiler_schedule: None = None
    rank: Optional[int] = None
    start_step: int = 0
    checkpoint: Optional[dict[str, torch.Tensor]] = None

    def __attrs_post_init__(self):
        if self.mixed_precision_dtype == torch.float16:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.loss_accumulators = {
            f"loss_interval/{i}": SN(acc=0.0, interval=i)
            for i in self.loss_log_intervals
        }
        self.loss_accumulators["loss"] = SN(
            acc=0.0, interval=self.logging_interval_loss
        )

        self.byttok_loss = {
            f"comp/interval/byttok_loss/{i}": SN(acc=0.0, interval=i)
            for i in self.loss_log_intervals
        }
        self.deftok_loss = {
            f"comp/interval/deftok_loss/{i}": SN(acc=0.0, interval=i)
            for i in self.loss_log_intervals
        }
        self.fb_time_deftok_loss = {
            f"comp/interval/time/fb/deftok_loss/{i}": SN(
                acc=0.0, interval=i, last_time=0.0, last_step=0
            )
            for i in self.loss_log_intervals
        }
        self.fb_time_byttok_loss = {
            f"comp/interval/time/fb/byttok_loss/{i}": SN(
                acc=0.0, interval=i, last_time=0.0, last_step=0
            )
            for i in self.loss_log_intervals
        }
        self.fb_time_acc = 0
        self.fb_time_acc_interval = 0.0
        self.ts_start_training = None

        self.correct_tokens_accumulator = 0.0
        self.total_tokens_accumulator = 0.0
        self.auxiliary_losses_accumulator = dict()
        self._calculate_loss_and_gradient = make_loss_and_gradient_function(
            loss_checkpoint_chungs=self.loss_checkpoint_chungs,
        )
        self.layer_manager = LayerManager(
            self.model,
            self.logging_interval_light,
            self.logging_interval_heavy,
            self.steps_until_start_temperature_learn,
        )
        # if temp training is delayed, turn if off for now
        self.layer_manager.manage_learnable_temperature(0)
        self._check_config()

    def _before_train_operations(self):
        propagate_forward_pass_cache(self.model)
        update_model_fit_gpu_info(
            self.model_fit_gpu_info_database_path,
            self.model_fit_gpu_info_params,
            "failure",
        )

    def _after_train_operations(self):
        update_model_fit_gpu_info(
            self.model_fit_gpu_info_database_path,
            self.model_fit_gpu_info_params,
            "success",
        )

    def _after_step_operations(self, step):
        self.model.forward_pass_cache.clear()
        self.layer_manager.manage_learnable_temperature(step)

    def train(self, n_steps: int):
        """
        Train the model for n_steps steps.
        """
        self._before_train_operations()
        if self.scaler is not None and self.checkpoint is not None:
            load_scaler_state(self.scaler, self.checkpoint)

        self.ts_start_training = time()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=self.profiler_schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                self.profiler_trace_path
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
        ) as p:
            for step in range(self.start_step, n_steps + 1):
                self._train_step(step)
                if self.profiler_enabled:
                    p.step()

                if (
                    step > 0
                    and self.eval_interval > 0
                    and step % self.eval_interval == 0
                ):
                    self._eval_step(step)

                if (
                    step > 0
                    and self.model_type == "gpt"
                    and self.decoding_interval > 0
                    and step % self.decoding_interval == 0
                    and self.is_logging_process
                ):
                    try:  # dev
                        self._decode_samples(step)
                    except:
                        print("Decoding failed, skipping...")
                self._after_step_operations(step)

    def _train_step(
        self,
        step,
    ):
        self.model.train()
        if self.is_logging_process:
            self.layer_manager.prepare_for_logging(step)
        processed_batch = self.train_dataloader.get_batch()

        self.lr_scheduler.set_lr(step=step, optimizer=self.optimizer)
        loss, aux_info = self.calculate_loss_and_gradient(processed_batch)
        self.fb_time_acc += aux_info["fb_time"]
        self.fb_time_acc_interval += aux_info["fb_time"]

        self._apply_gradient()
        if self.is_logging_process:
            self._log_train_stats(loss, step)
            self._log_acc_stats(self.byttok_loss, aux_info["byttok_loss"], step)
            self._log_acc_stats(self.deftok_loss, loss, step)
            self._log_acc_time_stats(
                self.fb_time_byttok_loss,
                aux_info["byttok_loss"],
                step,
                int(self.fb_time_acc),
            )
            self._log_acc_time_stats(
                self.fb_time_deftok_loss, loss, step, int(self.fb_time_acc)
            )
            self._log_avg_time_interval(100, "average/time/fb", step)
            self._log_accuracy(aux_info, step)
            self.layer_manager.log(step)
            self._log_weights_and_gradients(step)
            self._log_auxiliary_losses(aux_info["losses"], step)
        self._save_weights(step)

    def calculate_loss_and_gradient(self, processed_batch: LLMBatch):
        """gradient accumulation: slice the batch into minibatches, get gradients from each, then average and apply them
        NOTE: this function will not set the gradients for the model if model is in eval mode
        """
        total_cross_entropy_loss = 0.0
        correct_tokens_value = 0
        total_masked_tokens_value = 0
        losses = {}
        byttok_loss_acc = 0
        fb_time_acc = 0

        for i in range(self.gradient_accumulation_steps):
            # TODO: make a way to avoid copying the whole batch just to get a slice
            batch_copy = copy.deepcopy(processed_batch)
            for _, tensor in batch_copy:
                tensor.data = get_ith_chunk(
                    tensor.data, self.gradient_accumulation_steps, i
                )

            ts_start_fb = time()
            cross_entropy_loss, aux_info = self._calculate_loss_and_gradient(
                batch=batch_copy,
                model=self.model,
                mixed_precision=self.mixed_precision,
                mixed_precision_dtype=self.mixed_precision_dtype,
                num_checkpoint_accumulation_steps=self.gradient_accumulation_steps,
                scaler=self.scaler,
            )
            fb_time = time() - ts_start_fb
            fb_time_acc += fb_time

            total_cross_entropy_loss += cross_entropy_loss
            correct_tokens_value += aux_info["correct_tokens"]
            total_masked_tokens_value += aux_info["total_masked_tokens"]

            byttok_loss_acc += aux_info["byttok_loss"]
            for key, value in aux_info["losses"].items():
                losses[key] = losses.get(key, 0) + value.item()

        return total_cross_entropy_loss, {
            "correct_tokens": correct_tokens_value,
            "total_masked_tokens": total_masked_tokens_value,
            "losses": losses,
            "byttok_loss": byttok_loss_acc,
            "fb_time": fb_time_acc,
        }

    def _apply_gradient(self):
        if self.scaler is None:
            if self.gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clipping
                )
            self.optimizer.step()
        else:
            if self.gradient_clipping is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clipping
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        self.optimizer.zero_grad()

    def _eval_step(self, step: int):
        batches = [self.eval_dataloader.get_batch() for _ in range(self.n_eval_batches)]
        self._eval_single_variant(
            batches=batches,
            step=step,
            variant_name="normal",
        )

    def _eval_single_variant(
        self, batches: Iterable[LLMBatch], step: int, variant_name: str
    ):
        self.model.eval()
        total_loss = 0.0
        total_correct_tokens = 0
        total_masked_tokens = 0
        extra_losses = defaultdict(float)
        for processed_batch in batches:
            with torch.no_grad():
                loss, aux_info = self.calculate_loss_and_gradient(processed_batch)
            total_loss += loss
            total_correct_tokens += aux_info["correct_tokens"]
            total_masked_tokens += aux_info["total_masked_tokens"]
            for name, loss_value in aux_info["losses"].items():
                extra_losses[name] += loss_value
        if self.is_logging_process:
            self.logger.report_scalar(
                title=f"eval/total_loss/{variant_name}",
                value=total_loss / self.n_eval_batches,
                iteration=step,
            )
            self.logger.report_scalar(
                title=f"eval/accuracy/{variant_name}",
                value=total_correct_tokens / total_masked_tokens,
                iteration=step,
            )
            for name, loss_value in extra_losses.items():
                self.logger.report_scalar(
                    title=f"eval/{name}/{variant_name}",
                    value=loss_value / self.n_eval_batches,
                    iteration=step,
                )

    @staticmethod
    def decode_single_example(
        model: torch.nn.Module,
        max_sequence_length: int,
        input_tokens_ids: torch.Tensor,
        end_token_id: int,
    ) -> torch.Tensor:
        output_tokens_ids = torch.nn.functional.pad(
            input_tokens_ids, (0, max_sequence_length - len(input_tokens_ids[0]))
        )
        output_length = len(input_tokens_ids[0])
        model.eval()

        with torch.no_grad():
            while True:
                predictions = model(output_tokens_ids)[0]
                next_token_id = torch.argmax(predictions, dim=-1)[
                    output_length - 1
                ].item()
                output_tokens_ids[0][output_length] = next_token_id
                output_length += 1
                if (
                    output_length == max_sequence_length
                    or next_token_id == end_token_id
                ):
                    break
        return output_tokens_ids[0][:output_length].to("cpu").numpy()

    def _decode_samples(self, step):
        examples = [
            "1, 2, 3, 4, 5",
            "Our Father, who art in heaven,",
            "Warsaw -> Poland Paris -> France Berlin ->",
            "Speech at a funeral of a fly: ",
        ]
        # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer = ReversedGPT2Tokenizer(
            GPT2Tokenizer.from_pretrained("research/tokenizex/model/reversed_tokenizer")
        )

        for example in examples:
            tokens = torch.tensor(
                [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example))]
            ).to(self.train_dataloader.device)
            output_tokens = TemplateTrainer.decode_single_example(
                model=self.model,
                max_sequence_length=self.max_sequence_length,
                input_tokens_ids=tokens,
                end_token_id=tokenizer.convert_tokens_to_ids(tokenizer.eos_token),
            )
            decoded_output = tokenizer.decode(output_tokens)
            print(f"{example}: {decoded_output}")
            self.logger.report_text(
                title=f"decoding_sample/{example}",
                value=decoded_output,
                iteration=step,
            )

    def _log_train_stats(self, loss_value, step):
        self.logger.report_scalar(title="step", value=step, iteration=step)
        self.logger.report_scalar(
            title="lr", value=self.lr_scheduler.get_lr(step=step), iteration=step
        )
        if self.dataset_type == "c4":
            self._log_fraction_dataset_processed(step)
        for name, stats in self.loss_accumulators.items():
            stats.acc += loss_value
            if stats.interval > 0 and step > 0 and step % stats.interval == 0:
                self.logger.report_scalar(
                    title=name,
                    value=stats.acc / stats.interval,
                    iteration=step,
                )
                stats.acc = 0.0

    def _log_weights_and_gradients(self, step):
        g_norms, w_norms = {}, {}
        if (
            self.logging_interval_heavy > 0
            and step % self.logging_interval_heavy == 0
            and step > 0
            and self.log_gradients_and_weights
        ):
            for name, value in self.model.named_parameters():
                if value.grad is not None:
                    norm = torch.linalg.norm(value.grad)
                    g_norms[f"weight_norms/{name.replace('.', '/')}/grad"] = norm
                if value.requires_grad:
                    norm = torch.linalg.norm(value)
                    w_norms[f"weight_norms/{name.replace('.', '/')}/weight"] = norm
            g_norms["weight_norms/grad_norm_total"] = torch.linalg.norm(
                torch.tensor(list(g_norms.values()))
            )
            w_norms["weight_norms/weight_norm_total"] = torch.linalg.norm(
                torch.tensor(list(w_norms.values()))
            )
            for name, value in {**g_norms, **w_norms}.items():
                self.logger.report_scalar(title=name, value=value, iteration=step)

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

    def _log_acc_stats(self, stat_accumulator, stat_value, step):
        for name, stats in stat_accumulator.items():
            stats.acc += stat_value
            if stats.interval > 0 and step > 0 and step % stats.interval == 0:
                self.logger.report_scalar(
                    title=name,
                    value=stats.acc / stats.interval,
                    iteration=step,
                )
                stats.acc = 0.0

    def _log_acc_time_stats(self, stat_accumulator, stat_value, step, time_passed):
        for name, stats in stat_accumulator.items():
            stats.acc += stat_value
            if (
                stats.interval > 0
                and time_passed > 0
                and (time_passed - stats.last_time) > stats.interval
            ):
                self.logger.report_scalar(
                    title=name,
                    value=stats.acc
                    / (
                        1 if (step - stats.last_step) == 0 else (step - stats.last_step)
                    ),
                    iteration=time_passed,
                )
                stats.last_time = time_passed
                stats.last_step = step
                stats.acc = 0.0

    def _log_avg_time_interval(self, interval, name, iteration):
        if iteration % interval == 0 and iteration > 0:
            self.logger.report_scalar(
                title=name,
                value=self.fb_time_acc_interval / interval,
                iteration=iteration,
            )
            self.fb_time_acc_interval = 0.0

    def _save_weights(self, step):
        if (
            self.save_weights_path is not None
            and self.save_weights_interval > 0
            and step % self.save_weights_interval == 0
        ):
            save_checkpoint(
                self.model,
                self.optimizer,
                self.scaler,
                self.save_weights_path,
                self.rank,
                step,
            )

    def _check_config(self):
        if self.eval_dynamic_groupsize:
            assert self.eval_max_group_size_logfactor is not None
            assert self.eval_min_group_size_logfactor is not None
            assert (
                self.eval_min_group_size_logfactor <= self.eval_max_group_size_logfactor
            )
