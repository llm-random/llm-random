from collections import defaultdict
import copy
from time import time
from types import SimpleNamespace as SN
from typing import Callable, Iterable, Optional, Literal

import torch
from torch.profiler import profile, ProfilerActivity
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from attr import define
from lizrd.core.misc import propagate_forward_pass_cache
from lizrd.support.decoding import decode_single_example
from lizrd.support.logging import AbstractLogger
from lizrd.support.misc import (
    convert_steps_to_tokens,
    get_ith_chunk,
    get_batch_size,
)
from lizrd.text.data import LLMBatch
from lizrd.train.checkpoints_manager import (
    create_slide_checkpoint,
    end_training_checkpoint,
    job_out_of_time_checkpoint,
)
from lizrd.train.scheduler import AbstractLRScheduler
from research.batch_size_rampup_config import BatchSizeRampupConfig
from research.conditional.moe_layers.continuous_moe import ContinuousMoE
from research.conditional.moe_layers._expert_choice_old import ExpertChoiceFFOld
from research.conditional.moe_layers.expert_choice import ExpertChoiceFF
from research.conditional.utils.layer_manager import LayerManager
from research.conditional.utils.misc_tools import get_slurm_job_id, temp_modify_attr
from research.conditional.utils.model_utils import (
    make_loss_and_gradient_function,
    update_model_fit_gpu_info,
)
from research.datasets import DataloaderWrapper
from lizrd.text.datasets import C4Dataset
from transformers import GPT2Tokenizer
from lizrd.train.load_and_save_model import load_scaler_state, save_checkpoint


@define(slots=False)
class ConditionalTrainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    train_dataloader: DataloaderWrapper
    eval_dataloader: Optional[DataloaderWrapper]
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
    cutoff: int
    lr_scheduler: AbstractLRScheduler
    repeater_job_end_time: int = None
    _calculate_loss_and_gradient: Optional[Callable] = None
    mask_percent: Optional[float] = None
    scaler: Optional[torch.cuda.amp.GradScaler] = None
    layer_manager: Optional[LayerManager] = None
    loss_accumulator: Optional[float] = None
    n_gpus: int = 1
    save_weights_path: Optional[str] = None
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
    model_fit_gpu_info_params: Optional[str] = None
    profiler_enabled: bool = False
    profiler_trace_path: str = None
    profiler_schedule: None = None
    rank: Optional[int] = None
    start_step: int = 0
    batch_size_rampup_config: Optional[BatchSizeRampupConfig] = None
    checkpoint: Optional[dict[str, torch.Tensor]] = None
    scheduler_trapezoidal_slides: Optional[list[dict]] = None
    args_override: Optional[dict] = None
    get_final_eval_dataloader: Optional[Callable[..., DataloaderWrapper]] = None
    final_eval_dataloader_batch_size: Optional[int] = None
    n_final_eval_batches: int = None
    loaded_training_loop_accumulators: dict = None
    model_active_params:int = 1
    gpu_flops: int = 1

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
        self.correct_tokens_accumulator = 0.0
        self.total_tokens_accumulator = 0.0
        self.auxiliary_losses_accumulator = dict()
        self.other_training_states = dict()
        self.other_training_states['step_fb_time_acc_sec'] = 0.0
        self.other_training_states['last_mfu_fb_time_state_sec'] = 0.0
        self.other_training_states['last_mfu_tokens_state'] = 0.0
        
        if self.loaded_training_loop_accumulators:
            assert list(self.loss_accumulators.keys()) == list(self.loaded_training_loop_accumulators["loss_accumulators"].keys())
            # assert list(self.auxiliary_losses_accumulator.keys()) == list(self.loaded_training_loop_accumulators["auxiliary_losses_accumulator"].keys()) #dev TODO validate this, to have loaded model - config coherence

            self.loss_accumulators = self.loaded_training_loop_accumulators["loss_accumulators"]
            self.correct_tokens_accumulator = self.loaded_training_loop_accumulators["correct_tokens_accumulator"]
            self.total_tokens_accumulator = self.loaded_training_loop_accumulators["total_tokens_accumulator"]
            self.auxiliary_losses_accumulator = self.loaded_training_loop_accumulators["auxiliary_losses_accumulator"]
            self.other_training_states = self.loaded_training_loop_accumulators["other_training_states"]
                
        self._calculate_loss_and_gradient = make_loss_and_gradient_function(
            loss_checkpoint_chungs=self.loss_checkpoint_chungs,
        )
        self.layer_manager = LayerManager(
            self.model,
            self.logging_interval_light,
            self.logging_interval_heavy,
            self.steps_until_start_temperature_learn,
        )
        self.n_devices = (
            self.n_gpus if self.n_gpus != 0 else 1
        )  # self.n_gpus is 0 when the model is run on cpu
        # if temp training is delayed, turn if off for now
        self.layer_manager.manage_learnable_temperature(0)
        self._check_config()

    def _before_train_operations(self):
        if self.is_logging_process:
            self.logger.start_job_metadata(self.start_step)
        propagate_forward_pass_cache(self.model)
        update_model_fit_gpu_info(
            self.model_fit_gpu_info_database_path,
            self.model_fit_gpu_info_params,
            "failure",
        )
        self.num_processed_tokens = 0

    def _after_train_operations(
        self, n_steps: int
    ):  # TODO move n_steps form train method args to training class properties
        update_model_fit_gpu_info(
            self.model_fit_gpu_info_database_path,
            self.model_fit_gpu_info_params,
            "success",
        )

        if self.current_step >= n_steps:  # - end of model training operations
            if self.is_logging_process:
                self.logger.exit_job_metadata(self.current_step)
            if self.save_weights_path:
                job_id = get_slurm_job_id()
                end_training_checkpoint(
                    job_id,
                    self.is_logging_process,
                    self.model,
                    self.optimizer,
                    self.scaler,
                    self.save_weights_path,
                    self.rank,
                    self.current_step,
                    self.batch_size,
                    self.cutoff,
                    self.logger.loggers if self.is_logging_process else None,
                    self.loss_accumulators, 
                    self.correct_tokens_accumulator,
                    self.total_tokens_accumulator,
                    self.auxiliary_losses_accumulator,
                    self.other_training_states,
                    self.args_override,
                )

    def _after_step_operations(self, step):
        self.model.forward_pass_cache.clear()
        self.layer_manager.manage_learnable_temperature(step)

    def _final_eval(
        self,
        n_steps: int,
    ):
        if self.current_step == n_steps and self.n_final_eval_batches > 0:
            del self.train_dataloader
            del self.eval_dataloader
            final_eval_dataloader = self.get_final_eval_dataloader()
            self.model.eval()
            losses = []
            for _ in range(self.n_final_eval_batches):
                batch = final_eval_dataloader.get_batch()
                with torch.no_grad():
                    loss, _ = self.calculate_loss_and_gradient(
                        batch, num_batch_chunks=1
                    )
                    losses.append(loss)

            losses_average = torch.tensor(losses, dtype=torch.float64).mean()

            if self.is_logging_process:
                self.logger.report_scalar(
                    title=f"final_eval",
                    value=losses_average.item(),
                    iteration=n_steps,
                )

    def train(self, n_steps: int):
        """
        Train the model for n_steps steps.
        """
        self._before_train_operations()
        if self.scaler is not None and self.checkpoint is not None:
            load_scaler_state(self.scaler, self.checkpoint)
        self.n_steps = n_steps + 1 #def TODO

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
                self.current_step = step
                self._train_step(step)
                if self._repeater_rerun(step, self.repeater_job_end_time):
                    break

                if self.profiler_enabled:
                    p.step()

                if (
                    step > 0
                    and self.eval_interval > 0
                    and step % self.eval_interval == 0
                    and self.eval_dataloader is not None
                ):
                    self._eval_step(step)
                if (
                    self.model_type == "gpt"
                    and self.decoding_interval > 0
                    and step % self.decoding_interval == 0
                    and self.is_logging_process
                ):
                    try:
                        self._decode_samples(step)
                    except:
                        print("Decoding failed, skipping...")
                if self.scheduler_trapezoidal_slides:
                    for slide in self.scheduler_trapezoidal_slides:
                        if step == slide["split_step"]:
                            split_loggers = None
                            if self.is_logging_process:
                                split_loggers = [self.logger.loggers[0]]
                                del self.logger.loggers[0]
                            create_slide_checkpoint(
                                get_slurm_job_id(),
                                self.is_logging_process,
                                self.model,
                                self.optimizer,
                                self.scaler,
                                self.save_weights_path,
                                self.rank,
                                step,
                                self.batch_size,
                                self.cutoff,
                                split_loggers,
                                self.loss_accumulators,
                                self.correct_tokens_accumulator,
                                self.total_tokens_accumulator,
                                self.auxiliary_losses_accumulator,
                                self.other_training_states,
                                args_override={
                                    "n_steps": slide["n_steps"],
                                    "scheduler_trapezoidal_slides": None,
                                },
                            )
                self._final_eval(n_steps)
                self._after_step_operations(step)
        self._after_train_operations(n_steps)

    def _train_step(
        self,
        step,
    ):
        self.model.train()
        if self.is_logging_process:
            self.layer_manager.prepare_for_logging(step)

        if self.batch_size_rampup_config is None:
            current_batch_size_per_gpu = self.batch_size // self.n_devices
            num_processed_tokens = convert_steps_to_tokens(
                step=step,
                seq_len=self.cutoff,
                target_batch_size=self.batch_size,
            )
        else:
            current_batch_size_per_gpu = (
                get_batch_size(
                    step,
                    target_batch_size=self.batch_size,
                    transition_points=self.batch_size_rampup_config.transition_points,
                    batch_sizes=self.batch_size_rampup_config.batch_sizes,
                )
                // self.n_devices
            )
            num_processed_tokens = convert_steps_to_tokens(
                step=step,
                seq_len=self.cutoff,
                target_batch_size=self.batch_size,
                transition_points=self.batch_size_rampup_config.transition_points,
                batch_sizes=self.batch_size_rampup_config.batch_sizes,
            )
        self.num_processed_tokens = num_processed_tokens
        processed_batch = self.train_dataloader.get_batch(
            current_batch_size_per_gpu=current_batch_size_per_gpu,
        )

        self.lr_scheduler.set_lr(step=step, optimizer=self.optimizer)
        num_batch_chunks = calculate_num_batch_chunks(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            target_batch_size=self.batch_size,
            current_batch_size=current_batch_size_per_gpu * self.n_devices,
        )
        fb_start = time()
        loss, aux_info = self.calculate_loss_and_gradient(
            processed_batch, num_batch_chunks=num_batch_chunks
        )
        if self.rank is not None:
            dist.all_reduce(torch.tensor(loss, device="cuda"), op=dist.ReduceOp.AVG)
        self._apply_gradient()
        self.other_training_states['step_fb_time_acc_sec'] += time() - fb_start

        if self.is_logging_process:
            self._log_train_stats(
                loss,
                step,
                current_batch_size_per_gpu * self.n_devices,
                num_processed_tokens,
            )
            self._log_accuracy(aux_info, step)
            self._log_mfu(num_processed_tokens, step)
            self._log_progress(step)
            self.layer_manager.log(step)
            self._log_weights_and_gradients(step)
            self._log_auxiliary_losses(aux_info["losses"], step)
        self._save_weights(step)

    def calculate_loss_and_gradient(
        self, processed_batch: LLMBatch, num_batch_chunks: int
    ):
        """gradient accumulation: slice the batch into minibatches, get gradients from each, then average and apply them
        NOTE: this function will not set the gradients for the model if model is in eval mode
        """
        total_cross_entropy_loss = 0.0
        correct_tokens_value = 0
        total_masked_tokens_value = 0
        losses = {}

        for i in range(num_batch_chunks):
            # TODO: make a way to avoid copying the whole batch just to get a slice
            batch_copy = copy.deepcopy(processed_batch)
            for _, tensor in batch_copy:
                tensor.data = get_ith_chunk(
                    tensor.data,
                    num_batch_chunks,
                    i,
                )

            cross_entropy_loss, aux_info = self._calculate_loss_and_gradient(
                batch=batch_copy,
                model=self.model,
                mixed_precision=self.mixed_precision,
                mixed_precision_dtype=self.mixed_precision_dtype,
                num_checkpoint_accumulation_steps=num_batch_chunks,
                scaler=self.scaler,
            )

            total_cross_entropy_loss += cross_entropy_loss
            correct_tokens_value += aux_info["correct_tokens"]
            total_masked_tokens_value += aux_info["total_masked_tokens"]

            for key, value in aux_info["losses"].items():
                losses[key] = losses.get(key, 0) + value.item()

        return total_cross_entropy_loss, {
            "correct_tokens": correct_tokens_value,
            "total_masked_tokens": total_masked_tokens_value,
            "losses": losses,
        }

    def _apply_gradient(self):
        if self.scaler is None:
            if self.gradient_clipping is not None:
                if isinstance(self.model, FSDP):
                    self.model.clip_grad_norm_(self.gradient_clipping)
                else:
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
        layers = [
            l
            for _, l in self.layer_manager._layers
            if isinstance(
                l,
                (
                    ContinuousMoE,
                    ExpertChoiceFFOld,
                    ExpertChoiceFF,
                ),
            )
        ]
        if self.eval_dynamic_groupsize:
            original_group_size = layers[0].group_size
            for log_group_size_factor in range(
                self.eval_min_group_size_logfactor,
                self.eval_max_group_size_logfactor + 1,
            ):
                current_group_size = int(
                    2**log_group_size_factor * original_group_size
                )
                if (
                    current_group_size
                    <= self.batch_size // self.gradient_accumulation_steps
                    and current_group_size > 0
                ):
                    with temp_modify_attr(layers, "group_size", current_group_size):
                        self._eval_single_variant(
                            batches=batches,
                            step=step,
                            variant_name=f"group size={current_group_size}",
                        )

        if self.eval_discrete_mot:
            with temp_modify_attr(layers, "use_discrete_routing", True):
                self._eval_single_variant(
                    batches=batches,
                    step=step,
                    variant_name="discrete MoT routing",
                )

    def _eval_single_variant(
        self, batches: Iterable[LLMBatch], step: int, variant_name: str
    ):
        self.model.eval()
        total_loss = 0.0
        total_correct_tokens = 0
        total_masked_tokens = 0
        extra_losses = defaultdict(float)
        num_batch_chuks = calculate_num_batch_chunks(
            gradient_accumulation_steps=self.gradient_accumulation_steps
        )
        for processed_batch in batches:
            with torch.no_grad():
                loss, aux_info = self.calculate_loss_and_gradient(
                    processed_batch=processed_batch, num_batch_chunks=num_batch_chuks
                )
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

    def _decode_samples(self, step):
        examples = [
            "1, 2, 3, 4, 5",
            "Our Father, who art in heaven,",
            "Warsaw -> Poland Paris -> France Berlin ->",
            "Speech at a funeral of a fly: ",
        ]
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        for example in examples:
            tokens = torch.tensor(
                tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example))
            ).to(self.train_dataloader.device)
            output_tokens = decode_single_example(
                self.model,
                self.max_sequence_length,
                tokens,
                tokenizer._convert_token_to_id("<|endoftext|>"),
            )
            decoded_output = tokenizer.decode(output_tokens)
            print(f"{example}: {decoded_output}")
            self.logger.report_text(
                title=f"decoding_sample/{example}",
                value=decoded_output,
                iteration=step,
            )

    def _log_train_stats(
        self, loss_value, step, current_batch_size, num_processed_tokens
    ):
        self.logger.report_scalar(title="step", value=step, iteration=step)
        self.logger.report_scalar(
            title="lr", value=self.lr_scheduler.get_lr(step=step), iteration=step
        )
        self.logger.report_scalar(
            title="batch_size", value=current_batch_size, iteration=step
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
                    processed_tokens_so_far=num_processed_tokens,
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
            g_norms[f"weight_norms/grad_norm_total"] = torch.linalg.norm(
                torch.tensor(list(g_norms.values()))
            )
            w_norms[f"weight_norms/weight_norm_total"] = torch.linalg.norm(
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
    def _log_mfu(self, total_processed_tokens, step):
        if step % self.logging_interval_loss == 0 and step > 0:
            model_flops = self.model_active_params * 6 * (total_processed_tokens - self.other_training_states['last_mfu_tokens_state']) / (self.other_training_states['step_fb_time_acc_sec'] - self.other_training_states['last_mfu_fb_time_state_sec'])
            self.logger.report_scalar(
                title=f"Model FLOPS",
                value=model_flops,
                iteration=step,
            )
            self.logger.report_scalar(
                title=f"MFU",
                value=model_flops/(self.n_gpus*self.gpu_flops),
                iteration=step,
            )
            self.other_training_states['last_mfu_tokens_state'] = total_processed_tokens
            self.other_training_states['last_mfu_fb_time_state_sec'] = self.other_training_states['step_fb_time_acc_sec']

    def _log_progress(self, step):
        if step >= self.n_steps-1:
            self.logger.report_scalar(
                title=f"Experiment progress",
                value=float(1),
                iteration=step,
            )
        elif step % self.logging_interval_loss == 0 and step > 0:            
            self.logger.report_scalar(
                title=f"Experiment progress",
                value=float(step/self.n_steps),
                iteration=step,
            )
        


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
                self.batch_size,
                self.cutoff,
                self.logger.loggers,
                loss_accumulators = self.loss_accumulators,
                correct_tokens_accumulator = self.correct_tokens_accumulator,
                total_tokens_accumulator = self.total_tokens_accumulator,
                auxiliary_losses_accumulator = self.auxiliary_losses_accumulator,
                other_training_states = self.other_training_states
            )

    def _repeater_rerun(
        self, step, repeater_job_end_time: Optional[int], buffer=45 * 60 #dev TODO onece was too short in constrained 190x32v2
    ) -> bool:
        if repeater_job_end_time and ((repeater_job_end_time - time())) < buffer:
            job_id = get_slurm_job_id()
            job_out_of_time_checkpoint(
                job_id,
                self.is_logging_process,
                self.model,
                self.optimizer,
                self.scaler,
                self.save_weights_path,
                self.rank,
                step,
                self.batch_size,
                self.cutoff,
                self.logger.loggers if self.is_logging_process else None,
                self.loss_accumulators,
                self.correct_tokens_accumulator,
                self.total_tokens_accumulator,
                self.auxiliary_losses_accumulator,
                self.other_training_states,
                self.args_override,
            )

            return True
        else:
            return False

    def _check_config(self):
        if self.eval_dynamic_groupsize:
            assert self.eval_max_group_size_logfactor is not None
            assert self.eval_min_group_size_logfactor is not None
            assert (
                self.eval_min_group_size_logfactor <= self.eval_max_group_size_logfactor
            )


def calculate_num_batch_chunks(
    gradient_accumulation_steps, target_batch_size=None, current_batch_size=None
):
    if target_batch_size is None:
        return gradient_accumulation_steps
    else:
        if current_batch_size is None:
            rampup_factor = 1
        else:
            rampup_factor = target_batch_size // current_batch_size
        num_batch_chunks = max(gradient_accumulation_steps // rampup_factor, 1)
        return num_batch_chunks
