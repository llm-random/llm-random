import os.path
import copy
from typing import Optional, Literal, List
import time
from typing import Callable, Optional, Literal

import torch
from attr import define
from lizrd.core.misc import propagate_store, propagate_names, propagate_object_names
from lizrd.datasets import wikibookdata
from lizrd.support.decoding import decode_single_example
import lizrd.datasets.processed_batch
from lizrd.support.logging import AbstractLogger
from research.conditional.moe_layers.continuous_moe import ContinuousMoE
from research.conditional.utils.layer_manager import LayerManager
from research.conditional.utils.misc_tools import get_ith_chunk
from research.conditional.utils.model_utils import make_loss_function
from lizrd.datasets.c4 import NUM_C4_TOKENS
from transformers import GPT2Tokenizer


@define(slots=False)
class ConditionalTrainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    train_dataloader: wikibookdata.ProcessedDatasetWrapper
    vocab_size: int
    mixed_precision: bool
    logger: Optional[AbstractLogger]
    model_type: Literal["bert", "gpt"]
    logging_interval_loss: int
    logging_interval_light: int
    logging_interval_heavy: int
    max_sequence_length: int
    _calculate_loss: Optional[Callable] = None
    mask_percent: Optional[float] = None
    scaler: Optional[torch.cuda.amp.GradScaler] = None
    layer_manager: Optional[LayerManager] = None
    loss_accumulator: Optional[float] = None
    n_gpus: int = 1
    hack_name: str = None
    save_weights_path: str = None
    save_weights_interval: int = 1000
    load_weights_path: str = None
    gradient_clipping: float = None
    loss_checkpoint_chungs: int = 0
    gradient_accumulation_steps: int = 1
    objects_for_propagation: Optional[List[str]] = None
    decoding_logging_steps: int = 5_000
    total_time_trainsteps: float = 0.0
    total_time_decoding: float = 0.0
    total_time_afterstep: float = 0.0

    def __attrs_post_init__(self):
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.loss_accumulator = 0.0
        self.correct_tokens_accumulator = 0.0
        self.total_tokens_accumulator = 0.0
        self._calculate_loss = make_loss_function(
            model=self.model_type,
            loss_checkpoint_chungs=self.loss_checkpoint_chungs,
        )
        self.layer_manager = LayerManager(
            self.model, self.logging_interval_light, self.logging_interval_heavy
        )

    def _restore_weights(self):
        if self.load_weights_path is not None:
            if os.path.exists(self.load_weights_path):
                print(f"Loading weights from {self.load_weights_path}")
                self.model.load_state_dict(
                    torch.load(self.load_weights_path), strict=False
                )
            else:
                print(
                    f"No weights found at {self.load_weights_path}, training from scratch"
                )

    def _save_weights(self, step):
        if (
            self.save_weights_path is not None
            and step % self.save_weights_interval == 0
        ):
            torch.save(self.model.state_dict(), self.save_weights_path)
            print(f"Weights saved to {self.save_weights_path} (step {step})")

    def _before_train_operations(self):
        propagate_store(self.model)
        propagate_names(self.model)
        propagate_object_names(self.model, self.objects_for_propagation)

    def _after_step_operations(self):
        self.model.store.clear()

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

            if self.model_type == "gpt" and step % self.decoding_logging_steps == 0:
                self._decode_samples(step)

            t2 = time.time()
            self._after_step_operations()

            t3 = time.time()

            self.total_time_trainsteps += t1 - t0
            self.total_time_decoding += t2 - t1
            self.total_time_afterstep += t3 - t2

            if step % 1000 == 0:
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

    def _optimize(self, loss, should_apply_gradient=False):
        # since we sum gradients averaged over multiple smaller batches, we need to normalize here
        loss /= self.gradient_accumulation_steps
        if self.gradient_accumulation_steps == 1:
            self.optimizer.zero_grad()
        # clear computation graph, store gradients
        self.scaler.scale(loss).backward()
        if should_apply_gradient:
            if self.gradient_clipping is not None:
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
        if self.logger is not None:
            self.layer_manager.prepare_for_logging(step)
        processed_batch: lizrd.datasets.processed_batch.ProcessedBatch = (
            self.train_dataloader.get_batch()
        )

        loss, aux_info = self.optimize_with_gradient_accumulation(processed_batch)
        if self.logger is not None:
            self._log_train_stats(loss, step)
            self._log_accuracy(aux_info, step)
            self.layer_manager.log(step)
        self._save_weights(step)

    def optimize_with_gradient_accumulation(
        self, processed_batch: wikibookdata.ProcessedBatch
    ):
        """gradient accumulation: slice the batch into minibatches, get gradients from each, then average and apply them"""
        loss_value = 0.0
        correct_tokens_value = 0
        total_masked_tokens_value = 0

        for i in range(self.gradient_accumulation_steps):
            batch_copy = copy.deepcopy(processed_batch)
            for _, tensor in batch_copy:
                tensor.data = get_ith_chunk(
                    tensor.data, self.gradient_accumulation_steps, i
                )

            loss, aux_info = self._calculate_loss(
                batch=batch_copy,
                model=self.model,
                mixed_precision=self.mixed_precision,
                vocab_size=self.vocab_size,
            )

            # clear computation graph, store gradients, only apply gradients at the end
            should_apply_gradient = i == self.gradient_accumulation_steps - 1
            self._optimize(loss, should_apply_gradient=should_apply_gradient)
            loss_value += loss.item()
            correct_tokens_value += aux_info["correct_tokens"]
            total_masked_tokens_value += aux_info["total_masked_tokens"]

        return loss_value, {
            "correct_tokens": correct_tokens_value,
            "total_masked_tokens": total_masked_tokens_value,
        }

    def _log_train_stats(self, loss_value, step):
        self.logger.report_scalar(title="step", value=step, iteration=step)
        if self.train_dataloader.dataset_type == "c4":
            self._log_fraction_dataset_processed(step)
        self.loss_accumulator += loss_value
        if step % self.logging_interval_loss == 0 and step > 0:
            self.logger.report_scalar(
                title="loss",
                value=self.loss_accumulator / self.logging_interval_loss,
                iteration=step,
            )
            self.loss_accumulator = 0.0

    def _log_fraction_dataset_processed(self, step):
        batch_size = self.train_dataloader.batch_size
        seq_len = self.train_dataloader.sequence_length
        processed = step * batch_size * seq_len
        total = NUM_C4_TOKENS
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

    def _hack(self, hack_name, step):
        if hack_name == "batch_size":
            self._hack_for_batch_size(step)
        elif hack_name == "expert_size":
            self._hack_for_contmoe_expert_size(step)
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
        processed_batch: lizrd.datasets.processed_batch.ProcessedBatch = (
            self.train_dataloader.get_batch()
        )
        for tensor in processed_batch:
            tensor.data = tensor[:1].repeat(step + 1, 1).data
        loss = self._calculate_loss(
            batch=processed_batch,
            model=self.model,
            mixed_precision=self.mixed_precision,
            vocab_size=self.vocab_size,
        )
        self._optimize(loss, should_apply_gradient=True)
        if self.logger is not None:
            self.logger.report_scalar(
                title="max batch size", value=step * self.n_gpus, iteration=step
            )

    def _hack_for_contmoe_expert_size(
        self,
        step,
    ):
        """
        This is a hack to easily determine the maximal batch size that can be used with given GPU memory and model size.
        """
        assert all(
            [
                isinstance(layer, ContinuousMoE)
                for name, layer in self.layer_manager._layers
            ]
        )
        self.model.train()
        processed_batch: lizrd.datasets.processed_batch.ProcessedBatch = (
            self.train_dataloader.get_batch()
        )
        for block_name, layer in self.layer_manager._layers:
            layer.expertsize = step + 1
            layer.init_parameters()
            layer.to(torch.device("cuda"))
        loss = self._calculate_loss(
            batch=processed_batch,
            model=self.model,
            mixed_precision=self.mixed_precision,
            vocab_size=self.vocab_size,
        )
        self._optimize(loss, should_apply_gradient=True)
        self.logger.report_scalar(title="max expert size", value=step, iteration=step)
