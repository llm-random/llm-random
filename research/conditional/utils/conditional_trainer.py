from typing import Optional

import torch
from attr import define

from lizrd.datasets import wikibookdata
from lizrd.support.logging import AbstractLogger
from research.conditional.moe_layers.continuous_moe import ContinuousMoE
from research.conditional.utils.layer_manager import LayerManager
from research.conditional.utils.model_utils import make_loss_function


@define(slots=False)
class ConditionalTrainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    train_dataloader: wikibookdata.ProcessedDatasetWrapper
    batch_size: int
    vocab_size: int
    mixed_precision: bool
    logger: Optional[AbstractLogger]
    model_type: str
    logging_interval_loss: int
    logging_interval_light: int
    logging_interval_heavy: int
    _calculate_loss: Optional[callable] = None
    mask_percent: Optional[float] = None
    scaler: Optional[torch.cuda.amp.GradScaler] = None
    layer_manager: Optional[LayerManager] = None
    loss_accumulator: Optional[float] = None
    n_gpus: int = 1
    hack_name: str = None
    loss_checkpoint_chungs: int = 0

    def __attrs_post_init__(self):
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.loss_accumulator = 0.0
        self._calculate_loss = make_loss_function(
            model=self.model_type,
            loss_checkpoint_chungs=self.loss_checkpoint_chungs,
            mask_percentage=self.mask_percent,
        )
        self.layer_manager = LayerManager(
            self.model, self.logging_interval_light, self.logging_interval_heavy
        )

    def train(self, n_steps: int):
        for step in range(n_steps + 1):
            if self.hack_name is not None:
                self._hack(self.hack_name, step)
            else:
                self._train_step(step)
            if step % 1000 == 0:
                print(f"Step {step}")

    def _optimize(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _train_step(
        self,
        step,
    ):
        self.model.train()
        if self.logger is not None:
            self.layer_manager.prepare_for_logging(step)
        processed_batch = self.train_dataloader.get_batch()
        assert isinstance(processed_batch, wikibookdata.ProcessedBatch)

        breakpoint()
        loss = self._calculate_loss(
            batch=processed_batch,
            model=self.model,
            mixed_precision=self.mixed_precision,
            vocab_size=self.vocab_size,
        )
        self._optimize(loss)
        if self.logger is not None:
            self._log_loss(loss, step)
            self.layer_manager.log(step)

    def _log_loss(self, loss, step):
        self.logger.report_scalar(title="step", value=step, iteration=step)
        self.loss_accumulator += loss.item()
        if step % self.logging_interval_loss == 0 and step > 0:
            self.logger.report_scalar(
                title="loss",
                value=self.loss_accumulator / self.logging_interval_loss,
                iteration=step,
            )
            self.loss_accumulator = 0.0

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
        processed_batch = self.train_dataloader.get_batch()
        assert isinstance(processed_batch, wikibookdata.ProcessedBatch)
        for tensor in vars(processed_batch).values():
            if hasattr(tensor, "shape"):
                tensor.data = tensor[:1].repeat(step + 1, 1).data
        loss = self._calculate_loss(
            processed_batch, self.mixed_precision, self.mask_percent, self.vocab_size
        )
        self._optimize(loss)
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
        processed_batch = self.train_dataloader.get_batch()
        assert isinstance(processed_batch, wikibookdata.ProcessedBatch)
        for block_name, layer in self.layer_manager._layers:
            layer.expertsize = step + 1
            layer.init_parameters()
            layer.to(torch.device("cuda"))
        loss = self._calculate_loss(
            processed_batch, self.model, self.mixed_precision, self.vocab_size
        )
        self._optimize(loss)
        self.logger.report_scalar(title="max expert size", value=step, iteration=step)
