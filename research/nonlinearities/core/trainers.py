from typing import Optional, Dict

import torch
import torch.nn.functional as F
from attr import define
from torch.utils.tensorboard import SummaryWriter

from lizrd.datasets import wikibookdata
from research.nonlinearities.core.misc_logging import (
    register_activation_hooks,
    log_tensor,
    log_scalar,
)
from research.nonlinearities.train.utils import (
    clean_name_for_logging,
    process_and_remove_nan,
)


@define
class NonlinearityTrainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    train_dataloader: wikibookdata.ProcessedDatasetWrapper
    eval_dataloader: wikibookdata.ProcessedDatasetWrapper
    batch_size: int
    vocab_size: int
    mask_percent: float
    mask_loss_weight: float
    modelpath: str
    save_model_checkpoints: str
    mixed_precision: bool = False
    writer: Optional[SummaryWriter] = None
    scaler: Optional[torch.cuda.amp.GradScaler] = None
    distribution_logging: bool = False
    logging_frequency: int = 1000000000
    hook_handles: Optional[list] = None
    saved_activations: Optional[Dict[str, torch.Tensor]] = None

    def __attrs_post_init__(self):
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

    def optimize(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _train_step(
        self,
        step,
    ):
        self.model.train()
        processed_batch = self.train_dataloader.get_batch()
        assert isinstance(processed_batch, wikibookdata.ProcessedBatch)
        x_set = processed_batch.masked_tokens
        y_token_set = processed_batch.tokens
        y_mask_set = processed_batch.mask_mask

        self.attach_logging_hooks(step)
        with torch.autocast(
            device_type="cuda", enabled=self.mixed_precision, dtype=torch.float16
        ):
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

        self.optimize(total_loss)
        self.log_distributions(step)
        self.detach_logging_hooks(step)

        if step and self.writer:
            self.writer.add_scalar("loss/train_total", total_loss.item(), step)
            self.writer.add_scalar("loss/train_mask", mask_loss.item(), step)

    def _eval_step(self, step, sample):
        self.model.eval()
        with torch.no_grad():
            total_mask_loss = 0.0
            for _ in range(sample):
                processed_batch = self.eval_dataloader.get_batch()
                assert isinstance(processed_batch, wikibookdata.ProcessedBatch)
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

            if step and self.writer:
                self.writer.add_scalar("loss/eval_mask", total_mask_loss, step)

            return total_mask_loss

    def train(
        self,
        n_steps: int,
        n_steps_eval: int,
    ):
        for step in range(n_steps):
            self._train_step(step)
            self.writer.add_scalar("step", step, step)
            if step % n_steps_eval == 0:
                eval_loss = self._eval_step(step, sample=n_steps_eval // 2)
                print(f"Eval loss:", eval_loss)
                if self.save_model_checkpoints:
                    torch.save(self.model.state_dict(), f"{self.modelpath}/model.pt")
            if step % 500 == 0:
                print(f"Step {step}")

    def attach_logging_hooks(self, step):
        if step % self.logging_frequency == 0:
            self.saved_activations, self.hook_handles = register_activation_hooks(
                self.model
            )
            assert not all(len(m._forward_hooks) == 0 for m in self.model.modules())

    def detach_logging_hooks(self, step):
        if step % self.logging_frequency == 0:
            for hook in self.hook_handles:
                hook.remove()
            assert all(len(m._forward_hooks) == 0 for m in self.model.modules())
            self.hook_handles = []
            self.saved_activations = {}

    def log_distributions(self, step):
        if step % self.logging_frequency != 0 or not self.distribution_logging:
            return
        for tag, tensor in self.model.named_parameters():
            if "logging" in tag:
                series, name = clean_name_for_logging(tag)
                tag = f"{series}/{name}"
                tensor_clean, nan_frequency = process_and_remove_nan(tensor)
                log_tensor(tensor_clean, f"{name} weight", series, step)
                log_scalar(
                    tensor_clean.mean().item(), f"{name} weight mean", series, step
                )
                log_scalar(
                    tensor_clean.std().item(), f"{name} weight std", series, step
                )
                log_scalar(nan_frequency, f"{name} weight is_nan", series, step)
                if tensor.grad is not None:
                    grad_clean, nan_frequency = process_and_remove_nan(tensor.grad)
                    log_tensor(grad_clean, f"{name} grad", series, step)
                    log_scalar(
                        grad_clean.mean().item(), f"{tag} grad mean", series, step
                    )
                    log_scalar(grad_clean.std().item(), f"{tag} grad std", series, step)
                    log_scalar(nan_frequency, f"{tag} grad is_nan", series, step)
        for name, tensor in self.saved_activations.items():
            series, name = clean_name_for_logging(name)
            tensor_data, nan_frequency = process_and_remove_nan(tensor)
            log_tensor(tensor_data, f"{name} activation", series, step)
            log_scalar(
                tensor_data.mean().item(), f"{name} activation mean", series, step
            )
            log_scalar(tensor_data.std().item(), f"{name} activation std", series, step)
            log_scalar(nan_frequency, f"{name} activation is_nan", series, step)
