from typing import Optional, Dict

import torch
import torch.nn.functional as F
from attr import define

from lizrd.datasets import wikibookdata
from research.nonlinearities.core.misc_logging import (
    register_activation_hooks,
    log_tensor_distribution,
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
    batch_size: int
    vocab_size: int
    mask_percent: float
    logging_frequency: int
    mixed_precision: bool = False
    scaler: Optional[torch.cuda.amp.GradScaler] = None
    distribution_logging: bool = False
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

        self.optimize(mask_loss)
        self.log_distributions(step)
        self.detach_logging_hooks(step)

        if step:
            log_scalar(
                name="loss/train",
                value=mask_loss.item(),
                step=step,
                series="train",
            )

    def train(self, n_steps: int):
        for step in range(n_steps):
            self._train_step(step)
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
                tensor_clean, nan_frequency = process_and_remove_nan(tensor)
                log_tensor_distribution(
                    tensor=tensor_clean, name=f"{name} weight", series=series, step=step
                )
                if tensor.grad is not None:
                    grad_clean, nan_frequency = process_and_remove_nan(tensor.grad)
                    log_tensor_distribution(
                        tensor=grad_clean, name=f"{name} grad", series=series, step=step
                    )
        for name, tensor in self.saved_activations.items():
            series, name = clean_name_for_logging(name)
            activations_clean, nan_frequency = process_and_remove_nan(tensor)
            log_tensor_distribution(
                tensor=activations_clean,
                name=f"{name} activation",
                series=series,
                step=step,
            )
