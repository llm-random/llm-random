from abc import ABC
import math

from torch.optim import Optimizer


def get_scheduler(
    scheduler_type,
    lr_warmup_steps,
    learning_rate,
    final_lr_step,
    final_lr_fraction,
):
    if scheduler_type == "constant":
        return ConstantScheduler(
            lr_warmup_steps=lr_warmup_steps, lr=learning_rate
        )
    elif scheduler_type == "cosine":
        return CosineScheduler(
            lr_warmup_steps=lr_warmup_steps,
            lr=learning_rate,
            final_lr_step=final_lr_step,
            final_lr_fraction=final_lr_fraction,
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


class AbstractLRScheduler(ABC):
    def get_lr(self, step):
        raise NotImplementedError

    def set_lr(self, optimizer: Optimizer, step: int):
        new_lr = self.get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr


class ConstantScheduler(AbstractLRScheduler):
    def __init__(self, lr_warmup_steps: int, lr: float):
        self.lr_warmup_steps = lr_warmup_steps
        self.lr = lr

    def get_lr(self, step: int):
        if step < self.lr_warmup_steps:
            return self.lr * (step + 1) / self.lr_warmup_steps
        else:
            return self.lr


class CosineScheduler(AbstractLRScheduler):
    def __init__(
        self,
        lr_warmup_steps: int,
        lr: float,
        final_lr_step: int,
        final_lr_fraction: float,
    ):
        assert isinstance(lr_warmup_steps, int)
        assert isinstance(lr, float)
        assert isinstance(final_lr_step, int)
        assert isinstance(final_lr_fraction, float)

        self.lr_warmup_steps = lr_warmup_steps
        self.lr = lr
        self.final_lr_step = final_lr_step
        self.final_lr_fraction = final_lr_fraction

    def get_lr(self, step: int):
        if step < self.lr_warmup_steps:
            return self.lr * (step + 1) / self.lr_warmup_steps
        # cosine schedule that ends at final_lr_fraction * lr, then constant
        elif step < self.final_lr_step:
            return self.final_lr_fraction * self.lr + 0.5 * (
                1 - self.final_lr_fraction
            ) * self.lr * (
                1
                + math.cos(
                    math.pi
                    * (step - self.lr_warmup_steps)
                    / (self.final_lr_step - self.lr_warmup_steps)
                )
            )
        else:
            return self.lr * self.final_lr_fraction
