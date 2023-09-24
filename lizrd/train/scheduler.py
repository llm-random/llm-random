from abc import ABC
import math

import torch
from attr import define

from torch.optim import Optimizer


def get_scheduler(args):
    if args.scheduler == "constant":
        return ConstantScheduler(
            lr_warmup_steps=args.lr_warmup_steps, lr=args.learning_rate
        )
    elif args.scheduler == "cosine":
        return CosineScheduler(
            lr_warmup_steps=args.lr_warmup_steps,
            lr=args.learning_rate,
            final_lr_step=args.final_lr_step,
            final_lr_fraction=args.final_lr_fraction,
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")


def get_temperature_scheduler(args, model):
    if args.steps_until_start_temperature_anneal is not None:
        assert args.steps_until_start_temperature_anneal < args.n_steps
        temperature_scheduler = AbstractTemperatureScheduler(
            model,
            args.n_steps,
            args.steps_until_start_temperature_anneal,
        )
    else:
        temperature_scheduler = None
    return temperature_scheduler


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


@define
class AbstractTemperatureScheduler:
    model: torch.nn.Module
    steps_until_start_temperature_anneal: int
    steps_until_finish_temperature_anneal: int
    previous_multiplier: float = 1.0
    current_multiplier: float = 1.0

    def step(self, current_step):
        if (
            current_step > self.steps_until_start_temperature_anneal
            and current_step <= self.steps_until_finish_temperature_anneal
        ):
            self.previous_multiplier = self.current_multiplier
            self.current_multiplier = self.get_multiplier(current_step)
            self.update_model_temperature()

    def update_model_temperature(self):
        for module in self.model.modules():
            if hasattr(module, "temperature"):
                module.temperature = (
                    module.temperature
                    * self.current_multiplier
                    / self.previous_multiplier
                )
            elif hasattr(module, "temperature_merge"):
                # learnable temperature
                module.temperature_merge *= (
                    self.current_multiplier / self.previous_multiplier
                )

                module.temperature_emit *= (
                    self.current_multiplier / self.previous_multiplier
                )

    def get_multiplier(self, current_step):
        raise NotImplementedError


class LinearTempScheduler(AbstractTemperatureScheduler):
    def get_multiplier(self, current_step):
        fraction = 1 - (
            (current_step - self.steps_until_start_temperature_anneal)
            / (
                self.steps_until_finish_temperature_anneal
                - self.steps_until_start_temperature_anneal
            )
        )
        return max(0, fraction)


class CosineTempScheduler(AbstractTemperatureScheduler):
    def get_multiplier(self, current_step):
        if (
            current_step <= self.steps_until_start_temperature_anneal
            or current_step > self.steps_until_finish_temperature_anneal
        ):
            return 1.0
        else:
            fraction = 0.5 * (
                1
                + math.cos(
                    math.pi
                    * (current_step - self.steps_until_start_temperature_anneal)
                    / (
                        self.steps_until_finish_temperature_anneal
                        - self.steps_until_start_temperature_anneal
                    )
                )
            )
            print(f"current step: {current_step}, fraction: {fraction}")
            return max(0.0, fraction)
