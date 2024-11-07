from abc import ABC
import math
from typing import Optional

from torch.optim import Optimizer


def get_scheduler(
    args,
    relative_lrs_in_group_order: Optional[list[float]] = None,
    final_lr_fractions_in_group_order: Optional[list[float]] = None,
) -> "AbstractLRScheduler":
    if relative_lrs_in_group_order is None:
        relative_lrs_in_group_order = [1.0]
    if final_lr_fractions_in_group_order is None:
        final_lr_fractions_in_group_order = [1.0]
    if args.lr_warmup_percent is not None:
        assert (
            not args.lr_warmup_steps
        ), "You cannot set both lr_warmap_percent and lr_warmap_steps"
        args.lr_warmup_steps = math.ceil(args.n_steps * args.lr_warmup_percent)
    if args.final_lr_step == -1 or args.final_lr_step is None:
        args.final_lr_step = (
            args.n_steps
        )  # dev - 1 seems proper, may cause problems, idk

    if args.scheduler == "constant":
        return ConstantScheduler(
            lr_warmup_steps=args.lr_warmup_steps,
            lr=args.learning_rate,
            ratios_lr=relative_lrs_in_group_order,
        )
    elif args.scheduler == "cosine":
        return CosineScheduler(
            lr_warmup_steps=args.lr_warmup_steps,
            lr=args.learning_rate,
            final_lr_step=args.final_lr_step,
            final_lr_fraction=args.final_lr_fraction,
            ratios_lr=relative_lrs_in_group_order,
            final_lr_fractions=final_lr_fractions_in_group_order,
        )
    elif args.scheduler == "trapezoidal":
        return TrapezoidalScheduler(
            lr_warmup_steps=args.lr_warmup_steps,
            lr_decay_steps=math.ceil(args.n_steps * args.lr_trapezoidal_decay_fraction),
            n_steps=args.n_steps,
            lr=args.learning_rate,
            ratios_lr=relative_lrs_in_group_order,
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")


class AbstractLRScheduler(ABC):
    def __init__(self, ratios_lr: list[float]):
        self.ratios_lr = ratios_lr

    def get_lr(self, step):
        raise NotImplementedError

    def set_lr(self, optimizer: Optimizer, step: int):
        new_lr = self.get_lr(step)
        for param_group, ratio_lr in zip(optimizer.param_groups, self.ratios_lr):
            param_group["lr"] = new_lr * ratio_lr


class ConstantScheduler(AbstractLRScheduler):
    def __init__(self, lr_warmup_steps: int, lr: float, ratios_lr: list[float]):
        super().__init__(ratios_lr=ratios_lr)
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
        ratios_lr: list[float],
        final_lr_fractions: list[float],
    ):
        super().__init__(ratios_lr=ratios_lr)
        assert isinstance(lr_warmup_steps, int)
        assert isinstance(lr, float)
        assert isinstance(final_lr_step, int)
        assert isinstance(final_lr_fraction, float)

        self.lr_warmup_steps = lr_warmup_steps
        self.lr = lr
        self.final_lr_step = final_lr_step
        self.final_lr_fraction = final_lr_fraction
        self.final_lr_fractions = final_lr_fractions

    def get_lr(self, step: int, start_lr=None, end_lr=None):
        start_lr = start_lr if start_lr is not None else self.lr
        end_lr = end_lr if end_lr is not None else self.lr * self.final_lr_fraction

        if step < self.lr_warmup_steps:
            return start_lr * (step + 1) / self.lr_warmup_steps
        # cosine schedule that ends at final_lr_fraction * lr, then constant
        elif step < self.final_lr_step:
            return end_lr + 0.5 * (1 - end_lr / start_lr) * start_lr * (
                1
                + math.cos(
                    math.pi
                    * (step - self.lr_warmup_steps)
                    / (self.final_lr_step - self.lr_warmup_steps)
                )
            )
        else:
            return end_lr

    # had to overwrite it here, because AbstractLRScheduler doesn't know final_lr_fraction and othr params necessary for relative lr fractions
    def set_lr(self, optimizer: Optimizer, step: int):
        if self.final_lr_fractions is None:
            super.set_lr(optimizer, step)
        else:
            for param_group, ratio_lr, fraction in zip(
                optimizer.param_groups,
                self.ratios_lr,
                self.final_lr_fractions,
            ):
                start_lr = self.lr * ratio_lr
                end_lr = self.lr * self.final_lr_fraction * fraction
                param_group["lr"] = self.get_lr(step, start_lr=start_lr, end_lr=end_lr)


class TrapezoidalScheduler(AbstractLRScheduler):
    def __init__(
        self,
        lr_warmup_steps: int,
        lr_decay_steps: int,
        n_steps: int,
        lr: float,
        ratios_lr: list[float],
    ):
        super().__init__(ratios=ratios_lr)
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.final_lr_step = n_steps
        self.lr = lr

    def get_lr(self, step: int):
        if step < self.lr_warmup_steps:
            return step / self.lr_warmup_steps * self.lr
        elif self.lr_warmup_steps <= step and step < (
            self.final_lr_step - self.lr_decay_steps
        ):
            return self.lr
        else:
            return max(
                (self.final_lr_step - (step + 1)) / self.lr_decay_steps * self.lr, 0
            )
