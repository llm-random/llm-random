from abc import ABC
import math
from typing import Optional

from torch.optim import Optimizer


def get_scheduler(
    args,
    ratios_lr_in_group_order: Optional[list[float]] = None,
    scheduler_fractions_in_group_order: Optional[list[float]] = None,
) -> "AbstractLRScheduler":
    print(args.scheduler)
    if ratios_lr_in_group_order is None:
        ratios_lr_in_group_order = [1.0]
    if scheduler_fractions_in_group_order is None:
        scheduler_fractions_in_group_order = [1.0]
    if args.scheduler == "constant":
        return ConstantScheduler(
            lr_warmup_steps=args.lr_warmup_steps,
            lr=args.learning_rate,
            ratios_lr=ratios_lr_in_group_order,
        )
    elif args.scheduler == "cosine" and "lr_restart_on_chimera" in args and args.lr_restart_on_chimera:
        if args.final_lr_step is None:
            args.final_lr_step = args.n_steps
        restart_time = int(args.chimera_change_after_percent * args.final_lr_step)
        return RestartCosineScheduler(
            restart_time=restart_time,
            lr_warmup_steps=args.lr_warmup_steps,
            lr=args.learning_rate,
            final_lr_step=args.final_lr_step,
            final_lr_fraction=args.final_lr_fraction,
            first_full=args.lr_restart_first_full,
            ratios=ratios_lr_in_group_order,
        )
    elif args.scheduler == "cosine":
        if args.final_lr_step is None:
            args.final_lr_step = args.n_steps
        return CosineScheduler(
            lr_warmup_steps=args.lr_warmup_steps,
            lr=args.learning_rate,
            final_lr_step=args.final_lr_step,
            final_lr_fraction=args.final_lr_fraction,
            ratios_lr=ratios_lr_in_group_order,
            scheduler_fractions=scheduler_fractions_in_group_order,
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
        for param_group, ratio_lr in zip(
            optimizer.param_groups, self.ratios_lr
        ):
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


class RestartCosineScheduler(AbstractLRScheduler):
    def __init__(
        self,
        restart_time: int,
        lr_warmup_steps: int,
        lr: float,
        final_lr_step: int,
        final_lr_fraction: float,
        first_full: bool,
        ratios: list[float],
    ):
        super().__init__(ratios=ratios)
        assert isinstance(restart_time, int)
        assert isinstance(lr_warmup_steps, int)
        assert isinstance(lr, float)
        assert isinstance(final_lr_step, int)
        assert isinstance(final_lr_fraction, float)

        self.restart_time = restart_time

        self.first_scheduler = CosineScheduler(
            lr_warmup_steps=lr_warmup_steps,
            lr=lr,
            final_lr_step=final_lr_step if first_full else restart_time,
            final_lr_fraction=final_lr_fraction,
            ratios=ratios,
        )
        self.second_scheduler = CosineScheduler(
            lr_warmup_steps=lr_warmup_steps,
            lr=lr,
            final_lr_step=final_lr_step - restart_time,
            final_lr_fraction=final_lr_fraction,
            ratios=ratios,
        )

    def get_lr(self, step: int):
        if step < self.restart_time:
            return self.first_scheduler.get_lr(step)
        else:
            return self.second_scheduler.get_lr(step - self.restart_time)


class CosineScheduler(AbstractLRScheduler):
    def __init__(
        self,
        lr_warmup_steps: int,
        lr: float,
        final_lr_step: int,
        final_lr_fraction: float,
        ratios_lr: list[float],
        scheduler_fractions: list[float],
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
        self.scheduler_fractions = scheduler_fractions

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

    # had to overwrite it here, because AbstractLRScheduler doesn't know final_lr_fraction and othr params necessary for relative lr fractions
    def set_lr(self, optimizer: Optimizer, step: int):
        if self.scheduler_fractions is None:
            super.set_lr(optimizer, step)
        else:
            new_lr = self.get_lr(step)
            for param_group, ratio_lr, fraction in zip(
                optimizer.param_groups,
                self.ratios_lr,
                self.scheduler_fractions,
            ):
                if step < self.lr_warmup_steps:
                    relative_lr = new_lr
                elif step < self.final_lr_step:
                    relative_lr = (new_lr - self.lr) * (
                        1 - self.final_lr_fraction * fraction
                    ) / (1 - self.final_lr_fraction) + self.lr
                    relative_lr = relative_lr
                else:
                    relative_lr = new_lr * fraction
                param_group["lr"] = relative_lr * ratio_lr
