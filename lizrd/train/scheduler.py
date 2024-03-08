from abc import ABC
import math

from torch.optim import Optimizer


def get_scheduler(args):
    if args.scheduler == "constant":
        return ConstantScheduler(
            lr_warmup_steps=args.lr_warmup_steps, lr=args.learning_rate
        )
    elif args.scheduler == "cosine":
        if args.final_lr_step is None:
            args.final_lr_step = args.n_steps
        return CosineScheduler(
            lr_warmup_steps=args.lr_warmup_steps,
            lr=args.learning_rate,
            final_lr_step=args.final_lr_step,
            final_lr_fraction=args.final_lr_fraction,
        )
    elif args.scheduler == "finetune_cosine":
        if args.final_lr_step is None:
            args.final_lr_step = args.n_steps
        return FinetuneCosineScheduler(
            lr_warmup_steps=args.lr_warmup_steps,
            lr=args.learning_rate,
            final_lr_step=args.final_lr_step,
            final_lr_fraction=args.final_lr_fraction,
            finetune_steps=args.finetune_steps,
            finetune_lr=args.finetune_lr,
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")


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


class FinetuneCosineScheduler(AbstractLRScheduler):
    """
    Scheduler with two cosine schedules, first for pretraining, second for finetuning.
    """

    def __init__(
        self,
        lr_warmup_steps: int,
        lr: float,
        final_lr_step: int,
        final_lr_fraction: float,
        finetune_steps: int,
        finetune_lr: float,
    ):
        assert isinstance(lr_warmup_steps, int)
        assert isinstance(lr, float)
        assert isinstance(final_lr_step, int)
        assert isinstance(final_lr_fraction, float)
        assert isinstance(finetune_steps, int)
        assert isinstance(finetune_lr, float)

        self.lr_warmup_steps = lr_warmup_steps
        self.lr = lr
        self.final_lr_step = final_lr_step
        self.final_lr_fraction = final_lr_fraction
        self.finetune_steps = finetune_steps
        self.finetune_lr = finetune_lr

        self.pretrain_scheduler = CosineScheduler(
            lr_warmup_steps=lr_warmup_steps,
            lr=lr,
            final_lr_step=final_lr_step,
            final_lr_fraction=final_lr_fraction,
        )
        self.finetune_scheduler = CosineScheduler(
            lr_warmup_steps=0,
            lr=finetune_lr,
            final_lr_step=final_lr_step + finetune_steps,
            final_lr_fraction=final_lr_fraction,
        )

    def get_lr(self, step: int):
        if step < self.final_lr_step:
            return self.pretrain_scheduler.get_lr(step)
        else:
            return self.finetune_scheduler.get_lr(step - self.final_lr_step)
