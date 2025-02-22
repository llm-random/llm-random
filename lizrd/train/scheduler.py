from abc import ABC
import math

from torch.optim import Optimizer


def get_scheduler(args) -> "AbstractLRScheduler":
    if hasattr(args, "lr_warmup_percent") and args.lr_warmup_percent is not None:
        assert (
            not args.lr_warmup_steps
        ), "You cannot set both lr_warmap_percent and lr_warmap_steps"
        args.lr_warmup_steps = math.ceil(args.n_steps * args.lr_warmup_percent)
    if args.final_lr_step == -1 or args.final_lr_step is None:
        args.final_lr_step = args.n_steps

    if args.scheduler == "constant":
        return ConstantScheduler(
            lr_warmup_steps=args.lr_warmup_steps,
            lr=args.learning_rate,
        )
    elif args.scheduler == "cosine":
        return CosineScheduler(
            lr_warmup_steps=args.lr_warmup_steps,
            lr=args.learning_rate,
            final_lr_step=args.final_lr_step,
            final_lr_fraction=args.final_lr_fraction,
        )
    elif args.scheduler == "trapezoidal":
        return TrapezoidalScheduler(
            lr_warmup_steps=args.lr_warmup_steps,
            lr_decay_steps=args.lr_trapezoidal_decay_steps,
            n_steps=args.n_steps,
            lr=args.learning_rate,
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")


class AbstractLRScheduler(ABC):
    def __init__(self):
        pass

    def get_lr(self, step):
        raise NotImplementedError

    # TODO optimizer get lr (like nanoGPT)
    def set_lr(self, optimizer: Optimizer, step: int):
        new_lr = self.get_lr(step)
        for param_group in optimizer.param_groups:
            ratio = param_group.get("lr_ratio", 1.0)
            param_group["lr"] = new_lr * ratio


class ConstantScheduler(AbstractLRScheduler):
    def __init__(self, lr_warmup_steps: int, lr: float):
        super().__init__()
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
        super().__init__()
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


class TrapezoidalScheduler(AbstractLRScheduler):
    def __init__(
        self,
        lr_warmup_steps: int,
        lr_decay_steps: int,
        n_steps: int,
        lr: float,
    ):
        super().__init__()
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
