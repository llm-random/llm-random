from abc import ABC, abstractmethod

import torch.nn.functional as F
from attr import define

from research.reinitialization.core.pruner import BasePruner


class BaseScheduler(ABC):
    @abstractmethod
    def prune(self):
        ...

    @abstractmethod
    def after_backprop(self):
        ...

    @abstractmethod
    def increment_step(self):
        ...

    def is_time_to_prune(self, step: int) -> bool:
        ...


@define
class DelayedConstScheduler(BaseScheduler):
    def __init__(
        self,
        pruner: BasePruner,
        n_steps_prune: int,
        prob: float,
        delay: int = 0,
        n_steps_retrain: int = None,
    ):
        self.pruner = pruner
        self.n_steps_prune = n_steps_prune
        self.prob = prob
        self.delay = delay
        self.current_step = 0
        self.n_steps_retrain = n_steps_retrain

    def prune(self):
        self.pruner.decrement_immunity()
        if (
            self.current_step % self.n_steps_prune == 0
            and self.current_step >= self.delay
        ):
            self.pruner.prune(self.prob)

    def after_backprop(self):
        self.pruner.after_backprop(self.current_step)

    def increment_step(self):
        self.current_step += 1

    def is_time_to_prune(self, step: int) -> bool:
        return step >= self.delay and step % self.n_steps_prune == 0
