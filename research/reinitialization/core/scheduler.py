from abc import ABC, abstractmethod

from research.reinitialization.core.pruner import BasePruner


class BaseScheduler(ABC):
    @abstractmethod
    def prune(self):
        ...

    @abstractmethod
    def log(self):
        ...

    @abstractmethod
    def increment_step(self):
        ...


class DelayedConstScheduler(BaseScheduler):
    def __init__(
        self,
        pruner: BasePruner,
        n_steps_prune: int,
        prob: float,
        delay: int = 0,
        n_steps_log: int = 5000,
    ):
        self.pruner = pruner
        self.n_steps_prune = n_steps_prune
        self.n_steps_log = n_steps_log
        self.prob = prob
        self.delay = delay
        self.current_step = 0

    def prune(self):
        self.pruner.decrement_immunity()
        if (
            self.current_step % self.n_steps_prune == 0
            and self.current_step >= self.delay
        ):
            self.pruner.prune(self.prob)

    def log(self):
        if self.n_steps_log and (self.current_step % self.n_steps_log == 0):
            self.pruner.log(step)

    def increment_step(self):
        self.current_step += 1
