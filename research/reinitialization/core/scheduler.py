from abc import ABC, abstractmethod

from research.reinitialization.core.pruner import BasePruner


class BaseScheduler(ABC):
    @abstractmethod
    def step(self):
        ...


class DelayedConstScheduler(BaseScheduler):
    def __init__(
        self, pruner: BasePruner, n_steps_prune: int, prob: float, delay: int = 0
    ):
        self.pruner = pruner
        self.n_steps_prune = n_steps_prune
        self.prob = prob
        self.delay = delay
        self.current_step = 0

    def step(self):
        self.pruner.decrement_immunity()
        if (
            self.current_step % self.n_steps_prune == 0
            and self.current_step >= self.delay
        ):
            self.pruner.prune(self.prob)
        self.current_step += 1
