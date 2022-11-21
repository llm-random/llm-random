from abc import ABC, abstractmethod

from research.reinitialization.core.pruner import BasePruner
from attr import define


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
        if (
            self.current_step % self.n_steps_prune == 0
            and self.current_step >= self.delay
        ):
            self.pruner.prune(self.prob)
        self.current_step += 1


@define
class MagnitudeStatScheduler(BaseScheduler):
    pruner: BasePruner
    n_steps_log: int
    delay: int = 0
    n_steps_log_recycl_hist: int = 5000
    n_steps_log_magnitude: int = 5000
    n_steps_hist_all: int = 5000

    def step(self):
        self.pruner.log_recently_pruned_magnitude(self.current_step)

        if (
            self.current_step % self.n_steps_log == 0
            and self.current_step >= self.delay
        ):
            self.pruner.log_magnitude(self.current_step)

        if self.current_step % self.n_steps_log_recycl_hist == 0:
            self.pruner.log_recycl_magnitude(self.current_step)

        if self.current_step % self.n_steps_log_magnitude == 0:
            self.pruner.log_magnitude(self.current_step)

        if self.current_step % self.n_steps_hist_all == 0:
            self.pruner.log_hist_all_weights(self.current_step)
        self.current_step += 1
