from abc import ABC, abstractmethod
from typing import Optional

from research.reinitialization.core.pruner import BasePruner
from attr import define


class BaseScheduler(ABC):
    @abstractmethod
    def step(self):
        ...


@define
class DelayedConstScheduler(BaseScheduler):
    pruner: BasePruner
    n_steps_prune: int
    prob: float
    delay: int = 0
    n_steps_log_recycle_hist: Optional[int] = None
    n_steps_log_magnitude: Optional[int] = None
    n_steps_hist_all: Optional[int] = None
    current_step = 0

    def step(self):
        if (
            self.current_step % self.n_steps_prune == 0
            and self.current_step >= self.delay
        ):
            self.pruner.prune(self.prob)

        if (
            self.n_steps_log_recycle_hist is not None
            and self.current_step % self.n_steps_log_recycle_hist == 0
        ):
            self.pruner.log_recycle_magnitude(self.current_step)

        if (
            self.n_steps_log_magnitude is not None
            and self.current_step % self.n_steps_log_magnitude == 0
        ):
            self.pruner.log_magnitude(self.current_step)

        if (
            self.n_steps_hist_all is not None
            and self.current_step % self.n_steps_hist_all == 0
        ):
            self.pruner.log_hist_all_weights(self.current_step)

        self.current_step += 1
