from abc import ABC, abstractmethod

from attr import define


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
class DelayedConstScheduler:
    n_steps_prune: int
    prob: float
    delay: int = 0
    n_steps_retrain: int = None

    def is_time_to_prune(self, step: int) -> bool:
        return step >= self.delay and step % self.n_steps_prune == 0
