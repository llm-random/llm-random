from abc import ABC, abstractmethod

import torch.nn.functional as F
from attr import define

from research.reinitialization.core.pruner import BasePruner


class BaseScheduler(ABC):
    @abstractmethod
    def time_to_prune(self, step: int) -> bool:
        ...


@define
class DelayedConstScheduler(BaseScheduler):
    pruner: BasePruner
    n_steps_prune: int
    prob: float
    delay: int = 0
    n_steps_retrain: int = None

    def time_to_prune(self, step: int) -> bool:
        return step % self.n_steps_prune == 0 and step >= self.delay
