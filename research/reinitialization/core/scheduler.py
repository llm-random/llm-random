from abc import ABC, abstractmethod

import torch.nn as nn
import torch
import torch.nn.functional as F
from attr import define
from torch.utils.tensorboard import SummaryWriter

from research.reinitialization.core.pruner import BasePruner, RetrainPruner
from lizrd.datasets import wikibookdata


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
    n_steps_retrain: int = 1000

    def time_to_prune(self, step: int) -> bool:
        return step % self.n_steps_prune == 0 and step >= self.delay
