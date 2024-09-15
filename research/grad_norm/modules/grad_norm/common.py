from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch

from lizrd.core.misc import LoggingLayer


@dataclass
class LoggedMetric:
    name: str
    statistics: List[Tuple[str, Callable[[Any], float]]]
    rename: Optional[str]
    take_norm: bool


LOGGED_METRICS = [
    LoggedMetric("raw_grad", [("mean", torch.mean), ("std", torch.std)], None, True),
    LoggedMetric("norm_grad", [("mean", torch.mean), ("std", torch.std)], None, True),
    LoggedMetric("activations", [("mean", torch.mean), ("std", torch.std)], "activation", True),
    LoggedMetric("k", [("value", lambda x: x)], None, False),
]


class GradLoggingLayer(LoggingLayer):
    def log_heavy(self):
        log_dict = super().log_heavy()

        for metric in LOGGED_METRICS:
            if metric.name not in self.logging_cache:
                continue

            value = self.logging_cache[metric.name]
            if metric.take_norm:
                value = torch.norm(value, dim=-1)

            for stat_name, stat_fn in metric.statistics:
                key = f"{metric.rename or metric.name}{'_norms' if metric.take_norm else ''}/{stat_name}"
                log_dict[key] = stat_fn(value)

        return log_dict
