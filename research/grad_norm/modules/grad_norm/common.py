import torch

from lizrd.core.misc import LoggingLayer

LOGGED_METRICS = ["raw_grad", "norm_grad", "activations"]

METRIC_RENAMES = {
    "activations": "activation",
}


class GradLoggingLayer(LoggingLayer):
    def log_heavy(self):
        log_dict = super().log_heavy()

        for metric in LOGGED_METRICS:
            if metric not in self.logging_cache:
                continue

            metric_norm = torch.norm(self.logging_cache[metric], dim=-1)

            metric_norm_mean = torch.mean(metric_norm)
            metric_norm_std = torch.std(metric_norm)

            metric_log_name = METRIC_RENAMES.get(metric, metric)

            log_dict[f"{metric_log_name}_norms/mean"] = metric_norm_mean
            log_dict[f"{metric_log_name}_norms/std"] = metric_norm_std

        return log_dict
