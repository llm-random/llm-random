from typing import Any, Callable

import torch

from lizrd.core.misc import LoggingLayer


class CaptureGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, log_fn: Callable[[str, Any], None]):
        ctx._log_fn = log_fn
        return x

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        ctx._log_fn("grad", grad_out)
        return grad_out, None


class GradCaptureLayer(LoggingLayer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.update_cache_for_logging("activations", x)
        return CaptureGradFunction.apply(x, self.update_cache_for_logging)

    def log_heavy(self):
        log_dict = super().log_heavy()

        grad_norms = torch.norm(self.logging_cache["grad"], dim=-1)
        activation_norms = torch.norm(self.logging_cache["activations"], dim=-1)

        grad_norms_mean = torch.mean(grad_norms)
        grad_norms_std = torch.std(grad_norms)
        activation_norms_mean = torch.mean(activation_norms)
        activation_norms_std = torch.std(activation_norms)

        log_dict["grad_norms/mean"] = grad_norms_mean
        log_dict["grad_norms/std"] = grad_norms_std
        log_dict["activation_norms/mean"] = activation_norms_mean
        log_dict["activation_norms/std"] = activation_norms_std

        return log_dict
