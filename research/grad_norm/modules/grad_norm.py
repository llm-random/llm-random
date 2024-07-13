from typing import Any, Callable

import torch

from lizrd.core.misc import LoggingLayer


class GradientSTDNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, c: float, log_fn: Callable[[str, Any], None]) -> torch.Tensor:
        ctx._c = c
        ctx._log_fn = log_fn
        ctx._is_unary = x.numel() <= 1  # .std() of size <= 1 is NaN
        return x

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        c = ctx._c
        is_unary = ctx._is_unary
        if is_unary:
            return grad_out, None, None
        ctx._log_fn("raw_grad", grad_out)
        normalized_grad = grad_out / (torch.pow(grad_out.std(), c) + 1e-8)
        ctx._log_fn("norm_grad", normalized_grad)
        return normalized_grad, None, None


class GradientSTDNormLayer(LoggingLayer):
    def __init__(self, c: float = 1):
        super().__init__()
        if not 0 <= c <= 1:
            raise ValueError("c must be in [0, 1]")
        self.c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, max_length, dmodel)
        self.update_cache_for_logging("activations", x)
        # GradientSTDNormFunction should log the pre and post gradients
        return GradientSTDNormFunction.apply(x, self.c, self.update_cache_for_logging)

    def log_heavy(self):
        log_dict = super().log_heavy()

        raw_grad_norms = torch.norm(self.logging_cache["raw_grad"], dim=-1)  # (batch_size, max_length)
        norm_grad_norms = torch.norm(self.logging_cache["norm_grad"], dim=-1)  # (batch_size, max_length)
        activation_norms = torch.norm(self.logging_cache["activations"], dim=-1)

        raw_grad_norms_mean = torch.mean(raw_grad_norms)  # avegare over all tokens and batches
        raw_grad_norms_std = torch.std(raw_grad_norms)
        norm_grad_norms_mean = torch.mean(norm_grad_norms)
        norm_grad_norms_std = torch.std(norm_grad_norms)
        activation_norms_mean = torch.mean(activation_norms)
        activation_norms_std = torch.std(activation_norms)

        log_dict["raw_grad_norms_mean"] = raw_grad_norms_mean
        log_dict["raw_grad_norms_std"] = raw_grad_norms_std
        log_dict["norm_grad_norms_mean"] = norm_grad_norms_mean
        log_dict["norm_grad_norms_std"] = norm_grad_norms_std
        log_dict["activation_norms_mean"] = activation_norms_mean
        log_dict["activation_norms_std"] = activation_norms_std

        return log_dict
