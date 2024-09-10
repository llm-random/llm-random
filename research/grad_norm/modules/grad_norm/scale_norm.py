"""Bases on https://arxiv.org/pdf/2106.09475"""

import math
from functools import partial
from typing import Any, Callable, Literal, Union

import torch
from torch.autograd.function import once_differentiable

from research.grad_norm.modules.grad_norm.common import GradLoggingLayer


def scale_norm_grad(grad: torch.Tensor, k: float, eps: float) -> torch.Tensor:
    # assuming shape of grad is (batch_size, max_length, dmodel)
    return k * (grad / (torch.linalg.vector_norm(grad, ord=2, dim=None, keepdim=True) + eps))


class BaseGradientScaleNormFunction(torch.autograd.Function):
    @staticmethod
    def setup_context(ctx: Any, inputs: torch.Tuple[Any], output: Any) -> Any:
        ctx._log_fn = inputs[1]
        ctx._grad_norm_fn = inputs[2]

    @staticmethod
    def forward(
        x: torch.Tensor, log_fn: Callable[[str, Any], None], grad_norm_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out: torch.Tensor):
        ctx._log_fn("raw_grad", grad_out)
        normalized_grad = ctx._grad_norm_fn(grad_out)
        ctx._log_fn("norm_grad", normalized_grad)
        return normalized_grad, None, None


class GradientScaleNormLayer(GradLoggingLayer):
    def __init__(self, k: Union[float, Literal["auto"]] = "auto", eps: float = 1e-6):
        super().__init__()
        self.k = k
        self.prev_shape = None
        self.eps = eps

    def _compute_k(self, x: torch.Tensor) -> float:
        # assuming shape of x is (batch_size, max_length, dmodel)
        return math.sqrt(x.shape[1] * x.shape[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, max_length, dmodel)
        if self.k == "auto":
            self.k = self._compute_k(x)

        if self.prev_shape is None:
            self.prev_shape = x.shape
        if x.shape != self.prev_shape:
            raise ValueError(
                f"GradientScaleNormLayer can only be used with tensors of the same shape. Expected {self.prev_shape}, got {x.shape}"
            )

        self.update_cache_for_logging("activations", x)
        self.update_cache_for_logging("k", self.k)
        # GradientSTDNormFunction should log the pre and post gradients
        return BaseGradientScaleNormFunction.apply(
            x, self.update_cache_for_logging, partial(scale_norm_grad, k=self.k, eps=self.eps)
        )
