"Inspired by https://arxiv.org/pdf/1707.04822"


from functools import partial
from typing import Any, Callable, Optional, Tuple

import torch
from torch.autograd.function import once_differentiable

from research.grad_norm.modules.grad_norm.common import GradLoggingLayer


def _norm(x: torch.Tensor, dim: Optional[Tuple[int, ...]]) -> torch.Tensor:
    "assuming x is (batch_size, max_length, dmodel)"
    return torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=True)


def activation_norm_grad(
    grad: torch.Tensor, activation_norm: torch.Tensor, norm_dims: Tuple[int, ...], eps: float
) -> torch.Tensor:
    # assuming shape of grad is (batch_size, max_length, dmodel)
    ratio = activation_norm / (_norm(grad, norm_dims) + eps)
    return grad * ratio


class BaseGradientActivationNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        log_fn: Callable[[str, Any], None],
        grad_norm_fn: Callable[[torch.Tensor], torch.Tensor],
        norm_dims: Tuple[int, ...],
    ) -> torch.Tensor:
        ctx.save_for_backward(_norm(x, norm_dims))
        ctx._log_fn = log_fn
        ctx._grad_norm_fn = grad_norm_fn
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out: torch.Tensor):
        (activation_norm,) = ctx.saved_tensors
        ctx._log_fn("raw_grad", grad_out)
        normalized_grad = ctx._grad_norm_fn(grad_out, activation_norm)
        ctx._log_fn("norm_grad", normalized_grad)
        return normalized_grad, None, None, None


class GradientActivationNormLayer(GradLoggingLayer):
    def __init__(self, norm_dims: Tuple[int, ...], eps: float = 1e-6):
        super().__init__()
        self.norm_dims = norm_dims
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, max_length, dmodel)
        self.update_cache_for_logging("activations", x)
        # GradientSTDNormFunction should log the pre and post gradients
        return BaseGradientActivationNormFunction.apply(
            x,
            self.update_cache_for_logging,
            partial(activation_norm_grad, norm_dims=self.norm_dims, eps=self.eps),
            self.norm_dims,
        )
