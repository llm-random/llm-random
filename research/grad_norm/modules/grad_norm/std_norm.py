from functools import partial
from typing import Any, Callable

import torch
from torch.autograd.function import once_differentiable

from research.grad_norm.modules.grad_norm.common import GradLoggingLayer


def std_grad_norm_v1(grad: torch.Tensor, c: float, eps: float) -> torch.Tensor:
    return grad / (torch.pow(grad.std(), c) + eps)


def std_grad_norm_v2(grad: torch.Tensor, c: float, eps: float) -> torch.Tensor:
    # assuming grad is (batch_size, max_length, dmodel)
    dmodel_std = grad.std(axis=-1, correction=0, keepdim=True)  # (batch_size, max_length)
    return grad / (torch.pow(dmodel_std, c) + eps)


def std_grad_norm_v3(grad: torch.Tensor, c: float, eps: float) -> torch.Tensor:
    dmodel_std = grad.std(axis=-1, correction=0, keepdim=True)  # (batch_size, max_length)
    global_std = grad.std(correction=0)  # (,)
    return grad * global_std / (torch.pow(dmodel_std, c) + eps)


class BaseGradientSTDNormFunction(torch.autograd.Function):
    @staticmethod
    def setup_context(ctx: Any, inputs: torch.Tuple[Any], output: Any) -> Any:
        ctx._is_unary = inputs[0].numel() <= 1
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
        is_unary = ctx._is_unary
        if is_unary:
            return grad_out, None, None
        ctx._log_fn("raw_grad", grad_out)
        normalized_grad = ctx._grad_norm_fn(grad_out)
        ctx._log_fn("norm_grad", normalized_grad)
        return normalized_grad, None, None


class GradientSTDNormLayerV1(GradLoggingLayer):
    def __init__(self, c: float = 1, eps: float = 1e-8):
        super().__init__()
        self.c = c
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, max_length, dmodel)
        self.update_cache_for_logging("activations", x)
        # GradientSTDNormFunction should log the pre and post gradients
        return BaseGradientSTDNormFunction.apply(
            x, self.update_cache_for_logging, partial(std_grad_norm_v1, c=self.c, eps=self.eps)
        )


class GradientSTDNormLayerV2(GradientSTDNormLayerV1):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, max_length, dmodel)
        self.update_cache_for_logging("activations", x)
        # GradientSTDNormFunction should log the pre and post gradients
        return BaseGradientSTDNormFunction.apply(
            x, self.update_cache_for_logging, partial(std_grad_norm_v2, c=self.c, eps=self.eps)
        )


class GradientSTDNormLayerV3(GradientSTDNormLayerV1):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, max_length, dmodel)
        self.update_cache_for_logging("activations", x)
        # GradientSTDNormFunction should log the pre and post gradients
        return BaseGradientSTDNormFunction.apply(
            x, self.update_cache_for_logging, partial(std_grad_norm_v3, c=self.c, eps=self.eps)
        )
