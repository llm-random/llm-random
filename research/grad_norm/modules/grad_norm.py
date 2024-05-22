import torch
import torch.nn as nn


class GradientSTDNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, c: float) -> torch.Tensor:
        ctx._c = c
        ctx._is_unary = x.numel() <= 1  # .std() of size <= 1 is NaN
        return x

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        c = ctx._c
        is_unary = ctx._is_unary
        if is_unary:
            return grad_out, None
        return grad_out / (torch.pow(grad_out.std(), c) + 1e-8), None


class GradientSTDNormLayer(nn.Module):
    def __init__(self, c: float = 1):
        super().__init__()
        if not 0 <= c <= 1:
            raise ValueError("c must be in [0, 1]")
        self.c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientSTDNormFunction.apply(x, self.c)
