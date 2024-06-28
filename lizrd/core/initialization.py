import torch
from torch.nn.init import trunc_normal_
from functools import partial
import torch.nn as nn
from typing import Literal, TypeAlias

ValidInits = ["kaiming_uniform", "truncated_normal", "truncated_normal_fixed"]

ValidInitType: TypeAlias = Literal[
    "kaiming_uniform", "truncated_normal", "truncated_normal_fixed"
]


def get_init_weight(
    shape, fan_in, init_type: ValidInitType, scale, dtype=torch.float32
):
    if init_type == "kaiming_uniform":
        return init_kaiming_uniform(
            shape=shape, fan_in=fan_in, scale=scale, dtype=dtype
        )
    elif init_type == "truncated_normal":
        return init_truncated_normal(
            shape=shape, fan_in=fan_in, scale=scale, dtype=dtype
        )
    elif init_type == "truncated_normal_fixed":
        return init_truncated_normal_fixed(
            shape=shape, fan_in=fan_in, scale=scale, dtype=dtype
        )
    else:
        raise ValueError(f"Unknown init_type: {init_type}")


def get_init_fun(init_type: ValidInitType, init_scale):
    get_init = partial(get_init_weight, init_type=init_type, scale=init_scale)
    return lambda *a, **aa: nn.Parameter(get_init(*a, **aa)).requires_grad_(True)


def init_kaiming_uniform(shape, fan_in, scale, dtype=torch.float32):
    range_ = scale * (3 / fan_in) ** 0.5
    return torch.zeros(shape, dtype=dtype).uniform_(-range_, range_)


def init_truncated_normal(shape, fan_in, scale, dtype=torch.float32):
    std = (scale / fan_in) ** 0.5
    low = -2 * scale
    high = 2 * scale
    t = torch.zeros(shape, dtype=dtype)
    return trunc_normal_(t, mean=0.0, std=std, a=low, b=high)


def init_truncated_normal_fixed(shape, fan_in, scale, dtype=torch.float32):
    std = scale * (1 / fan_in) ** 0.5
    low = -2 * std
    high = 2 * std
    t = torch.zeros(shape, dtype=dtype)
    return trunc_normal_(t, mean=0.0, std=std, a=low, b=high)


def get_init_bias(shape, fan_in=None, fan_out=None, dtype=torch.float32):
    if fan_in is not None:
        raise ValueError("fan_in unsupported")
    if fan_out is not None:
        raise ValueError("fan_out unsupported")
    return torch.zeros(shape, dtype=dtype)
