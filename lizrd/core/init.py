import torch
from torch.nn.init import trunc_normal_


def init_uniform(shape, fan_in, gain=1.0, dtype=torch.float32):
    range_ = gain * (3 / fan_in) ** 0.5
    return torch.zeros(shape, dtype=dtype).uniform_(-range_, range_)


def init_truncated_normal(shape, fan_in, scale=0.1, dtype=torch.float32):
    std = (scale / fan_in) ** 0.5
    low = -2 * scale
    high = 2 * scale
    t = torch.zeros(shape, dtype=dtype)
    return trunc_normal_(t, mean=0.0, std=std, a=low, b=high)


def get_init_weight(shape, fan_in, init_type="uniform", scale=1.0, dtype=torch.float32):
    if init_type == "uniform":
        return init_uniform(shape=shape, fan_in=fan_in, gain=scale, dtype=dtype)
    elif init_type == "truncated_normal":
        return init_truncated_normal(
            shape=shape, fan_in=fan_in, scale=scale, dtype=dtype
        )
    else:
        raise ValueError(f"Unknown init_type: {init_type}")


def get_init_bias(shape, fan_in=None, fan_out=None, dtype=torch.float32):
    if fan_in is not None:
        raise ValueError("fan_in unsupported")
    if fan_out is not None:
        raise ValueError("fan_out unsupported")
    return torch.zeros(shape, dtype=dtype)
