import torch
import torch.nn as nn
from research.muP_MoE.utils.layer_manager import LoggingLayer
from research.muP_MoE.moe_layers.expert_choice import ExpertChoiceFF
from research.muP_MoE.moe_layers.token_choice import TokenChoiceFF
from research.muP_MoE.moe_layers.expert_types import ExpertLinear
from functools import partial
from lizrd.core.misc import resolve_activation_name


def get_moe(routing_type, topk_fraction, routing_top_k, *args, **kwargs):
    if routing_type == "expert_choice":
        return ExpertChoiceFF(topk_fraction=topk_fraction, *args, **kwargs)
    elif routing_type == "token_choice":
        return TokenChoiceFF(routing_top_k=routing_top_k, *args, **kwargs)
    else:
        raise ValueError(f"Unknown routing type: {routing_type}")


def with_nonlinearity(layer, activation_type, act_first=False):
    activation = resolve_activation_name(activation_type)
    return (
        nn.Sequential(activation, layer)
        if act_first
        else nn.Sequential(layer, activation)
    )


class DoubleChoiceInner(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        n_experts: int,
        expert_size: int,
        activation_type: str,
        linear_first: bool = False,
        relu_with_first: bool = False,
        init_topk: bool = False,
        routing_top_k: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.doutput = dmodel

        fan_in = int(routing_top_k if init_topk else n_experts) * expert_size
        linear_1 = ExpertLinear(dmodel, n_experts, expert_size, **kwargs)
        linear_2 = ExpertLinear(expert_size, n_experts, dmodel, fan_in=fan_in, **kwargs)

        # activation placement
        act = partial(with_nonlinearity, activation_type=activation_type)
        if relu_with_first:
            linear_1 = act(linear_1)
        else:
            linear_2 = act(linear_2, act_first=True)

        # second routing should not "expand" number of tokens inside
        kwargs.update(topk_fraction=1 / n_experts, routing_top_k=1)
        route = partial(get_moe, n_experts=n_experts, *args, **kwargs)
        if linear_first:
            self.linear_1 = linear_1
            self.linear_2 = route(dmodel=expert_size, expert_inner_function=linear_2)
        else:
            self.linear_1 = route(dmodel=dmodel, expert_inner_function=linear_1)
            self.linear_2 = linear_2

    def forward(self, x: torch.Tensor):
        x = self.linear_1(x)
        x = self.linear_2(x)
        return x


class DoubleChoiceFF(LoggingLayer):
    def __init__(self, *args, **kwargs):
        """
        These kwargs are meant to initialize Token Choice or Expert Choice, dependently on the routing type.
        """
        super().__init__()
        inner = DoubleChoiceInner(*args, **kwargs)
        self.router = get_moe(expert_inner_function=inner, *args, **kwargs)

    def forward(self, x: torch.Tensor):
        return self.router(x)
