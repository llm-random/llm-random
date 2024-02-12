# Copyright (c) 2023, Tri Dao, Albert Gu.

import math

import torch
import torch.nn as nn

from einops import rearrange, repeat

from token_choice import TokenChoiceRouter
from research.conditional.utils.layer_manager import measure_time, LoggingLayer

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None


class MambaTokenChoice(LoggingLayer):
    def __init__(
        self,
        d_model,
        input_module: nn.Module,
        gate_module: nn.Module,
        output_module: nn.Module,
        router: nn.Module = None,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = False
        self.layer_idx = layer_idx

        self.input_module = input_module
        self.gate_module = gate_module
        self.output_module = output_module

        self.router = router

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        # We do matmul and transpose BLH -> HBL at the same time
        x = self.input_module(hidden_states)  # (B L D) -> (B L D)
        z = self.gate_module(hidden_states)  # (B L D) -> (B L D)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat

        # Compute short convolution
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            assert self.activation in ["silu", "swish"]
            x = causal_conv1d_fn(
                x,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        assert self.activation in ["silu", "swish"]
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )
        y = rearrange(y, "b d l -> b l d")
        out = self.output_module(y)
        assert out.shape == (batch, seqlen, dim)
        return out


class MambaRouter(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        n_experts: int,
        capacity_factor: float,
        load_balancing_loss_weight: float,
        routing_groups: list[list[str]],
        init_type: str,
        init_scale: float,
    ):
        """
        Args:
            dmodel: dimension of the input
            n_experts: number of experts
            expert_size: size of each expert
            capacity_factor: scalar that determines how many tokens can be assigned to each expert
            load_balancing_loss_weight: weight of the auxillary loss
            expert_logic: expert logic layer, takes input of shape (n_experts, capacity, dmodel) and returns output of shape (n_experts, capacity, dmodel)
        """
        super().__init__()

        self.dmodel = dmodel
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.load_balancing_loss_weight = load_balancing_loss_weight
        self.routing_groups = routing_groups

        self.routers = nn.ModuleList(
            [
                self._make_router(init_type=init_type, init_scale=init_scale)
                for _ in routing_groups
            ]
        )

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        (
            experts_input,
            top_tokens_per_expert_indices,  # [c x n_experts]
            masked_expert_gate,
        ) = self.router(x)

        x = x.flatten(start_dim=0, end_dim=1)

        experts_output = self.expert_inner_function(experts_input)

        experts_output = experts_output.to(x.dtype)
        output = torch.zeros_like(x)

        with measure_time(self, "assign_tokens_to_output"):
            output.index_add_(
                dim=0,
                index=top_tokens_per_expert_indices.T.flatten(),
                source=experts_output.reshape(
                    self.n_experts * experts_output.shape[1], self.dmodel
                ),
            )
            output *= masked_expert_gate.sum(dim=1, keepdim=True)

        output = output.reshape((batch_size, seq_len, self.dmodel))

        return output

    def _make_router(self, init_type, init_scale):
        return TokenChoiceRouter(
            dmodel=self.dmodel,
            n_experts=self.n_experts,
            capacity_factor=self.capacity_factor,
            load_balancing_loss_weight=self.load_balancing_loss_weight,
            init_type=init_type,
            init_scale=init_scale,
            routing_top_k=1,
            use_einsum=False,
            vectorize=True,
        )
