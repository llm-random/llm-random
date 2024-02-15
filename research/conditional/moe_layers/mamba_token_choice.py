# Copyright (c) 2023, Tri Dao, Albert Gu.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from lizrd.support.logging import make_histogram

from fancy_einsum import einsum
from lizrd.train import checkpointing

from einops import rearrange, repeat

from lizrd.core.initialization import get_init_weight

from research.conditional.moe_layers.token_choice import (
    calculate_load_balancing_loss,
)
from research.conditional.utils.layer_manager import measure_time, LoggingLayer

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None


MAMBA_MODULES_ORDER = ["input", "gate", "output"]


class TokenChoiceSeparateRouter(LoggingLayer):
    def __init__(
        self,
        dinput: int,
        n_experts: int,
        capacity_factor: float,
        load_balancing_loss_weight: float,
        init_type: str,
        init_scale: float,
        routing_top_k: int = 1,
        use_einsum: bool = False,
        vectorize: bool = True,
    ):
        super().__init__()

        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.load_balancing_loss_weight = load_balancing_loss_weight
        self.use_einsum = use_einsum
        self.dinput = dinput
        self.routing_top_k = routing_top_k
        self._checkpointed_expert_index = None
        self._checkpointed_top_tokens_per_expert_indices = None
        self.vectorize = vectorize

        if vectorize and routing_top_k != 1:
            raise ValueError("vectorize and routing_top_k != 1 are incompatible")

        self.gate = nn.Parameter(
            get_init_weight(
                shape=(dinput, n_experts),
                fan_in=dinput,
                init_type=init_type,
                scale=init_scale,
            )
        )

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        n_tokens = batch_size * seq_len
        self.update_cache_for_logging("n_tokens", torch.Tensor([n_tokens]))

        x = x.reshape((n_tokens, self.dinput))

        with measure_time(self, "expert_embedding"):
            if self.use_einsum:
                gate_out = einsum(
                    "n_tokens dmodel, dmodel n_experts -> n_tokens n_experts",
                    x,
                    self.gate,
                )
            else:
                gate_out = torch.matmul(x, self.gate)

        assert gate_out.shape == (n_tokens, self.n_experts)
        capacity = int(
            self.capacity_factor * n_tokens * self.routing_top_k / self.n_experts
        )
        if self.vectorize:
            capacity = min(capacity, n_tokens)

        # perform softmax over experts for each token
        with measure_time(self, "softmax"):
            gate_out = torch.softmax(gate_out, dim=1)

        self.update_cache_for_logging("gate_softmax_all_values", gate_out)

        with measure_time(self, "choose_expert"):
            expert_gate, expert_index = self.choose_expert(gate_out)

        with measure_time(self, "create_expert_mask"):
            expanded_expert_mask = F.one_hot(expert_index, num_classes=self.n_experts)
            assert expanded_expert_mask.shape == (
                n_tokens,
                self.routing_top_k,
                self.n_experts,
            )
            expert_mask = expanded_expert_mask.sum(dim=1)
            assert expert_mask.shape == (n_tokens, self.n_experts)

        if self.vectorize:
            with measure_time(self, "experts_lists"):
                (
                    top_tokens_per_expert_values,
                    top_tokens_per_expert_indices,
                ) = expert_mask.topk(k=capacity, dim=0)
            with measure_time(self, "create_truncated_mask"):
                truncated_expert_mask = torch.zeros_like(expert_mask)
                truncated_expert_mask.scatter_(
                    dim=0,
                    index=top_tokens_per_expert_indices,
                    src=top_tokens_per_expert_values,
                )
        else:
            with measure_time(self, "experts_lists"):
                indices_of_tokens_for_expert = [
                    single_expert_mask.nonzero(as_tuple=True)[0][: (capacity - 1)]
                    for single_expert_mask in expert_mask.transpose(0, 1)
                ]
            with measure_time(self, "create_truncated_mask"):
                truncated_expert_mask = torch.zeros_like(expert_mask)
                for i, indices in enumerate(indices_of_tokens_for_expert):
                    truncated_expert_mask[indices, i] = 1

        n_selected_tokens = truncated_expert_mask.sum().item()

        self.update_cache_for_logging(
            "dropped_tokens_ratio",
            ((n_tokens * self.routing_top_k) - n_selected_tokens)
            / (n_tokens * self.routing_top_k),
        )

        with measure_time(self, "calculate aux loss"):
            tokens_per_expert = expert_mask.sum(dim=0, dtype=gate_out.dtype)
            load_balancing_loss = calculate_load_balancing_loss(
                self.load_balancing_loss_weight,
                gate_out,
                tokens_per_expert,
                use_einsum=self.use_einsum,
            )

        if "load_balancing_losses" not in self.forward_pass_cache:
            self.forward_pass_cache["load_balancing_losses"] = [load_balancing_loss]
        else:
            self.forward_pass_cache["load_balancing_losses"].append(load_balancing_loss)

        self.update_cache_for_logging("gate_softmax_values", expert_gate)
        self.update_cache_for_logging("max_indices", expert_index)
        self.update_cache_for_logging("tokens_per_expert", tokens_per_expert)
        self.update_cache_for_logging("load_balancing_loss", load_balancing_loss)

        masked_expert_gate = gate_out * truncated_expert_mask

        if self.vectorize:
            return (
                top_tokens_per_expert_values,
                top_tokens_per_expert_indices,
                masked_expert_gate,
                capacity,
            )
        else:
            return indices_of_tokens_for_expert, masked_expert_gate, capacity

    def choose_tokens(self, expert_mask, capacity) -> tuple[torch.Tensor, torch.Tensor]:
        checkpointing_enabled = (
            checkpointing.is_in_first_forward() or checkpointing.is_in_second_forward()
        )
        if checkpointing_enabled:
            if checkpointing.is_in_first_forward():
                with torch.no_grad():
                    (
                        _,
                        top_tokens_per_expert_indices,
                    ) = expert_mask.topk(k=capacity, dim=0)
                    self._checkpointed_top_tokens_per_expert_indices = (
                        top_tokens_per_expert_indices
                    )

            if checkpointing.is_in_second_forward():
                with torch.no_grad():
                    top_tokens_per_expert_indices = (
                        self._checkpointed_top_tokens_per_expert_indices
                    )

            assert isinstance(top_tokens_per_expert_indices, torch.Tensor)
            top_tokens_per_expert_values = torch.gather(
                expert_mask,
                dim=1,
                index=top_tokens_per_expert_indices,
            )
        else:
            (
                top_tokens_per_expert_values,
                top_tokens_per_expert_indices,
            ) = expert_mask.topk(k=capacity, dim=0)
        return top_tokens_per_expert_values, top_tokens_per_expert_indices

    def choose_expert(self, gate_out) -> tuple[torch.Tensor, torch.Tensor]:
        checkpointing_enabled = (
            checkpointing.is_in_first_forward() or checkpointing.is_in_second_forward()
        )
        if checkpointing_enabled:
            if checkpointing.is_in_first_forward():
                with torch.no_grad():
                    _, expert_index = torch.topk(gate_out, k=self.routing_top_k, dim=1)
                    self._checkpointed_expert_index = expert_index

            if checkpointing.is_in_second_forward():
                with torch.no_grad():
                    expert_index = self._checkpointed_expert_index

            assert isinstance(expert_index, torch.Tensor)
            expert_gate = torch.gather(gate_out, dim=1, index=expert_index)
        else:
            expert_gate, expert_index = torch.topk(
                gate_out, k=self.routing_top_k, dim=1
            )
        return expert_gate, expert_index

    def log_light(self):
        return {
            "dropped_tokens_ratio": self.logging_cache["dropped_tokens_ratio"],
            "load_balancing_loss": self.logging_cache["load_balancing_loss"],
        }

    def log_heavy(self):
        return {
            "gate_softmax_all_values": make_histogram(
                self.logging_cache["gate_softmax_all_values"].flatten()  # move
            ),
            "tokens_per_expert_counts": make_histogram(
                self.logging_cache["tokens_per_expert"]
            ),
        }

    def route(
        self,
        x: torch.Tensor,
        top_tokens_per_expert_values: torch.Tensor,
        top_tokens_per_expert_indices: torch.Tensor,
        masked_expert_gate: torch.Tensor,
        capacity: int,
    ):
        with measure_time(self, "assign_tokens_to_input"):
            experts_input = x[
                top_tokens_per_expert_indices.T.reshape((self.n_experts * capacity,)),
                :,
            ] * top_tokens_per_expert_values.T.reshape((self.n_experts * capacity, 1))
            experts_input = experts_input.reshape(self.n_experts, capacity, self.dinput)
        return experts_input, top_tokens_per_expert_indices, masked_expert_gate


class MambaRouter(LoggingLayer):
    def __init__(
        self,
        dinput: int,
        capacity_factor: float,
        load_balancing_loss_weight: float,
        n_experts_per_group: list[int],
        routing_groups: list[list[str]],
        init_type: str,
        init_scale: float,
    ):
        """
        Args:
        """
        super().__init__()

        self.dinput = dinput
        self.n_experts_per_group = n_experts_per_group
        self.capacity_factor = capacity_factor
        self.load_balancing_loss_weight = load_balancing_loss_weight
        self.routing_groups = routing_groups
        assert len(routing_groups) == len(n_experts_per_group)
        self.routers = nn.ModuleList(
            [
                self._make_router(
                    init_type=init_type, init_scale=init_scale, n_experts=n_experts
                )
                for n_experts in n_experts_per_group
            ]
        )

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        n_tokens = batch_size * seq_len

        router_outputs = [router(x) for router in self.routers]
        output_unordered = []
        x = x.flatten(0, 1)
        for (
            experts_input,
            top_tokens_per_expert_indices,
            masked_expert_gate,
        ) in router_outputs:
            assert masked_expert_gate.shape[0] == n_tokens
            dropped_tokens_mask = masked_expert_gate.sum(dim=1) == 0
            assert dropped_tokens_mask.shape == (n_tokens,)
            dropped_tokens = x[dropped_tokens_mask]
            output_unordered.append(
                (
                    experts_input,
                    top_tokens_per_expert_indices,
                    dropped_tokens_mask,
                    dropped_tokens,
                )
            )
        output = []
        # organize the output, so that the order matches MAMBA_MODULES_ORDER
        for module in MAMBA_MODULES_ORDER:
            for group, router_out in zip(self.routing_groups, output_unordered):
                if module in group:
                    output.append(router_out)
                    break
        return output

    def _make_router(self, init_type, init_scale, n_experts):
        return TokenChoiceSeparateRouter(
            dinput=self.dinput,
            n_experts=n_experts,
            capacity_factor=self.capacity_factor,
            load_balancing_loss_weight=self.load_balancing_loss_weight,
            init_type=init_type,
            init_scale=init_scale,
            routing_top_k=1,
            use_einsum=False,
            vectorize=True,
        )


class MambaTokenChoiceFunction(LoggingLayer):
    def __init__(
        self,
        n_experts,
    ):
        super().__init__()
        self.n_experts = n_experts

    def forward(
        self,
        expert_inputs,
        top_tokens_per_expert_indices,
        dropped_tokens_mask,
        dropped_tokens,
    ):
        experts_output, dropped_tokens_output = self._inner_forward(
            expert_inputs, dropped_tokens
        )

        num_tokens = dropped_tokens_mask.shape[0]
        doutput = experts_output.shape[-1]
        output = torch.zeros(num_tokens, doutput, device=experts_output.device)

        with measure_time(self, "assign_tokens_to_output"):
            output.index_add_(
                dim=0,
                index=top_tokens_per_expert_indices.T.flatten(),
                source=experts_output.reshape(
                    self.n_experts * experts_output.shape[1], doutput
                ),
            )
            output *= ~dropped_tokens_mask[:, None]
        output[dropped_tokens_mask] = dropped_tokens_output

        return output

    def _inner_forward(
        self,
        expert_inputs,
        dropped_tokens,
    ):
        raise NotImplementedError()


class LinearExpertsMamba(MambaTokenChoiceFunction):
    def __init__(self, dinput, doutput, n_experts, init_type, init_scale):
        super().__init__(n_experts)
        self.dinput = dinput
        self.doutput = doutput
        self.lin_experts = nn.Parameter(
            get_init_weight(
                shape=(n_experts, dinput, doutput),
                fan_in=dinput,
                init_type=init_type,
                scale=init_scale,
            )
        )

        self.lin_dropped = nn.Parameter(
            get_init_weight(
                shape=(dinput, doutput),
                fan_in=dinput,
                init_type=init_type,
                scale=init_scale,
            )
        )

    def _inner_forward(self, expert_inputs, dropped_tokens):
        (n_experts, capacity, dinput) = expert_inputs.shape
        assert n_experts == self.n_experts
        assert dinput == self.dinput

        with measure_time(self, "process_by_experts"):
            experts_output = torch.matmul(expert_inputs, self.lin_experts)

        with measure_time(self, "process_dropped"):
            dropped_output = torch.matmul(dropped_tokens, self.lin_dropped)

        assert experts_output.shape == (n_experts, capacity, self.doutput)
        assert dropped_output.shape == (capacity, self.doutput)

        return experts_output, dropped_output


class NoExpertsMamba(MambaTokenChoiceFunction):
    """
    This should work like a "normal" linear layer, just with common interface for ease of use
    """

    def __init__(self, dinput, doutput, n_experts, init_type, init_scale):
        super().__init__(n_experts)
        self.dinput = dinput
        self.doutput = doutput
        self.lin = nn.Parameter(
            get_init_weight(
                shape=(dinput, doutput),
                fan_in=dinput,
                init_type=init_type,
                scale=init_scale,
            )
        )

    def _inner_forward(self, expert_inputs, dropped_tokens):
        (n_experts, capacity, dinput) = expert_inputs.shape
        assert self.n_experts == n_experts
        assert dinput == self.dinput

        with measure_time("process_by_experts"):
            experts_output = torch.matmul(expert_inputs, self.lin)

        with measure_time("process_dropped"):
            dropped_output = torch.matmul(dropped_tokens, self.lin)

        assert experts_output.shape == (n_experts, capacity, self.doutput)
        assert dropped_output.shape == (capacity, self.doutput)

        return experts_output, dropped_output


class MambaTokenChoice(LoggingLayer):
    """
    As of now, output_module does NOT work with router, I have to fix that in some weird way
    """

    def __init__(
        self,
        d_model,
        input_module: MambaTokenChoiceFunction,
        gate_module: MambaTokenChoiceFunction,
        output_module: MambaTokenChoiceFunction,
        router: MambaRouter,
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

        router_input, router_gate, router_output = self.router(hidden_states)

        # We do matmul and transpose BLH -> HBL at the same time
        x = self.input_module(*router_input)  # (B L D) -> (B L D)
        z = self.gate_module(*router_gate)  # (B L D) -> (B L D)

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
