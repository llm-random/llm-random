import torch
import torch.nn.functional as F
from fancy_einsum import einsum

from lizrd.core import nn
from lizrd.core.initialization import get_init_weight
from lizrd.support.logging import make_histogram
from lizrd.train import checkpointing
from research.conditional.utils.layer_manager import LoggingLayer
from research.conditional.utils.layer_manager import measure_time


class TokenChoiceRouter(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        n_experts: int,
        capacity_factor: float,
        load_balancing_loss_weight: float,
        init_type: str,
        init_scale: float,
        routing_top_k: int = 1,
        use_einsum: bool = False,
    ):
        super().__init__()

        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.load_balancing_loss_weight = load_balancing_loss_weight
        self.use_einsum = use_einsum
        self.dmodel = dmodel
        self.routing_top_k = routing_top_k
        self._checkpointed_expert_index = None

        self.gate = nn.Parameter(
            get_init_weight(
                shape=(dmodel, n_experts),
                fan_in=dmodel,
                init_type=init_type,
                scale=init_scale,
            )
        )

    def forward(self, x: torch.Tensor):
        # x is (batch, seq_len, dmodel)
        batch_size, seq_len, _ = x.shape
        n_tokens = batch_size * seq_len
        self.update_cache_for_logging("n_tokens", torch.Tensor([n_tokens]))

        x = x.flatten(start_dim=0, end_dim=1)

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

        # perform softmax over experts for each token
        with measure_time(self, "softmax"):
            gate_out = torch.softmax(gate_out, dim=1)

        self.update_cache_for_logging("gate_softmax_all_values", gate_out)

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

        with measure_time(self, "calculate expert indexes"):
            position_in_expert = torch.cumsum(expert_mask, dim=0) * expert_mask
            in_capacity_tokens_mask = torch.lt(position_in_expert, capacity)
            expert_mask *= in_capacity_tokens_mask
            n_selected_tokens = expert_mask.sum().item()

            self.update_cache_for_logging(
                "dropped_tokens_ratio",
                ((n_tokens * self.routing_top_k) - n_selected_tokens)
                / (n_tokens * self.routing_top_k),
            )

        with measure_time(self, "calculate aux loss"):
            position_in_expert_mask = position_in_expert.bool()
            tokens_per_expert = position_in_expert_mask.sum(dim=0, dtype=gate_out.dtype)
            load_balancing_loss = calculate_load_balancing_loss(
                self.load_balancing_loss_weight,
                gate_out,
                tokens_per_expert,
                use_einsum=self.use_einsum,
            )

            if "load_balancing_losses" not in self.forward_pass_cache:
                self.forward_pass_cache["load_balancing_losses"] = [load_balancing_loss]
            else:
                self.forward_pass_cache["load_balancing_losses"].append(
                    load_balancing_loss
                )

        self.update_cache_for_logging("gate_softmax_values", expert_gate)
        self.update_cache_for_logging("max_indices", expert_index)
        self.update_cache_for_logging("tokens_per_expert", tokens_per_expert)
        self.update_cache_for_logging("load_balancing_loss", load_balancing_loss)
        # group tokens indices by expert it should be processed by
        with measure_time(self, "experts_lists"):
            indices_of_tokens_for_expert = [
                single_expert_mask.nonzero(as_tuple=True)[0]
                for single_expert_mask in expert_mask.transpose(0, 1)
            ]

        masked_expert_gate = gate_out * expert_mask

        # create empty input and  assign only tokens to be processed
        experts_input = torch.zeros(
            self.n_experts, capacity, self.dmodel, dtype=x.dtype, device=x.device
        )
        with measure_time(self, "assign_tokens_to_input"):
            for i, indices in enumerate(indices_of_tokens_for_expert):
                experts_input[i, : len(indices)] = x[indices]

        return experts_input, indices_of_tokens_for_expert, masked_expert_gate

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


class TokenChoiceFF(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        n_experts: int,
        expert_size: int,
        capacity_factor: float,
        load_balancing_loss_weight: float,
        init_type: str,
        init_scale: float,
        routing_top_k: int = 1,
        use_einsum: bool = False,
    ):
        """
        Args:
            dmodel: dimension of the input
            n_experts: number of experts
            expert_size: size of each expert
            capacity_factor: scalar that determines how many tokens can be assigned to each expert
            load_balancing_loss_weight: weight of the auxillary loss
        """
        super().__init__()

        self.dmodel = dmodel
        self.dinner = 2 * dmodel
        self.n_experts = n_experts
        self.expert_size = expert_size
        self.capacity_factor = capacity_factor
        self.load_balancing_loss_weight = load_balancing_loss_weight
        self.use_einsum = use_einsum

        self.in_proj = nn.Parameter(
            get_init_weight(
                shape=(n_experts, dmodel, 2 * self.dinner),
                fan_in=dmodel,
                init_type=init_type,
                scale=init_scale,
            )
        )
        self.out_proj = nn.Parameter(
            get_init_weight(
                shape=(n_experts, self.dinner, dmodel),
                fan_in=dmodel,
                init_type=init_type,
                scale=init_scale,
            )
        )

        self.router = TokenChoiceRouter(
            dmodel=dmodel,
            n_experts=n_experts,
            capacity_factor=capacity_factor,
            load_balancing_loss_weight=load_balancing_loss_weight,
            init_type=init_type,
            init_scale=init_scale,
            routing_top_k=routing_top_k,
            use_einsum=use_einsum,
        )

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        experts_input, indices_of_tokens_for_expert, masked_expert_gate = self.router(x)
        x = x.flatten(start_dim=0, end_dim=1)

        # with measure_time(self, "process_by_experts"):
        #     if self.use_einsum:
        #         experts_output = einsum(
        #             "n_experts capacity dmodel, n_experts dmodel expert_size -> n_experts capacity expert_size",
        #             experts_input,
        #             self.lin1_weight,
        #         )
        #     else:
        #         experts_output = torch.matmul(experts_input, self.lin1_weight)

        #     experts_output = F.relu(experts_output)
        #     if self.use_einsum:
        #         experts_output = einsum(
        #             "n_experts capacity expert_size, n_experts expert_size dmodel -> n_experts capacity dmodel",
        #             experts_output,
        #             self.lin2_weight,
        #         )
        #     else:
        #         experts_output = torch.matmul(experts_output, self.lin2_weight)

        experts_output = experts_output.to(x.dtype)
        output = torch.zeros_like(x)

        with measure_time(self, "assign_tokens_to_output"):
            for expert_id in range(self.n_experts):
                tokens_to_update = indices_of_tokens_for_expert[expert_id]
                num_of_tokens_to_update = len(tokens_to_update)

                output[tokens_to_update] += experts_output[
                    expert_id, :num_of_tokens_to_update
                ] * masked_expert_gate[tokens_to_update, expert_id].unsqueeze(dim=1)

        output = output.reshape((batch_size, seq_len, self.dmodel))

        return output


def calculate_load_balancing_loss(
    alpha: float,
    softmax_per_token: torch.Tensor,
    n_tokens_in_each_expert: torch.Tensor,
    use_einsum: bool = False,
):
    """
    Calculates the load balancing loss for the token choice layer.

    :param str alpha: aux loss weigth parameter
    :param torch.Tensor softmax_per_token: tensor of shape (tokens, n_experts)
    :param torch.Tensor tokens_in_each_expert: tensor of shape (n_experts)
    """
    n_tokens, n_experts = softmax_per_token.shape
    assert n_experts == n_tokens_in_each_expert.shape[0]

    per_expert_softmax_sum = torch.mean(softmax_per_token, dim=0)

    if use_einsum:
        dot_product = einsum("i, i ->", per_expert_softmax_sum, n_tokens_in_each_expert)
    else:
        dot_product = torch.dot(per_expert_softmax_sum, n_tokens_in_each_expert)
    load_balancing_loss = alpha * n_experts * dot_product / n_tokens
    return load_balancing_loss


import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class MambaInner(nn.Module):
    def __init__(
        self,
        d_model,
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
        bias=False,
        use_fast_path=True,  # Fused kernel options
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
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )  # 4 dmodel^2

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )  # 16 dmodel^2

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )  # dmodel * (dmodel/8 + 64) = dmodel^2 / 8 + 64 dmodel (< dmodel^2)
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

        # self.out_proj = nn.Linear(
        #     self.d_inner, self.d_model, bias=bias, **factory_kwargs
        # )

    def forward(self, xz, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        # batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        # if inference_params is not None:
        #     conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
        #     if inference_params.seqlen_offset > 0:
        #         # The states are updated inplace
        #         out, _, _ = self.step(hidden_states, conv_state, ssm_state)
        #         return out

        # We do matmul and transpose BLH -> HBL at the same time
        # xz = rearrange(
        #     self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
        #     "d (b l) -> b d l",
        #     l=seqlen,
        # )
        # if self.in_proj.bias is not None:
        #     xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if (
            self.use_fast_path and inference_params is None
        ):  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(
                    F.pad(x, (self.d_conv - x.shape[-1], 0))
                )  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
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
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert (
            hidden_states.shape[1] == 1
        ), "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(
                torch.roll(conv_state, shifts=-1, dims=-1)
            )  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(
                conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
            )  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state,
                x,
                dt,
                A,
                B,
                C,
                self.D,
                z=z,
                dt_bias=self.dt_proj.bias,
                dt_softplus=True,
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size,
            self.d_model * self.expand,
            self.d_conv,
            device=device,
            dtype=conv_dtype,
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size,
            self.d_model * self.expand,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        return conv_state, ssm_state

    def _get_states_from_cache(
        self, inference_params, batch_size, initialize_states=False
    ):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (
                conv_state,
                ssm_state,
            )
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[
                self.layer_idx
            ]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
