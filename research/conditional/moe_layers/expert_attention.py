from typing import Literal
import torch
import torch.nn.functional as F
from torch.nn import LayerNorm
from fancy_einsum import einsum

from lizrd.core import nn
from lizrd.core.initialization import get_init_weight
from lizrd.core.llm import attention_mechanism
from lizrd.core.misc import Linear
from lizrd.support.logging import make_histogram
from research.conditional.moe_layers.expert_choice import make_heatmap
from research.conditional.utils.layer_manager import LoggingLayer
from research.conditional.utils.layer_manager import measure_time


class ExpertChoiceAttention(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        n_experts: int,
        expert_dhead: int,
        topk: int,
        init_type: Literal["kaiming_uniform", "truncated_normal"],
        init_scale: float,
        softmax_over: Literal["tokens", "experts"] = "experts",
        n_gating_heatmaps: int = 4,
        flash: bool = False,
        use_einsum: bool = False,
    ):
        """
        Args:
            dmodel: dimension of the input
            n_experts: number of experts
            expert_dhead: size of each expert
            topk: number of tokens that will be chosen for each expert
        """
        super().__init__()

        self.dmodel = dmodel
        self.n_experts = n_experts
        self.expert_dhead = expert_dhead
        self.topk = topk
        self.n_gating_heatmaps = n_gating_heatmaps
        self.flash = flash
        self.use_einsum = use_einsum

        assert softmax_over in ["tokens", "experts"]

        self.input_projection = nn.Parameter(
            get_init_weight(
                (self.n_experts, self.dmodel, 3 * self.expert_dhead),
                fan_in=self.dmodel,
                init_type=init_type,
                scale=init_scale,
            ),
        )
        self.output_projection = Linear(
            self.n_experts * self.expert_dhead,
            self.dmodel,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.gate = nn.Parameter(
            get_init_weight(
                (self.dmodel, self.n_experts),
                fan_in=self.dmodel,
                init_type=init_type,
                scale=init_scale,
            )
        ).requires_grad_(True)
        self.ln = LayerNorm(dmodel)
        self.softmax_over = softmax_over

    def forward(self, x: torch.Tensor):
        # x is (batch, seq_len, dmodel)
        batch_size, seq_len = x.shape[0], x.shape[1]

        topk_indices, topk_values = self.expert_gating(x, batch_size, seq_len)

        if self.use_einsum:
            x, one_hot = self.extract_with_linear_einsum(
                x, topk_indices, seq_len, self.input_projection
            )
        else:
            x, one_hot = self.extract_with_linear_bmm(
                x, topk_indices, seq_len, self.input_projection
            )

        q, k, v = torch.chunk(x, chunks=3, dim=-1)

        attention_output = attention_mechanism(
            query=q,
            key=k,
            value=v,
            dhead=self.expert_dhead,
            causal=False,  # encoder-only
            flash=self.flash,
        ).transpose(0, 1)

        # Multiply by topk_values
        topk_values = torch.unsqueeze(topk_values, -1)
        attention_output = torch.mul(attention_output, topk_values)

        # Unsqueeze topk (dense form used for attention) to seq_len

        if self.use_einsum:
            attention_output = einsum(
                "n_exp batch_size topk exp_size, n_exp batch_size topk seq_len "
                "-> batch_size seq_len n_exp exp_size",
                attention_output,
                one_hot,
            )
        else:
            one_hot = one_hot.transpose(2, 3)
            # MATMUL:
            # n_exp x batch_size x seq_len x topk           <- one_hot (after transpose)
            # n_exp x batch_size x topk    x exp_size       <- attention_output
            # ---------------------------------------
            # n_exp x batch_size x seq_len x exp_size
            attention_output = torch.matmul(one_hot, attention_output).permute(
                1, 2, 0, 3
            )

        # Output projection
        attention_output = self.output_projection(attention_output.flatten(-2))

        with measure_time(self, "moatt_layer_norm"):
            attention_output = self.ln(attention_output)

        return attention_output

    def expert_gating(self, x: torch.Tensor, batch_size: int, seq_len: int):
        # expert embedding
        with measure_time(self, "moatt_expert_embedding"):
            gate = self.gate.unsqueeze(0).expand(batch_size, -1, -1)
            gate_out = torch.bmm(x, gate).permute(2, 0, 1)
            assert gate_out.shape == (self.n_experts, batch_size, seq_len)
        self.update_cache_for_logging("moatt_unflatten_gate_out", gate_out)

        # perform softmax either over tokens for each expert or over experts for each token
        with measure_time(self, "moatt_softmax"):
            if self.softmax_over == "tokens":
                gate_out = torch.softmax(gate_out, dim=-1)
            elif self.softmax_over == "experts":
                gate_out = torch.softmax(gate_out, dim=0)

        self.update_cache_for_logging("moatt_gate_softmax_all_values", gate_out)
        # choose topk tokens for each expert
        with measure_time(self, "moatt_topk"):
            topk_values, topk_indices = torch.topk(gate_out, k=self.topk, dim=-1)

        # cache values for logging
        self.update_cache_for_logging("moatt_gate_softmax_topk_vals", topk_values)
        self.update_cache_for_logging("moatt_topk_indices", topk_indices)
        self.update_cache_for_logging(
            "moatt_n_tokens", torch.Tensor([batch_size * seq_len])
        )

        return topk_indices, topk_values

    def extract_with_linear_einsum(
        self, x: torch.Tensor, topk_indices: torch.Tensor, seq_len, weight
    ):
        with measure_time(self, "moatt_gate_preprocess_with_linear"):
            one_hot = F.one_hot(topk_indices, num_classes=seq_len).type(
                x.dtype
            )  # n_exp x batch_size x topk x seq_len
            x = einsum(
                "batch_size seq_len dmodel, n_exp batch_size topk seq_len, "
                "n_exp dmodel exp_size "
                "-> batch_size n_exp topk exp_size",
                x,
                one_hot,
                weight,
            )
            return x, one_hot

    def extract_with_linear_bmm(
        self, x: torch.Tensor, topk_indices: torch.Tensor, seq_len, weight
    ):
        with measure_time(self, "moatt_gate_preprocess_with_linear"):
            one_hot = F.one_hot(topk_indices, num_classes=seq_len).type(
                x.dtype
            )  # n_exp x batch_size x topk x seq_len
            # MATMUL:
            # n_exp x batch_size x topk    x seq_len     <- one_hot
            #         batch_size x seq_len x dmodel      <- x
            # --------------------------------------
            # n_exp x batch_size x topk x dmodel
            x = torch.matmul(one_hot, x).transpose(0, 1)
            # MATMUL:
            # batch_size x n_exp x topk   x dmodel         <- x (last output)
            #              n_exp x dmodel x exp_size       <- weight
            # --------------------------------------
            # batch_size x n_exp x topk x exp_size
            x = torch.matmul(x, weight)
            return x, one_hot

    def log_light(self):
        return dict()

    def log_heavy(self):
        # calculate indexes choose counts
        chosen_indexes = self.logging_cache["moatt_topk_indices"].flatten()
        chosen_indexes = torch.cat(
            (
                chosen_indexes,
                torch.Tensor([self.logging_cache["moatt_n_tokens"] - 1]).type(
                    chosen_indexes.type()
                ),
            )
        )  # make sure bincount takes into account the whole range of indexes
        indexes_choose_counts = chosen_indexes.bincount()
        return {
            "moatt_gate_softmax_topk_vals": make_histogram(
                self.logging_cache["moatt_gate_softmax_topk_vals"].flatten()
            ),
            "moatt_gate_softmax_all_values": make_histogram(
                self.logging_cache["moatt_gate_softmax_all_values"].flatten()
            ),
            "moatt_indexes_choose_counts": make_histogram(indexes_choose_counts),
            **{
                f"moatt_gating_heatmap_{i}": make_heatmap(
                    self.logging_cache["moatt_unflatten_gate_out"], i
                )
                for i in range(min(self.n_gating_heatmaps, self.n_experts))
            },
        }
