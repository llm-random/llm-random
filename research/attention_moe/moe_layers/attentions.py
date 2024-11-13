import torch
from torch import nn
import torch.nn.functional as F
import math
from fancy_einsum import einsum

from lizrd.core.misc import LoggingLayer
from lizrd.support.logging import make_histogram


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, bias=False):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.n_head = n_head
        self.n_embd = n_embd
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = False
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(block_size, block_size)).view(
                    1, 1, block_size, block_size
                ),
            )

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        return y


class CausalMQA(nn.Module):
    def __init__(self, n_embd, n_head, block_size, bias=False):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        # self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        head_dim = n_embd // n_head
        self.kv_proj = nn.Linear(n_embd, 2 * head_dim, bias=bias)
        self.q_proj = nn.Linear(n_embd, n_embd, bias=bias)

        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = head_dim
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = False
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(block_size, block_size)).view(
                    1, 1, block_size, block_size
                ),
            )

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k, v = self.kv_proj(x).split(self.head_dim, dim=2)
        q = self.q_proj(x)
        k = k.unsqueeze(1)  # (B, 1, T, hs)
        v = v.unsqueeze(1)  # (B, 1, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            raise NotImplementedError("Flash Attention not implemented")
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        return y


def calculate_load_balancing_loss(
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
    load_balancing_loss = n_experts * dot_product / n_tokens
    return load_balancing_loss


def calculate_z_loss(zloss_weight: float = 0, gate_logits: torch.Tensor = None):
    zloss = torch.logsumexp(gate_logits, dim=0)
    zloss = torch.square(zloss)
    zloss = zloss.mean()
    zloss = zloss_weight * zloss

    return zloss


class MoMQA(LoggingLayer):
    def __init__(
        self,
        n_embd,
        n_head,
        block_size,
        load_balancing_loss_weight: float,
        multiply_by_n_head: bool,
        bias=False,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        # self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.load_balancing_loss_weight = load_balancing_loss_weight
        head_dim = n_embd // n_head
        self.gating = nn.Linear(n_embd, n_head, bias=bias)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.multiply_by_n_head = multiply_by_n_head

        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = head_dim
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = False
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(block_size, block_size)).view(
                    1, 1, block_size, block_size
                ),
            )

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # k, v = self.kv_proj(x).split(self.head_dim, dim=2)
        # q = self.q_proj(x)
        # k = k.unsqueeze(1)  # (B, 1, T, hs)
        # v = v.unsqueeze(1)  # (B, 1, T, hs)
        # q = q.view(B, T, self.n_head, C // self.n_head).transpose(
        #     1, 2
        # )  # (B, nh, T, hs)

        gating = self.gating(x)  # (B, T, nh)
        gating = gating.softmax(dim=-1)
        topk = gating.topk(k=1, dim=-1)
        topk_values = topk.values
        topk_indices = topk.indices
        gating_top1 = torch.zeros_like(gating)
        indicator = torch.zeros_like(gating)
        indicator.scatter_(dim=-1, index=topk_indices, src=torch.ones_like(topk_values))
        gating_top1.scatter_(dim=-1, index=topk_indices, src=topk_values)
        # breakpoint()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head)
        k = (k * indicator.unsqueeze(-1)).sum(dim=-2, keepdim=True).transpose(-2, -3)
        v = v.view(B, T, self.n_head, C // self.n_head)
        v = (v * gating_top1.unsqueeze(-1)).sum(dim=-2, keepdim=True).transpose(-2, -3)

        # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            raise NotImplementedError("Flash Attention not implemented")
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        n_tokens_per_expert = indicator.reshape(B * T, self.n_head).sum(dim=0)
        load_balancing_loss = calculate_load_balancing_loss(
            gating.reshape(B * T, self.n_head),
            n_tokens_per_expert,
        )

        if "load_balancing_losses" not in self.forward_pass_cache:
            self.forward_pass_cache["load_balancing_losses"] = []

        self.forward_pass_cache["load_balancing_losses"].append(
            self.load_balancing_loss_weight * load_balancing_loss
        )

        self.update_cache_for_logging(
            "load_balancing_loss",
            load_balancing_loss,
        )
        self.update_cache_for_logging(
            "gate_softmax_all_values", gating.reshape(B * T, self.n_head)
        )
        self.update_cache_for_logging("tokens_per_expert", n_tokens_per_expert)
        if self.multiply_by_n_head:
            y = y * self.n_head
        return y * self.n_head

    def log_light(self):
        return {
            # "dropped_tokens_ratio": self.logging_cache["dropped_tokens_ratio"],
            "load_balancing_loss": self.logging_cache["load_balancing_loss"],
            # "z_loss": self.logging_cache["z_loss"],
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


class DroppingMoMQA(LoggingLayer):
    def __init__(
        self,
        n_embd,
        n_head,
        block_size,
        load_balancing_loss_weight: float,
        multiply_by_n_head: bool,
        bias=False,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        # self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.load_balancing_loss_weight = load_balancing_loss_weight
        head_dim = n_embd // n_head
        self.gating = nn.Linear(n_embd, n_head, bias=bias)
        # self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.k_proj = torch.nn.Parameter(torch.randn(n_head, n_embd, head_dim))
        self.v_proj = torch.nn.Parameter(torch.randn(n_head, n_embd, head_dim))
        self.q_proj = torch.nn.Linear(n_embd, n_embd, bias=bias)
        self.dropped_k = torch.nn.Parameter(torch.randn(1, head_dim))
        self.dropped_v = torch.nn.Parameter(torch.randn(1, head_dim))
        self.multiply_by_n_head = multiply_by_n_head

        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = head_dim
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = False
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(block_size, block_size)).view(
                    1, 1, block_size, block_size
                ),
            )

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # k, v = self.kv_proj(x).split(self.head_dim, dim=2)
        # q = self.q_proj(x)
        # k = k.unsqueeze(1)  # (B, 1, T, hs)
        # v = v.unsqueeze(1)  # (B, 1, T, hs)
        # q = q.view(B, T, self.n_head, C // self.n_head).transpose(
        #     1, 2
        # )  # (B, nh, T, hs)
        x = x.view(B * T, C)
        gating = self.gating(x)  # (B, T, nh)
        gating = gating.softmax(dim=-1)
        topk = gating.topk(k=1, dim=-1)
        topk_values = topk.values
        topk_indices = topk.indices
        gating_top1 = torch.zeros_like(gating)
        indicator = torch.zeros_like(gating)
        indicator.scatter_(dim=-1, index=topk_indices, src=torch.ones_like(topk_values))
        capacity = B * T // self.n_head
        tokens_per_expert = indicator.sum(dim=0).type(torch.int32)
        token_idx_within_expert = (indicator.cumsum(dim=0) * indicator).type(
            torch.int32
        )
        token_is_dropped = (token_idx_within_expert > capacity).sum(1, keepdim=True)
        # truncated_token_idx_within_expert = token_idx_within_expert * (
        #     token_idx_within_expert <= capacity
        # )
        truncated_token_idx_within_expert = token_idx_within_expert * (
            1 - token_is_dropped
        )
        # experts_input = x.reshape(B * T, -1)
        experts_input = torch.zeros(
            self.n_head, capacity + 1, C, device=x.device, dtype=x.dtype
        )
        experts_input.scatter_(
            dim=1,
            index=truncated_token_idx_within_expert.T.unsqueeze(-1).expand(
                self.n_head, -1, C
            ),
            # src=x.reshape(B * T, -1),
            src=x.unsqueeze(0).expand(self.n_head, -1, -1),
        ).unsqueeze_(-2)
        k = torch.matmul(experts_input, self.k_proj.unsqueeze(1)).squeeze(-2)
        k = k.reshape(-1, k.shape[-1])[
            (
                (
                    truncated_token_idx_within_expert
                    + torch.arange(0, self.n_head, device=x.device) * (capacity + 1)
                )
                * (truncated_token_idx_within_expert != 0)
            ).sum(-1)
        ]

        v = torch.matmul(experts_input, self.v_proj.unsqueeze(1)).squeeze(-2)
        v = v.reshape(-1, v.shape[-1])[
            (
                (
                    truncated_token_idx_within_expert
                    + torch.arange(0, self.n_head, device=x.device) * (capacity + 1)
                )
                * (truncated_token_idx_within_expert != 0)
            ).sum(-1)
        ]
        k[token_is_dropped] = self.dropped_k
        v[token_is_dropped] = self.dropped_v

        k = k.view(B, T, 1, self.head_dim).transpose(1, 2)
        v = v.view(B, T, 1, self.head_dim).transpose(1, 2)

        # q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # k = k.view(B, T, self.n_head, C // self.n_head)
        # v = v.view(B, T, self.n_head, C // self.n_head)

        q = self.q_proj(x)
        # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            raise NotImplementedError("Flash Attention not implemented")
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        n_tokens_per_expert = indicator.reshape(B * T, self.n_head).sum(dim=0)
        load_balancing_loss = calculate_load_balancing_loss(
            gating.reshape(B * T, self.n_head),
            n_tokens_per_expert,
        )

        if "load_balancing_losses" not in self.forward_pass_cache:
            self.forward_pass_cache["load_balancing_losses"] = []

        self.forward_pass_cache["load_balancing_losses"].append(
            self.load_balancing_loss_weight * load_balancing_loss
        )

        self.update_cache_for_logging(
            "load_balancing_loss",
            load_balancing_loss,
        )
        self.update_cache_for_logging(
            "gate_softmax_all_values", gating.reshape(B * T, self.n_head)
        )
        self.update_cache_for_logging("tokens_per_expert", n_tokens_per_expert)
        if self.multiply_by_n_head:
            y = y * self.n_head
        return y * self.n_head

    def log_light(self):
        return {
            # "dropped_tokens_ratio": self.logging_cache["dropped_tokens_ratio"],
            "load_balancing_loss": self.logging_cache["load_balancing_loss"],
            # "z_loss": self.logging_cache["z_loss"],
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
