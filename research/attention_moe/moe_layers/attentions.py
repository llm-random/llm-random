import torch
from torch import nn
import torch.nn.functional as F
import math


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


class MoMQA(nn.Module):
    def __init__(self, n_embd, n_head, block_size, bias=False):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        # self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        head_dim = n_embd // n_head
        self.gating = nn.Linear(n_embd, n_head, bias=bias)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)

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

        return y
