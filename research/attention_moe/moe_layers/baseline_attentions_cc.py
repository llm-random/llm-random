from typing import Optional
from lizrd.core.llm import RoPE
from lizrd.core.misc import Linear, LoggingLayer


import torch


class MQA(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        n_heads: int,
        init_type: str,
        init_scale: float,
    ):
        super().__init__()
        self.dmodel = dmodel
        self.n_heads = n_heads
        assert self.dmodel % self.n_heads == 0
        self.head_dim = self.dmodel // self.n_heads
        self.q_proj = Linear(
            self.dmodel, self.dmodel, init_type=init_type, init_scale=init_scale
        )
        self.o_proj = Linear(
            self.dmodel, self.dmodel, init_type=init_type, init_scale=init_scale
        )
        kv_proj = Linear(
            self.dmodel, 2 * self.head_dim, init_type=init_type, init_scale=init_scale
        )
        self.expert_weights = torch.nn.Parameter(kv_proj.weight.T)
        assert self.expert_weights.shape == (
            self.dmodel,
            2 * self.head_dim,
        )

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        kv = torch.einsum("ab,...a->...b", self.expert_weights, x)
        k, v = kv.split(self.head_dim, dim=-1)
        # with sdpa_kernel(backends=[SDPBackend.MATH]):
        k = k.unsqueeze(-3)
        v = v.unsqueeze(-3)
        # print("k", k.shape)
        # print("q", q.shape)
        # print("v", v.shape)
        y = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2).contiguous(),
            k.contiguous(),
            v.contiguous(),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            enable_gqa=True,
        ).transpose(1, 2)
        y = y.flatten(-2, -1)
        y = self.o_proj(y)
        return y


class GQA(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        n_kv_heads: Optional[int],
        n_heads: int,
        init_type: str,
        init_scale: float,
        use_rope: bool = False,
        use_qk_norm: bool = False,
        cutoff: Optional[int] = None,
    ):
        super().__init__()
        self.dmodel = dmodel
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        assert self.dmodel % self.n_heads == 0
        self.head_dim = self.dmodel // self.n_heads
        self.q_proj = Linear(
            self.dmodel, self.dmodel, init_type=init_type, init_scale=init_scale
        )
        self.o_proj = Linear(
            self.dmodel, self.dmodel, init_type=init_type, init_scale=init_scale
        )
        self.kv_proj = Linear(
            self.dmodel,
            2 * self.n_kv_heads * self.head_dim,
            init_type=init_type,
            init_scale=init_scale,
        )

        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = torch.nn.LayerNorm(self.head_dim)
            self.k_norm = torch.nn.LayerNorm(self.head_dim)

        self.use_rope = use_rope
        if self.use_rope:
            assert cutoff is not None
            self.rotary_emb = RoPE(dhead=self.head_dim, length=cutoff)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        kv = self.kv_proj(x).view(
            batch_size, seq_len, 2 * self.n_kv_heads, self.head_dim
        )
        k, v = kv.split(self.n_kv_heads, dim=-2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rope:
            q = self.rotary_emb(q)
            k = self.rotary_emb(k)

        y = torch.nn.functional.scaled_dot_product_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            enable_gqa=True,
        ).transpose(1, 2)
        y = y.flatten(-2, -1)
        y = self.o_proj(y)
        return y


class VanillaAttention(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        n_heads: int,
        init_type: str,
        init_scale: float,
    ):
        """
        Args:
            dmodel: dimension of the input
            doutput: dimension of the output (default: dmodel)
            n_experts: number of experts
            expert_size: size of each expert
            capacity_factor: scalar that determines how many tokens can be assigned to each expert
            load_balancing_loss_weight: weight of the auxillary loss
            expert_logic: expert logic layer, takes input of shape (n_experts, capacity, dmodel) and returns output of shape (n_experts, capacity, dmodel)
        """
        super().__init__()
        self.dmodel = dmodel
        self.n_heads = n_heads
        assert self.dmodel % self.n_heads == 0
        self.head_dim = self.dmodel // self.n_heads
        self.q_proj = Linear(
            self.dmodel, self.dmodel, init_type=init_type, init_scale=init_scale
        )
        self.o_proj = Linear(
            self.dmodel, self.dmodel, init_type=init_type, init_scale=init_scale
        )
        kv_proj = Linear(
            self.dmodel, 2 * self.dmodel, init_type=init_type, init_scale=init_scale
        )
        self.expert_weights = torch.nn.Parameter(kv_proj.weight.T)
        assert self.expert_weights.shape == (
            self.dmodel,
            2 * self.dmodel,
        )

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        kv = torch.einsum("ab,...a->...b", self.expert_weights, x)
        k, v = kv.view(batch_size, seq_len, self.n_heads, 2 * self.head_dim).split(
            self.head_dim, dim=-1
        )
        # print("k", k.shape)
        # print("q", q.shape)
        # print("v", v.shape)
        y = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            enable_gqa=False,
        ).transpose(1, 2)
        y = y.flatten(-2, -1)
        y = self.o_proj(y)
        return y
