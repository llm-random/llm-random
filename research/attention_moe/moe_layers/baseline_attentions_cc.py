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
