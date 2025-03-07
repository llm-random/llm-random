from lizrd.core.misc import Linear
from research.attention_moe.diff_attn.fast import (
    Lowrank,
    lambda_init_fn,
    manual_attention,
)

import torch
from torch import nn

from lizrd.core.misc import Linear, LoggingLayer

from .kernel.rotary import apply_rotary_emb
from flash_attn import flash_attn_func
from flash_attn.layers.rotary import RotaryEmbedding

try:
    from apex.normalization import FusedRMSNorm as RMSNorm
except ModuleNotFoundError:
    # print("No fused RMSNorm")
    from .rms_norm import RMSNorm


class AdapterDifferentialAttention(LoggingLayer):
    """
    (Recommended)
    DiffAttn implemented with FlashAttention, for packages that support different qk/v dimensions
    e.g., our customized-flash-attention (https://aka.ms/flash-diff) and xformers (https://github.com/facebookresearch/xformers)
    """

    def __init__(
        self,
        # args,
        embed_dim,
        # depth,
        num_heads,
        use_rope,
        seq_len,
        lowrank_inner_dim,
        flip_negative_heads,
        roll_negative_heads,
        init_type,
        init_scale,
        num_kv_heads=None,
        adapter_type: str = "lora",
        lowrank_dtype=None,
        use_qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        # self.args = args
        self.embed_dim = embed_dim
        self.save_attention_weights = False
        self.attention_weights = None
        self.num_positive_heads = num_heads // 2
        assert (int(roll_negative_heads) + int(flip_negative_heads)) <= 1

        self.num_positive_kv_heads = (num_kv_heads or num_heads) // 2
        self.n_rep = self.num_positive_heads // self.num_positive_kv_heads
        self.adapter_type = adapter_type

        self.head_dim = embed_dim // num_heads

        q_proj_out_dim = self.head_dim * self.num_positive_heads
        assert self.adapter_type == "lora"
        k_proj_out_dim = self.head_dim * self.num_positive_kv_heads

        self.adapter_type = adapter_type
        self.lowrank_inner_dim = lowrank_inner_dim
        if self.adapter_type == "lora" and self.lowrank_inner_dim > 0:
            self.lowrank_q = Lowrank(
                embed_dim,
                self.lowrank_inner_dim,
                init_type,
                init_scale,
                output_dim=q_proj_out_dim,
                dtype=lowrank_dtype,
            )
            self.lowrank_k = Lowrank(
                embed_dim,
                self.lowrank_inner_dim,
                init_type,
                init_scale,
                output_dim=k_proj_out_dim,
                dtype=lowrank_dtype,
            )
        else:
            raise ValueError(f"Adapter type {self.adapter_type} not supported")

        self.q_proj = Linear(
            embed_dim,
            q_proj_out_dim,
            bias=True,
            init_type=init_type,
            init_scale=init_scale,
        )

        self.k_proj = Linear(
            embed_dim,
            k_proj_out_dim,
            bias=True,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.v_dim = self.embed_dim // self.num_positive_heads
        self.v_proj = Linear(
            embed_dim,
            self.v_dim * self.num_positive_kv_heads,
            bias=True,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.out_proj = Linear(
            embed_dim, embed_dim, bias=False, init_type=init_type, init_scale=init_scale
        )

        self.lambda_init = None
        self.use_rope = use_rope
        self.seq_len = seq_len
        self.flip_negative_heads = flip_negative_heads
        self.roll_negative_heads = roll_negative_heads

        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )

        self.subln = RMSNorm(self.v_dim, eps=rms_norm_eps, elementwise_affine=True)
        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(
                self.head_dim, eps=rms_norm_eps, elementwise_affine=True
            )
            self.k_norm = RMSNorm(
                self.head_dim, eps=rms_norm_eps, elementwise_affine=True
            )

        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                base=rope_theta,
                interleaved=True,
            )
            self.rotary_emb._update_cos_sin_cache(self.seq_len, dtype=torch.float32)

    def forward(
        self,
        x,
        rel_pos=None,
        attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        if self.lambda_init is None:
            self.lambda_init = lambda_init_fn(self.block_number + 1)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.adapter_type == "lora":
            # self.lowrank_inner_dim > 0:
            q_negative = (q + self.lowrank_q(x)).view(
                bsz, tgt_len, self.num_positive_heads, self.head_dim
            )
            k_negative = (k + self.lowrank_k(x)).view(
                bsz, src_len, self.num_positive_kv_heads, self.head_dim
            )
            q = q.view(bsz, tgt_len, self.num_positive_heads, self.head_dim)
            k = k.view(bsz, src_len, self.num_positive_kv_heads, self.head_dim)
            v = v.view(bsz, src_len, self.num_positive_kv_heads, self.v_dim)
        else:
            raise ValueError(f"Adapter type {self.adapter_type} not supported")

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
            q_negative = self.q_norm(q_negative)
            k_negative = self.k_norm(k_negative)

        if self.use_rope:
            assert self.rotary_emb._cos_cached.dtype == torch.float32
            rel_pos = (
                self.rotary_emb._cos_cached.to(x.device),
                self.rotary_emb._sin_cached.to(x.device),
            )
            q = apply_rotary_emb(
                q.to(dtype=torch.float32), *rel_pos, interleaved=True
            ).to(x)
            k = apply_rotary_emb(
                k.to(dtype=torch.float32), *rel_pos, interleaved=True
            ).to(x)
            q_negative = apply_rotary_emb(
                q_negative.to(dtype=torch.float32), *rel_pos, interleaved=True
            ).to(x)
            k_negative = apply_rotary_emb(
                k_negative.to(dtype=torch.float32), *rel_pos, interleaved=True
            ).to(x)

        if self.adapter_type != "none":
            q1 = q
            q2 = q_negative
            k1 = k
            k2 = k_negative
            if self.num_positive_kv_heads != self.num_positive_heads:
                k1 = k1.repeat_interleave(self.n_rep, dim=2)
                k2 = k2.repeat_interleave(self.n_rep, dim=2)
                assert (
                    k1.shape == k2.shape == q1.shape == q2.shape
                ), f"Shapes don't match: {k1.shape}, {k2.shape}, {q1.shape}, {q2.shape}"

        if self.flip_negative_heads:
            q2 = torch.flip(q2, dims=(2,))
            k2 = torch.flip(k2, dims=(2,))
        elif self.roll_negative_heads:
            q2 = torch.roll(q2, shifts=1, dims=(2,))
            k2 = torch.roll(k2, shifts=1, dims=(2,))

        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        ).type_as(q)
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        ).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        self.update_cache_for_logging("lambda", lambda_full)

        if self.save_attention_weights:
            attn1, attn1_scores = manual_attention(
                q1.transpose(1, 2),
                k1.transpose(1, 2),
                v.transpose(1, 2),
                causal=True,
            )
            attn1 = attn1.transpose(1, 2)
            attn2, attn2_scores = manual_attention(
                q2.transpose(1, 2),
                k2.transpose(1, 2),
                v.transpose(1, 2),
                causal=True,
            )
            attn2 = attn2.transpose(1, 2)
            if False and self.block_number == 0:
                reference_attn1 = flash_attn_func(
                    q1,
                    k1,
                    v,
                    causal=True,
                )
                reference_attn2 = flash_attn_func(
                    q2,
                    k2,
                    v,
                    causal=True,
                )
                assert torch.allclose(
                    attn1, reference_attn1, atol=1e-3
                ), f"Manual attn1 does not match reference attn1: {attn1-reference_attn1}"
                assert torch.allclose(
                    attn2, reference_attn2, atol=1e-3
                ), f"Manual attn2 does not match reference attn2"

            differential_scores = attn1_scores - lambda_full * attn2_scores
            self.attention_weights = differential_scores
        else:
            attn1 = flash_attn_func(
                q1,
                k1,
                v,
                causal=True,
            )
            attn2 = flash_attn_func(
                q2,
                k2,
                v,
                causal=True,
            )

        attn = attn1 - lambda_full * attn2

        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.reshape(bsz, tgt_len, self.num_positive_heads * self.v_dim)

        attn = self.out_proj(attn)
        return attn

    def log_light(self):
        return {
            "lambda": self.logging_cache["lambda"],
        }
