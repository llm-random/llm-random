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

try:
    from flex_head_fa import flash_attn_func
    from flex_head_fa.layers.rotary import RotaryEmbedding
except ModuleNotFoundError:
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
        dmodel,
        # depth,
        n_heads,
        use_rope,
        seq_len,
        lowrank_inner_dim,
        flip_negative_heads,
        roll_negative_heads,
        init_type,
        init_scale,
        lowrank_scaling,
        lowrank_bias,
        double_kv_cache,
        n_kv_heads=None,
        adapter_type: str = "lora",
        lowrank_dtype=None,
        use_qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        # self.args = args
        self.dmodel = dmodel
        self.save_attention_weights = False
        self.attention_weights = None
        self.n_positive_heads = n_heads if double_kv_cache else n_heads // 2
        assert (int(roll_negative_heads) + int(flip_negative_heads)) <= 1

        self.n_positive_kv_heads = (n_kv_heads or n_heads) if double_kv_cache else (n_kv_heads or n_heads) // 2
        self.n_rep = self.n_positive_heads // self.n_positive_kv_heads
        self.adapter_type = adapter_type

        self.dhead = dmodel // n_heads

        q_proj_out_dim = self.dhead * self.n_positive_heads
        assert self.adapter_type == "lora"
        k_proj_out_dim = self.dhead * self.n_positive_kv_heads

        self.adapter_type = adapter_type
        self.lowrank_inner_dim = lowrank_inner_dim
        if self.adapter_type == "lora" and self.lowrank_inner_dim > 0:
            self.lowrank_q = Lowrank(
                dmodel,
                self.lowrank_inner_dim,
                init_type,
                init_scale,
                lowrank_scaling=lowrank_scaling,
                lowrank_bias=lowrank_bias,
                output_dim=q_proj_out_dim,
                dtype=lowrank_dtype,
            )
            self.lowrank_k = Lowrank(
                dmodel,
                self.lowrank_inner_dim,
                init_type,
                init_scale,
                lowrank_scaling=lowrank_scaling,
                lowrank_bias=lowrank_bias,
                output_dim=k_proj_out_dim,
                dtype=lowrank_dtype,
            )
        elif self.adapter_type == "identity":
            pass
        else:
            raise ValueError(f"Adapter type {self.adapter_type} not supported")

        self.q_proj = Linear(
            dmodel,
            q_proj_out_dim,
            bias=True,
            init_type=init_type,
            init_scale=init_scale,
        )

        self.k_proj = Linear(
            dmodel,
            k_proj_out_dim,
            bias=True,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.v_dim = self.dmodel // self.n_positive_heads
        self.v_proj = Linear(
            dmodel,
            self.v_dim * self.n_positive_kv_heads,
            bias=True,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.out_proj = Linear(
            dmodel, dmodel, bias=False, init_type=init_type, init_scale=init_scale
        )

        self.lambda_init = None
        self.use_rope = use_rope
        self.seq_len = seq_len
        self.flip_negative_heads = flip_negative_heads
        self.roll_negative_heads = roll_negative_heads

        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.dhead, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.dhead, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.dhead, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.dhead, dtype=torch.float32).normal_(mean=0, std=0.1)
        )

        self.subln = RMSNorm(self.v_dim, eps=rms_norm_eps, elementwise_affine=True)
        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(
                self.dhead, eps=rms_norm_eps, elementwise_affine=True
            )
            self.k_norm = RMSNorm(
                self.dhead, eps=rms_norm_eps, elementwise_affine=True
            )

        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(
                self.dhead,
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
        bsz, _, _ = x.size()

        if self.lambda_init is None:
            self.lambda_init = lambda_init_fn(self.block_number + 1)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.adapter_type == "lora":
            # self.lowrank_inner_dim > 0:
            q_negative = (q + self.lowrank_q(x)).view(
                bsz, self.seq_len, self.n_positive_heads, self.dhead
            )
            k_negative = (k + self.lowrank_k(x)).view(
                bsz, self.seq_len, self.n_positive_kv_heads, self.dhead
            )
            q = q.view(bsz, self.seq_len, self.n_positive_heads, self.dhead)
            k = k.view(bsz, self.seq_len, self.n_positive_kv_heads, self.dhead)
            v = v.view(bsz, self.seq_len, self.n_positive_kv_heads, self.v_dim)
        elif self.adapter_type == "identity":
            q_negative = q.view(
                bsz, self.seq_len, self.n_positive_heads, self.dhead
            )
            k_negative = k.view(
                bsz, self.seq_len, self.n_positive_kv_heads, self.dhead
            )
            q = q.view(bsz, self.seq_len, self.n_positive_heads, self.dhead)
            k = k.view(bsz, self.seq_len, self.n_positive_kv_heads, self.dhead)
            v = v.view(bsz, self.seq_len, self.n_positive_kv_heads, self.v_dim)
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
            if self.n_positive_kv_heads != self.n_positive_heads:
                k1 = k1.repeat_interleave(self.n_rep, dim=2)
                k2 = k2.repeat_interleave(self.n_rep, dim=2)
                v = v.repeat_interleave(self.n_rep, dim=2)
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
        attn = attn.reshape(bsz, self.seq_len, self.n_positive_heads * self.v_dim)

        attn = self.out_proj(attn)
        return attn

    def log_light(self):
        return {
            "lambda": self.logging_cache["lambda"],
        }


class VanillaAttention(LoggingLayer):
    """
    (Recommended)
    DiffAttn implemented with FlashAttention, for packages that support different qk/v dimensions
    e.g., our customized-flash-attention (https://aka.ms/flash-diff) and xformers (https://github.com/facebookresearch/xformers)
    """

    def __init__(
        self,
        # args,
        dmodel,
        # depth,
        n_heads,
        use_rope,
        seq_len,
        init_type,
        init_scale,
        n_kv_heads=None,
        use_qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        # self.args = args
        self.dmodel = dmodel
        self.save_attention_weights = False
        self.attention_weights = None

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.dhead = dmodel // n_heads

        v_proj_out_dim = k_proj_out_dim = self.dhead * self.n_kv_heads

        self.q_proj = Linear(
            dmodel,
            dmodel,
            bias=True,
            init_type=init_type,
            init_scale=init_scale,
        )

        self.k_proj = Linear(
            dmodel,
            k_proj_out_dim,
            bias=True,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.v_proj = Linear(
            dmodel,
            v_proj_out_dim,
            bias=True,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.out_proj = Linear(
            dmodel, dmodel, bias=False, init_type=init_type, init_scale=init_scale
        )

        self.use_rope = use_rope
        self.seq_len = seq_len

        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(
                self.dhead, eps=rms_norm_eps, elementwise_affine=True
            )
            self.k_norm = RMSNorm(
                self.dhead, eps=rms_norm_eps, elementwise_affine=True
            )

        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(
                self.dhead,
                base=rope_theta,
                interleaved=True,
            )
            self.rotary_emb._update_cos_sin_cache(self.seq_len, dtype=torch.float32)

        self.attention_checked = False

    def forward(
        self,
        x,
        rel_pos=None,
        attn_mask=None,
    ):
        bsz, _, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, self.seq_len, self.n_heads, self.dhead)
        k = k.view(bsz, self.seq_len, self.n_kv_heads, self.dhead)
        v = v.view(bsz, self.seq_len, self.n_kv_heads, self.dhead)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

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

        if self.n_heads != self.n_kv_heads:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)
            assert k.shape == q.shape, f"Shapes don't match: {k.shape}, {q.shape}"

        if self.save_attention_weights:
            attn, attn_scores = manual_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                causal=True,
            )
            attn = attn.transpose(1, 2)
            if False and not self.attention_checked:
                reference_attn = flash_attn_func(
                    q,
                    k,
                    v,
                    causal=True,
                )
                assert torch.allclose(
                    attn, reference_attn, atol=1e-2
                ), f"Manual attn1 does not match reference attn1: {attn - reference_attn}"
                self.attention_checked = True

            self.attention_weights = attn_scores
        else:
            attn = flash_attn_func(
                q,
                k,
                v,
                causal=True,
            )

        attn = attn.reshape(bsz, self.seq_len, self.dmodel)

        attn = self.out_proj(attn)
        return attn


class GroupedDifferentialAttention(LoggingLayer):
    """
    (Recommended)
    DiffAttn implemented with FlashAttention, for packages that support different qk/v dimensions
    e.g., our customized-flash-attention (https://aka.ms/flash-diff) and xformers (https://github.com/facebookresearch/xformers)
    """

    def __init__(
        self,
        dmodel,
        n_heads,
        use_rope,
        seq_len,
        init_type,
        init_scale,
        # repeat_or_interleave: str,
        negative_heads_permutation: str,
        adapter_type: str,
        n_kv_heads=None,
        n_negative_heads=None,
        use_qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        # self.args = args
        self.adapter_type = adapter_type or "none"
        self.dmodel = dmodel
        self.save_attention_weights = False
        self.attention_weights = None
        self.n_positive_heads = n_heads // 2
        # self.repeat_or_interleave = repeat_or_interleave
        self.negative_heads_permutation = negative_heads_permutation

        self.n_positive_kv_heads = (n_kv_heads or n_heads) // 2
        assert n_negative_heads <= self.n_positive_kv_heads
        self.n_negative_heads = n_negative_heads
        self.n_rep_kv = self.n_positive_heads // self.n_positive_kv_heads
        self.n_rep_negative = self.n_positive_heads // self.n_negative_heads

        self.dhead = dmodel // n_heads if adapter_type != "identity" else 2 * dmodel // n_heads

        plus_q_proj_out_dim = self.dhead * self.n_positive_heads
        plus_k_proj_out_dim = self.dhead * self.n_positive_kv_heads
        minus_qk_proj_out_dim = self.dhead * n_negative_heads if adapter_type != "identity" else 0

        self.q_proj = Linear(
            dmodel,
            plus_q_proj_out_dim + minus_qk_proj_out_dim,
            bias=True,
            init_type=init_type,
            init_scale=init_scale,
        )

        self.k_proj = Linear(
            dmodel,
            plus_k_proj_out_dim + minus_qk_proj_out_dim,
            bias=True,
            init_type=init_type,
            init_scale=init_scale,
        )

        self.v_dim = self.dmodel // self.n_positive_heads
        self.v_proj = Linear(
            dmodel,
            self.v_dim * self.n_positive_kv_heads,
            bias=True,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.out_proj = Linear(
            dmodel, dmodel, bias=False, init_type=init_type, init_scale=init_scale
        )

        self.lambda_init = None
        self.use_rope = use_rope
        self.seq_len = seq_len

        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.dhead, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.dhead, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.dhead, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.dhead, dtype=torch.float32).normal_(mean=0, std=0.1)
        )

        self.subln = RMSNorm(self.v_dim, eps=rms_norm_eps, elementwise_affine=True)
        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(
                self.dhead, eps=rms_norm_eps, elementwise_affine=True
            )
            self.k_norm = RMSNorm(
                self.dhead, eps=rms_norm_eps, elementwise_affine=True
            )

        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(
                self.dhead,
                base=rope_theta,
                interleaved=True,
            )
            self.rotary_emb._update_cos_sin_cache(self.seq_len, dtype=torch.float32)
        self.attention_checked = False

    def forward(
        self,
        x,
        rel_pos=None,
        attn_mask=None,
    ):
        bsz, _, _ = x.size()

        if self.lambda_init is None:
            self.lambda_init = lambda_init_fn(self.block_number + 1)

        q = self.q_proj(x).view(
            bsz,
            self.seq_len,
            self.n_positive_heads + self.n_negative_heads if self.adapter_type != "identity" else self.n_positive_heads,
            self.dhead,
        )
        k = self.k_proj(x).view(
            bsz,
            self.seq_len,
            self.n_positive_kv_heads + self.n_negative_heads if self.adapter_type != "identity" else self.n_positive_kv_heads,
            self.dhead,
        )

        if self.adapter_type != "identity":
            q, q_negative = (
                q[:, :, : self.n_positive_heads],
                q[:, :, self.n_positive_heads:],
            )
            k, k_negative = (
                k[:, :, : self.n_positive_kv_heads],
                k[:, :, self.n_positive_kv_heads:],
            )
        else:
            q_negative = q[:, :, :self.n_negative_heads]
            k_negative = k[:, :, :self.n_negative_heads]
        v = self.v_proj(x).view(bsz, self.seq_len, self.n_positive_kv_heads, self.v_dim)

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

        q1 = q
        q2 = q_negative
        k1 = k
        k2 = k_negative

        # WARNING niekompatybilne z GQA
        # if (
        #     self.n_positive_kv_heads != self.n_positive_heads
        #     or self.n_negative_heads != self.n_positive_heads
        # ):
        #     if self.repeat_or_interleave == "interleave":
        #         q2 = q2.repeat_interleave(self.n_rep_negative, dim=2)
        #         k1 = k1.repeat_interleave(self.n_rep_kv, dim=2)
        #         v = v.repeat_interleave(self.n_rep_kv, dim=2)
        #         k2 = k2.repeat_interleave(self.n_rep_negative, dim=2)
        #     else:
        #         q2 = q2.repeat(1, 1, self.n_rep_negative, 1)
        #         k1 = k1.repeat(1, 1, self.n_rep_kv, 1)
        #         v = v.repeat(1, 1, self.n_rep_kv, 1)
        #         k2 = k2.repeat(1, 1, self.n_rep_negative, 1)
        #     assert (
        #         k1.shape == k2.shape == q1.shape == q2.shape
        #     ), f"Shapes don't match: {k1.shape}, {k2.shape}, {q1.shape}, {q2.shape}"

        if self.negative_heads_permutation == "repeat":
            q2 = q2.repeat(1, 1, self.n_positive_heads // self.n_negative_heads, 1)
            k2 = k2.repeat(1, 1, self.n_positive_heads // self.n_negative_heads, 1)
        elif self.negative_heads_permutation == "interleave":
            q2 = q2.repeat_interleave(self.n_positive_heads // self.n_negative_heads, dim=2)
            k2 = k2.repeat_interleave(self.n_positive_heads // self.n_negative_heads, dim=2)
        elif self.negative_heads_permutation == "flip_repeat":
            q2 = torch.flip(q2, dims=(2,))
            q2 = q2.repeat(1, 1, self.n_positive_heads // self.n_negative_heads, 1)
            k2 = torch.flip(k2, dims=(2,))
            k2 = k2.repeat(1, 1, self.n_positive_heads // self.n_negative_heads, 1)
        elif self.negative_heads_permutation == "flip_interleave":
            q2 = torch.flip(q2, dims=(2,))
            q2 = q2.repeat_interleave(self.n_positive_heads // self.n_negative_heads, dim=2)
            k2 = torch.flip(k2, dims=(2,))
            k2 = k2.repeat_interleave(self.n_positive_heads // self.n_negative_heads, dim=2)
        elif self.negative_heads_permutation == "roll_repeat":
            q2 = torch.roll(q2, shifts=1, dims=(2,))
            q2 = q2.repeat(1, 1, self.n_positive_heads // self.n_negative_heads, 1)
            k2 = torch.roll(k2, shifts=1, dims=(2,))
            k2 = k2.repeat(1, 1, self.n_positive_heads // self.n_negative_heads, 1)
        elif self.negative_heads_permutation == "repeat_roll":
            q2 = q2.repeat(1, 1, self.n_positive_heads // self.n_negative_heads, 1)
            q2 = torch.roll(q2, shifts=1, dims=(2,))
            k2 = k2.repeat(1, 1, self.n_positive_heads // self.n_negative_heads, 1)
            k2 = torch.roll(k2, shifts=1, dims=(2,))
        elif self.negative_heads_permutation == "roll_interleave":
            q2 = torch.roll(q2, shifts=1, dims=(2,))
            q2 = q2.repeat_interleave(self.n_positive_heads // self.n_negative_heads, dim=2)
            k2 = torch.roll(k2, shifts=1, dims=(2,))
            k2 = k2.repeat_interleave(self.n_positive_heads // self.n_negative_heads, dim=2)

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
            if False and self.attention_checked == False:
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
                    attn1, reference_attn1, atol=1e-2
                ), f"Manual attn1 does not match reference attn1: {attn1-reference_attn1}"
                assert torch.allclose(
                    attn2, reference_attn2, atol=1e-2
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
        attn = attn.reshape(bsz, self.seq_len, self.n_positive_heads * self.v_dim)

        attn = self.out_proj(attn)
        return attn

    def log_light(self):
        return {
            "lambda": self.logging_cache["lambda"],
        }


class DifferentialAttention(LoggingLayer):
    """
    (Recommended)
    DiffAttn implemented with FlashAttention, for packages that support different qk/v dimensions
    e.g., our customized-flash-attention (https://aka.ms/flash-diff) and xformers (https://github.com/facebookresearch/xformers)
    """

    def __init__(
        self,
        dmodel,
        n_heads,
        use_rope,
        seq_len,
        init_type,
        init_scale,
        n_kv_heads=None,
        adapter_type: str = "lora",
        use_qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        # self.args = args
        self.dmodel = dmodel
        self.save_attention_weights = False
        self.attention_weights = None
        self.n_positive_heads = n_heads // 2

        self.n_positive_kv_heads = (n_kv_heads or n_heads) // 2
        self.n_rep = self.n_positive_heads // self.n_positive_kv_heads
        self.adapter_type = adapter_type

        self.dhead = dmodel // n_heads

        q_proj_out_dim = self.dhead * self.n_positive_heads
        k_proj_out_dim = self.dhead * self.n_positive_kv_heads

        self.q_proj = Linear(
            dmodel,
            2 * q_proj_out_dim,
            bias=True,
            init_type=init_type,
            init_scale=init_scale,
        )

        self.k_proj = Linear(
            dmodel,
            2 * k_proj_out_dim,
            bias=True,
            init_type=init_type,
            init_scale=init_scale,
        )

        self.v_dim = self.dmodel // self.n_positive_heads
        self.v_proj = Linear(
            dmodel,
            self.v_dim * self.n_positive_kv_heads,
            bias=True,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.out_proj = Linear(
            dmodel, dmodel, bias=False, init_type=init_type, init_scale=init_scale
        )

        self.lambda_init = None
        self.use_rope = use_rope
        self.seq_len = seq_len

        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.dhead, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.dhead, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.dhead, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.dhead, dtype=torch.float32).normal_(mean=0, std=0.1)
        )

        self.subln = RMSNorm(self.v_dim, eps=rms_norm_eps, elementwise_affine=True)
        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(
                self.dhead, eps=rms_norm_eps, elementwise_affine=True
            )
            self.k_norm = RMSNorm(
                self.dhead, eps=rms_norm_eps, elementwise_affine=True
            )

        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(
                self.dhead,
                base=rope_theta,
                interleaved=True,
            )
            self.rotary_emb._update_cos_sin_cache(self.seq_len, dtype=torch.float32)
        self.attention_checked = False

    def forward(
        self,
        x,
        rel_pos=None,
        attn_mask=None,
    ):
        bsz, _, _ = x.size()

        if self.lambda_init is None:
            self.lambda_init = lambda_init_fn(self.block_number + 1)

        q = self.q_proj(x).view(
            bsz, self.seq_len, self.n_positive_heads, 2 * self.dhead
        )
        q, q_negative = q.chunk(2, dim=-1)
        k = self.k_proj(x).view(
            bsz, self.seq_len, self.n_positive_kv_heads, 2 * self.dhead
        )
        k, k_negative = k.chunk(2, dim=-1)
        v = self.v_proj(x).view(bsz, self.seq_len, self.n_positive_kv_heads, self.v_dim)

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

        # if self.adapter_type != "none":
        q1 = q
        q2 = q_negative
        k1 = k
        k2 = k_negative
        if self.n_positive_kv_heads != self.n_positive_heads:
            k1 = k1.repeat_interleave(self.n_rep, dim=2)
            k2 = k2.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)
            assert (
                k1.shape == k2.shape == q1.shape == q2.shape
            ), f"Shapes don't match: {k1.shape}, {k2.shape}, {q1.shape}, {q2.shape}"

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
            if False and self.attention_checked == False:
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
                    attn1, reference_attn1, atol=1e-2
                ), f"Manual attn1 does not match reference attn1: {attn1-reference_attn1}"
                assert torch.allclose(
                    attn2, reference_attn2, atol=1e-2
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
        attn = attn.reshape(bsz, self.seq_len, self.n_positive_heads * self.v_dim)

        attn = self.out_proj(attn)
        return attn

    def log_light(self):
        return {
            "lambda": self.logging_cache["lambda"],
        }
