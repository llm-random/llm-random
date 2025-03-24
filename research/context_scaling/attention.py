from typing import Callable, Optional
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


class VanillaAttention(LoggingLayer):
    def __init__(
        self,
        dmodel,
        n_heads,
        use_rope,
        seq_len,
        init_type,
        init_scale,
        n_kv_heads=None,
        use_qk_norm: bool = False,
        get_qk_norm: Optional[Callable[[int], torch.nn.Module]] = None,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        # self.args = args
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
            assert get_qk_norm is not None
            self.q_norm = get_qk_norm(self.dhead)
            self.k_norm = get_qk_norm(self.dhead)

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
