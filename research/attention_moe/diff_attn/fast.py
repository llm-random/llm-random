import math
import torch
from torch import nn

from lizrd.core.misc import Linear

from .kernel.rotary import apply_rotary_emb
from flash_attn import flash_attn_func
from flash_attn.layers.rotary import RotaryEmbedding

try:
    from apex.normalization import FusedRMSNorm as RMSNorm
except ModuleNotFoundError:
    print("No fused RMSNorm")
    from .rms_norm import RMSNorm


def init_method(tensor, **kwargs):
    nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class Lowrank(nn.Module):
    def __init__(self, outer_dim, inner_dim, init_type, init_scale, output_dim=None):
        super().__init__()
        self.inner_dim = inner_dim
        self.w1 = Linear(
            outer_dim, inner_dim, bias=False, init_type=init_type, init_scale=init_scale
        )
        self.w2 = Linear(
            inner_dim,
            output_dim or outer_dim,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )

    def forward(self, x):
        return self.w2(self.w1(x))


class MultiheadFlashDiff1(nn.Module):
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
        adapter_type="none",
    ):
        super().__init__()
        # self.args = args
        self.embed_dim = embed_dim
        # num_heads set to half of Transformer's #heads
        self.num_heads = num_heads  # // args.model_parallel_size
        # self.num_kv_heads = (
        #     args.decoder_kv_attention_heads // args.model_parallel_size
        #     if args.decoder_kv_attention_heads is not None
        #     else num_heads // args.model_parallel_size
        # )

        assert (int(roll_negative_heads) + int(flip_negative_heads)) <= 1

        self.num_kv_heads = num_kv_heads or num_heads
        self.n_rep = self.num_heads // self.num_kv_heads

        self.head_dim = embed_dim // num_heads // 2

        self.adapter_type = adapter_type
        self.lowrank_inner_dim = lowrank_inner_dim
        if self.adapter_type == "lora" and self.lowrank_inner_dim > 0:
            self.lowrank_q = Lowrank(
                embed_dim, self.lowrank_inner_dim, init_type, init_scale
            )
            self.lowrank_k = Lowrank(
                embed_dim,
                self.lowrank_inner_dim,
                init_type,
                init_scale,
                output_dim=2 * self.head_dim * self.num_kv_heads,
            )

        self.scaling = self.head_dim**-0.5

        self.q_proj = Linear(
            embed_dim,
            embed_dim,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.k_proj = Linear(
            embed_dim,
            self.head_dim * 2 * self.num_kv_heads,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.v_proj = Linear(
            embed_dim,
            self.head_dim * 2 * self.num_kv_heads,
            bias=False,
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
        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                base=10000.0,
                interleaved=True,
            )
            self.rotary_emb._update_cos_sin_cache(self.seq_len, dtype=torch.float32)
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

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(
        self,
        x,
        rel_pos=None,
        attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        if self.lambda_init is None:
            self.lambda_init = lambda_init_fn(self.block_number)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.adapter_type == "lora":
            # self.lowrank_inner_dim > 0:
            q_negative = (q + self.lowrank_q(x)).view(
                bsz, tgt_len, self.num_heads, 2 * self.head_dim
            )
            k_negative = (k + self.lowrank_k(x)).view(
                bsz, src_len, self.num_kv_heads, 2 * self.head_dim
            )
            q = q.view(bsz, tgt_len, self.num_heads, 2 * self.head_dim)
            k = k.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)
            v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)
        elif self.adapter_type == "additive":
            q = q.view(bsz, tgt_len, self.num_heads, 2 * self.head_dim)
            k = k.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)
            v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)
            k_negative = k.clone()
            q_negative = q + self.q_delta
        elif self.adapter_type == "multiplicative":
            q = q.view(bsz, tgt_len, self.num_heads, 2 * self.head_dim)
            k = k.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)
            v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)
            k_negative = k.clone()
            q_negative = q * self.q_delta
        elif self.adapter_type == "identity":
            q = q.view(bsz, tgt_len, self.num_heads, 2 * self.head_dim)
            k = k.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)
            v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)
            q_negative = q.clone()
            k_negative = k.clone()
        elif self.adapter_type == "none":
            q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
            k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
            v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)

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
            if self.lowrank_inner_dim > 0:
                q_negative = apply_rotary_emb(
                    q_negative.to(dtype=torch.float32), *rel_pos, interleaved=True
                ).to(x)
                k_negative = apply_rotary_emb(
                    k_negative.to(dtype=torch.float32), *rel_pos, interleaved=True
                ).to(x)

        # offset = src_len - tgt_len
        # effective_head_dim = self.head_dim
        # if self.lowrank_inner_dim > 0:
        #     effective_head_dim *= 2

        if self.adapter_type != "none":
            q1 = q
            q2 = q_negative
            k1 = k
            k2 = k_negative
            if self.num_kv_heads != self.num_heads:
                k1 = k1.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
                k2 = k2.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
                assert (
                    k1.shape == k2.shape == q1.shape == q2.shape
                ), f"Shapes don't match: {k1.shape}, {k2.shape}, {q1.shape}, {q2.shape}"
        else:
            q = q.reshape(bsz, tgt_len, self.num_heads, 2, self.head_dim)
            k = k.reshape(bsz, src_len, self.num_kv_heads, 2, self.head_dim)
            q1, q2 = q[:, :, :, 0], q[:, :, :, 1]
            k1, k2 = k[:, :, :, 0], k[:, :, :, 1]

        if self.flip_negative_heads:
            q2 = torch.flip(q2, dims=(2,))
            k2 = torch.flip(k2, dims=(2,))
        elif self.roll_negative_heads:
            q2 = torch.roll(q2, shifts=1, dims=(2,))
            k2 = torch.roll(k2, shifts=1, dims=(2,))

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

        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        ).type_as(q)
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        ).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn = attn1 - lambda_full * attn2

        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)

        attn = self.out_proj(attn)
        return attn


class VanillaFlashDiff1(nn.Module):
    """
    (Recommended)
    DiffAttn implemented with FlashAttention, for packages that support different qk/v dimensions
    e.g., our customized-flash-attention (https://aka.ms/flash-diff) and xformers (https://github.com/facebookresearch/xformers)
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        use_rope,
        seq_len,
        init_type,
        init_scale,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # num_heads set to half of Transformer's #heads
        self.num_heads = num_heads
        self.num_kv_heads = self.num_heads
        self.n_rep = self.num_heads // self.num_kv_heads

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.seq_len = seq_len
        self.use_rope = use_rope
        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim, base=10000.0, interleaved=True
            )
            self.rotary_emb._update_cos_sin_cache(self.seq_len, dtype=torch.float32)

        self.q_proj = Linear(
            embed_dim, embed_dim, bias=False, init_type=init_type, init_scale=init_scale
        )
        self.k_proj = Linear(
            embed_dim, embed_dim, bias=False, init_type=init_type, init_scale=init_scale
        )
        self.v_proj = Linear(
            embed_dim, embed_dim, bias=False, init_type=init_type, init_scale=init_scale
        )
        self.out_proj = Linear(
            embed_dim, embed_dim, bias=False, init_type=init_type, init_scale=init_scale
        )
        self.lambda_init = None

    def forward(
        self,
        x,
        # rel_pos=None,
        # attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        if self.lambda_init is None:
            self.lambda_init = lambda_init_fn(self.block_number)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, self.head_dim)

        if self.use_rope:
            assert self.rotary_emb._cos_cached.dtype == torch.float32
            rel_pos = (
                self.rotary_emb._cos_cached.to(x.device),
                self.rotary_emb._sin_cached.to(x.device),
            )
            q = apply_rotary_emb(q.to(dtype=torch.float32), *rel_pos, interleaved=True)
            k = apply_rotary_emb(k.to(dtype=torch.float32), *rel_pos, interleaved=True)

        # offset = src_len - tgt_len
        q = q.reshape(bsz, tgt_len, self.num_heads, self.head_dim)
        k = k.reshape(bsz, src_len, self.num_kv_heads, self.head_dim)
        attn = flash_attn_func(
            q.to(dtype=torch.bfloat16),
            k.to(dtype=torch.bfloat16),
            v.to(dtype=torch.bfloat16),
            causal=True,
        )

        attn = attn.reshape(bsz, tgt_len, self.num_heads * self.head_dim).to(x)

        attn = self.out_proj(attn)
        return attn
