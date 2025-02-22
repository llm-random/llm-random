import math
# from weakref import ref
import torch
from torch import nn
import torch.nn.functional as F

from lizrd.core.misc import Linear, LoggingLayer

from .kernel.rotary import apply_rotary_emb
from flash_attn import flash_attn_func
from flash_attn.layers.rotary import RotaryEmbedding

try:
    from apex.normalization import FusedRMSNorm as RMSNorm
except ModuleNotFoundError:
    # print("No fused RMSNorm")
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
    def __init__(
        self,
        outer_dim,
        inner_dim,
        init_type,
        init_scale,
        output_dim=None,
        dtype=None,
    ):
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
        self.dtype = dtype

    def forward(self, x):
        if self.dtype is None:
            return self.w2(self.w1(x))
        else:
            original_dtype = x.dtype
            forced_dtype = getattr(torch, self.dtype)
            x = x.to(forced_dtype)
            res = self.w2(self.w1(x))
            return res.to(original_dtype)


def manual_attention(q, key, value, causal=True, enable_gqa=False):
    """Preserves flashattention's interface, but also returns attention weights"""
    # ...# manual implementation of attention
    if not causal:
        raise NotImplementedError
    bsz, nh, seq_len, dhead = q.shape

    if enable_gqa:
        key = key.repeat_interleave(nh // key.shape[1], dim=1)
        value = value.repeat_interleave(nh // value.shape[1], dim=1)

    att = (q @ key.transpose(-2, -1)) * (1.0 / math.sqrt(dhead))
    att = att.masked_fill(
        torch.tril(torch.ones(seq_len, seq_len).to(att)) == 0, float("-inf")
    )
    att = F.softmax(att.to(torch.float32), dim=-1)
    y = att.to(value) @ value  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    return y, att


class MultiheadFlashDiff1(LoggingLayer):
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
        n_negative_heads,
        use_rope,
        seq_len,
        lowrank_inner_dim,
        # flip_negative_heads,
        # roll_negative_heads,
        negative_heads_permutation,
        init_type,
        init_scale,
        n_kv_heads=None,
        adapter_type="none",
        lowrank_dtype=None,
        use_qk_norm: bool = False,
        reuse_positive_k: bool = False,
    ):
        super().__init__()
        # self.args = args
        self.dmodel = dmodel
        self.save_attention_weights = False
        self.attention_weights = None
        # num_heads set to half of Transformer's #heads
        self.n_negative_heads = n_negative_heads or n_heads
        self.negative_heads_permutation = negative_heads_permutation
        self.seq_len = seq_len

        self.n_heads = n_heads

        self.reuse_positive_k = reuse_positive_k
        if self.reuse_positive_k:
            assert adapter_type == "none"

        # WARNING niekompatybilne z GQA!!!

        # TODO inverse GDA??
        # TODO GQA na Q + normalne GDA

        # TODO coś szybszego niż roll o 1?
        # TODO custom flash attention
        # TODO reduced outlier magnitude -> low-bit kernels?
        # TODO sparse attention -> kv-cache compression?

        # TODO scaling/B to 0's in LoRA?
        # TODO Q LoRA + K identity?

        # TODO softmax na końcu??
        # TODO niedzielona lambda?
        # TODO flash attn z ich papera (ale lepsze, bo na 3?)?

        # TODO inżynierka configów pairwise
        # TODO lora in pretraining?
        # TODO Skąd bierze pomysły?
        # TODO fineweb dataset?
        # TODO run exp --yes
        # TODO folder na expy o tej samej nazwie
        # TODO optymalny vocab size -> inne rzeczy dla MFU?


        # self.n_kv_heads = n_kv_heads or n_heads
        # assert (
        #     self.n_kv_heads < self.n_heads or self.n_negative_heads < self.n_heads
        # ), "GQA and GDA cannot be enabled together."
        # self.enable_gqa = self.n_heads != self.n_kv_heads
        # assert (
        #     self.n_heads % self.n_kv_heads == 0
        # ), "The number of heads must be divisible by the number of key-value heads."
        self.dhead = dmodel // n_heads

        self.adapter_type = adapter_type
        self.lowrank_inner_dim = lowrank_inner_dim
        if self.adapter_type == "lora":
            assert (
                self.lowrank_inner_dim > 0
            ), "LoRA requires a positive inner dimension."
            self.lowrank_q = Lowrank(
                self.dmodel,
                self.lowrank_inner_dim,
                init_type,
                init_scale,
                output_dim=self.dhead * self.n_negative_heads,
                dtype=lowrank_dtype,
            )
            self.lowrank_k = Lowrank(
                self.dmodel,
                self.lowrank_inner_dim,
                init_type,
                init_scale,
                # output_dim=self.dhead * min(self.n_negative_heads, self.n_kv_heads),
                output_dim=self.dhead * self.n_negative_heads,
                dtype=lowrank_dtype,
            )
        elif self.adapter_type == "dora":
            assert (
                self.lowrank_inner_dim > 0
            ), "DoRA requires a positive inner dimension."
            self.lowrank_q = Lowrank(
                self.dmodel,
                self.lowrank_inner_dim,
                init_type,
                init_scale,
                output_dim=self.dhead * self.n_negative_heads,
                dtype=lowrank_dtype,
            )
            self.lowrank_k = Lowrank(
                self.dmodel,
                self.lowrank_inner_dim,
                init_type,
                init_scale,
                # output_dim=self.dhead * min(self.n_negative_heads, self.n_kv_heads),
                output_dim=self.dhead * self.n_negative_heads,
                dtype=lowrank_dtype,
            )
            self.magnitude_q = nn.Parameter(
                self.q_proj.weight.norm(p=2, dim=0, keepdim=True)
            )
            self.magnitude_k = nn.Parameter(
                self.k_proj.weight.norm(p=2, dim=0, keepdim=True)
            )
        elif self.adapter_type == "additive":
            self.k_delta = nn.Parameter(
                torch.zeros(
                    # self.dhead * min(self.n_negative_heads, self.n_kv_heads),
                    self.dhead * self.n_negative_heads,
                    dtype=torch.float32,
                ).normal_(mean=0, std=0.1)
            )
            self.q_delta = nn.Parameter(
                torch.zeros(
                    self.dhead * self.n_negative_heads, dtype=torch.float32
                ).normal_(mean=0, std=0.1)
            )
        elif self.adapter_type == "multiplicative":
            self.k_delta = nn.Parameter(
                torch.zeros(
                    # self.dhead * min(self.n_negative_heads, self.n_kv_heads),
                    self.dhead * self.n_negative_heads,
                    dtype=torch.float32,
                ).normal_(mean=1, std=0.1)
            )
            self.q_delta = nn.Parameter(
                torch.zeros(
                    self.dhead * self.n_negative_heads, dtype=torch.float32
                ).normal_(mean=1, std=0.1)
            )
        elif self.adapter_type == "multiadd":
            self.k_delta_mult = nn.Parameter(
                torch.zeros(
                    # self.dhead * min(self.n_negative_heads, self.n_kv_heads),
                    self.dhead * self.n_negative_heads,
                    dtype=torch.float32,
                ).normal_(mean=1, std=0.1)
            )
            self.q_delta_mult = nn.Parameter(
                torch.zeros(
                    self.dhead * self.n_negative_heads, dtype=torch.float32
                ).normal_(mean=1, std=0.1)
            )
            self.k_delta_add = nn.Parameter(
                torch.zeros(
                    # self.dhead * min(self.n_negative_heads, self.n_kv_heads),
                    self.dhead * self.n_negative_heads,
                    dtype=torch.float32,
                ).normal_(mean=0, std=0.1)
            )
            self.q_delta_add = nn.Parameter(
                torch.zeros(
                    self.dhead * self.n_negative_heads, dtype=torch.float32
                ).normal_(mean=0, std=0.1)
            )
        elif self.adapter_type == "none" or self.adapter_type == "identity":
            pass
        else:
            raise NotImplementedError

        self.scaling = self.dhead ** -0.5

        self.q_proj = Linear(
            self.dmodel,
            self.dmodel if adapter_type != "none" else self.dhead // 2 * (self.n_heads + self.n_negative_heads),
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        # k_proj_dim = self.dhead * self.n_kv_heads
        # k_proj_dim = self.dmodel
        # if self.reuse_positive_k:
        #     k_proj_dim //= 2
        self.k_proj = Linear(
            self.dmodel,
            # k_proj_dim,
            self.dmodel if adapter_type != "none" else self.dhead // 2 * (self.n_heads + self.n_negative_heads),
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
            )
        self.v_proj = Linear(
            self.dmodel,
            # self.dhead * self.n_kv_heads,
            self.dmodel,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
            )
        self.out_proj = Linear(
            self.dmodel,
            self.dmodel,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )

        self.lambda_init = None
        self.use_rope = use_rope
        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(
                self.dhead if self.adapter_type != "none" else self.dhead // 2,
                base=500000.0,
                interleaved=True,
            )
            self.rotary_emb._update_cos_sin_cache(self.seq_len, dtype=torch.float32)
        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.dhead if self.adapter_type != "none" else self.dhead // 2,
                        dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.dhead if self.adapter_type != "none" else self.dhead // 2,
                        dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.dhead if self.adapter_type != "none" else self.dhead // 2,
                        dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.dhead if self.adapter_type != "none" else self.dhead // 2,
                        dtype=torch.float32).normal_(mean=0, std=0.1)
        )

        # self.n_kv_heads = min(self.n_negative_heads, self.n_kv_heads)

        self.subln = RMSNorm(self.dhead, eps=1e-5, elementwise_affine=True)

        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            if self.adapter_type != "none":
                qk_norm_dim = self.dhead
            else:
                qk_norm_dim = self.dhead // 2
            self.q_norm = RMSNorm(qk_norm_dim, eps=1e-5, elementwise_affine=True)
            self.k_norm = RMSNorm(qk_norm_dim, eps=1e-5, elementwise_affine=True)

    def forward(
        self,
        x,
        rel_pos=None,
        attn_mask=None,
    ):
        global q_negative
        global k_negative
        global q_trunc
        global k_trunc
        bsz, _, _ = x.size()

        if self.lambda_init is None:
            self.lambda_init = lambda_init_fn(self.block_number)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q_trunc = q[:, :, :self.n_negative_heads * self.dhead]
        k_trunc = k[:, :, :self.n_negative_heads * self.dhead]

        if self.adapter_type == "lora":
            q_negative = (q_trunc + self.lowrank_q(x)).view(
                bsz, self.seq_len, self.n_negative_heads, self.dhead
            )
            k_negative = (k_trunc + self.lowrank_k(x)).view(
                bsz,
                self.seq_len,
                # self.n_kv_heads,
                self.n_negative_heads,
                self.dhead,
            )
        elif self.adapter_type == "dora":
            lora_q = self.lowrank_q.w2.weight @ self.lowrank_q.w1.weight
            numerator_q = self.q_proj.weight[:, :self.n_negative_heads * self.dhead] + lora_q
            denominator_q = numerator_q.norm(p=2, dim=0, keepdim=True)
            directional_component_q = numerator_q / denominator_q
            q_negative_weights = self.magnitude_q * directional_component_q
            q_negative = (
                q_negative_weights @ x.view(bsz * self.seq_len, self.dmodel).T
            ).view(bsz, self.seq_len, self.n_negative_heads, self.dhead)

            lora_k = self.lowrank_k.w2.weight @ self.lowrank_k.w1.weight
            numerator_k = self.k_proj.weight[:, :self.n_negative_heads * self.dhead] + lora_k
            denominator_k = numerator_k.norm(p=2, dim=0, keepdim=True)
            directional_component_k = numerator_k / denominator_k
            k_negative_weights = self.magnitude_k * directional_component_k
            k_negative = (
                k_negative_weights @ x.view(bsz * self.seq_len, self.dmodel).T
            ).view(bsz, self.seq_len, self.n_negative_heads, self.dhead)
        elif self.adapter_type == "additive":
            q_negative = (q_trunc + self.q_delta.repeat(bsz, self.seq_len, 1)).view(
                bsz, self.seq_len, self.n_negative_heads, self.dhead
            )
            k_negative = (k_trunc + self.k_delta.repeat(bsz, self.seq_len, 1)).view(
                # bsz, self.seq_len, self.n_kv_heads, self.dhead
                bsz,
                self.seq_len,
                self.n_negative_heads,
                self.dhead,
            )
        elif self.adapter_type == "multiplicative":
            q_negative = (q_trunc * self.q_delta.repeat(bsz, self.seq_len, 1)).view(
                bsz, self.seq_len, self.n_negative_heads, self.dhead
            )
            k_negative = (k_trunc * self.k_delta.repeat(bsz, self.seq_len, 1)).view(
                # bsz, self.seq_len, self.n_kv_heads, self.dhead
                bsz,
                self.seq_len,
                self.n_negative_heads,
                self.dhead,
            )
        elif self.adapter_type == "multiadd":
            q_negative = (
                q_trunc * self.q_delta_mult.repeat(bsz, self.seq_len, 1)
                + self.q_delta_add.repeat(bsz, self.seq_len, 1)
            ).view(
                bsz, self.seq_len, self.n_negative_heads, self.dhead
            )
            k_negative = (
                k_trunc * self.k_delta_mult.repeat(bsz, self.seq_len, 1)
                + self.k_delta_add.repeat(bsz, self.seq_len, 1)
            ).view(
                # bsz, self.seq_len, self.n_kv_heads, self.dhead
                bsz,
                self.seq_len,
                self.n_negative_heads,
                self.dhead,
            )
        elif self.adapter_type == "identity":
            q_negative = q_trunc.clone().view(
                bsz, self.seq_len, self.n_negative_heads, self.dhead
            )
            k_negative = k_trunc.clone().view(
                bsz, self.seq_len, self.n_negative_heads, self.dhead
            )
            # q_negative = q.clone().view(bsz, self.seq_len, self.n_negative_heads, 2 * self.dhead)
            # k_negative = k.clone().view(bsz, self.seq_len, self.n_kv_heads, 2 * self.dhead)
        elif self.adapter_type == "none":
            q_negative = q.view(bsz, self.seq_len, self.n_heads + self.n_negative_heads, self.dhead // 2)[:, :, self.n_heads:]
            q = q.view(bsz, self.seq_len, self.n_heads + self.n_negative_heads, self.dhead // 2)[:, :, :self.n_heads]

            # if self.reuse_positive_k:
            #     k = k.view(bsz, src_len, self.num_kv_heads, self.head_dim)

            # k_negative = k.view(bsz, self.seq_len, 2 * self.n_kv_heads, self.dhead // 2)[:, :, :self.n_kv_heads]
            # k = k.view(bsz, self.seq_len, 2 * self.n_kv_heads, self.dhead // 2)[:, :, :self.n_kv_heads]
            k_negative = k.view(bsz, self.seq_len, self.n_heads + self.n_negative_heads, self.dhead // 2)[:, :, self.n_heads:]
            k = k.view(bsz, self.seq_len, self.n_heads + self.n_negative_heads, self.dhead // 2)[:, :, :self.n_heads]

        if self.adapter_type != "none":
            q = q.view(bsz, self.seq_len, self.n_heads, self.dhead)
            k = k.view(bsz, self.seq_len, self.n_heads, self.dhead)
            # k = k.view(bsz, self.seq_len, self.n_kv_heads, self.dhead)

        v = v.view(bsz, self.seq_len, self.n_heads, self.dhead)
        # v = v.view(bsz, self.seq_len, self.n_kv_heads, self.dhead)

        if self.use_qk_norm:
            q = self.q_norm(q)
            q_negative = self.q_norm(q_negative)
            k = self.k_norm(k)
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
            # if self.adapter_type != "none":
            q_negative = apply_rotary_emb(
                q_negative.to(dtype=torch.float32), *rel_pos, interleaved=True
            ).to(x)
            k_negative = apply_rotary_emb(
                k_negative.to(dtype=torch.float32), *rel_pos, interleaved=True
            ).to(x)

        # if self.adapter_type != "none":


        q1 = q
        k1 = k

        q2 = q_negative #.repeat(1, 1, self.n_heads // self.n_negative_heads, 1)
        k2 = k_negative #.repeat(1, 1, self.n_heads // self.n_negative_heads, 1)

        # assert (
        #             k1.shape == k2.shape == q1.shape == q2.shape
        #         ), f"Shapes don't match: {k1.shape}, {k2.shape}, {q1.shape}, {q2.shape}"


            # if self.num_kv_heads != self.num_heads:
            #     k1 = k1.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
            #     k2 = k2.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
            #     assert (
            #         k1.shape == k2.shape == q1.shape == q2.shape
            #     ), f"Shapes don't match: {k1.shape}, {k2.shape}, {q1.shape}, {q2.shape}"
        # else:
        #     q1 = q
        #     q2 = q_negative.repeat(1, 1, self.n_heads // self.n_negative_heads, 1)

            # if self.reuse_positive_k:
            #     k = k.reshape(bsz, src_len, self.num_kv_heads, self.head_dim)
            #     k1 = k
            #     k2 = k.clone()

            # k1 = k
            # k2 = k_negative.repeat(1, 1, self.n_heads // self.n_negative_heads, 1)

            #
            # k1 = k.repeat(1, 1, self.n_heads // self.n_kv_heads, 1)
            # k2 = k_negative.repeat(1, 1, self.n_heads // self.n_kv_heads, 1)

        if self.negative_heads_permutation == "repeat":
            q2 = q2.repeat(1, 1, self.n_heads // self.n_negative_heads, 1)
            k2 = k2.repeat(1, 1, self.n_heads // self.n_negative_heads, 1)
        elif self.negative_heads_permutation == "interleave":
            q2 = q2.repeat_interleave(self.n_heads // self.n_negative_heads, dim=2)
            k2 = k2.repeat_interleave(self.n_heads // self.n_negative_heads, dim=2)
        elif self.negative_heads_permutation == "flip_repeat":
            q2 = torch.flip(q2, dims=(2,))
            q2 = q2.repeat(1, 1, self.n_heads // self.n_negative_heads, 1)
            k2 = torch.flip(k2, dims=(2,))
            k2 = k2.repeat(1, 1, self.n_heads // self.n_negative_heads, 1)
        elif self.negative_heads_permutation == "flip_interleave":
            q2 = torch.flip(q2, dims=(2,))
            q2 = q2.repeat_interleave(self.n_heads // self.n_negative_heads, dim=2)
            k2 = torch.flip(k2, dims=(2,))
            k2 = k2.repeat_interleave(self.n_heads // self.n_negative_heads, dim=2)
        elif self.negative_heads_permutation == "roll_repeat":
            q2 = torch.roll(q2, shifts=1, dims=(2,))
            q2 = q2.repeat(1, 1, self.n_heads // self.n_negative_heads, 1)
            k2 = torch.roll(k2, shifts=1, dims=(2,))
            k2 = k2.repeat(1, 1, self.n_heads // self.n_negative_heads, 1)
        elif self.negative_heads_permutation == "repeat_roll":
            q2 = q2.repeat(1, 1, self.n_heads // self.n_negative_heads, 1)
            q2 = torch.roll(q2, shifts=1, dims=(2,))
            k2 = k2.repeat(1, 1, self.n_heads // self.n_negative_heads, 1)
            k2 = torch.roll(k2, shifts=1, dims=(2,))
        elif self.negative_heads_permutation == "roll_interleave":
            q2 = torch.roll(q2, shifts=1, dims=(2,))
            q2 = q2.repeat_interleave(self.n_heads // self.n_negative_heads, dim=2)
            k2 = torch.roll(k2, shifts=1, dims=(2,))
            k2 = k2.repeat_interleave(self.n_heads // self.n_negative_heads, dim=2)
        else:
            raise NotImplementedError

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
                # enable_gqa=self.enable_gqa,
            )
            attn1 = attn1.transpose(1, 2)
            attn2, attn2_scores = manual_attention(
                q2.transpose(1, 2),
                k2.transpose(1, 2),
                v.transpose(1, 2),
                causal=True,
                # enable_gqa=self.enable_gqa,
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
        attn = attn.reshape(bsz, self.seq_len, self.n_heads * self.dhead)

        attn = self.out_proj(attn)
        return attn

    def log_light(self):
        return {
            "lambda": self.logging_cache["lambda"],
        }


class VanillaFlashDiff1(nn.Module):
    """
    (Recommended)
    DiffAttn implemented with FlashAttention, for packages that support different qk/v dimensions
    e.g., our customized-flash-attention (https://aka.ms/flash-diff) and xformers (https://github.com/facebookresearch/xformers)
    """

    def __init__(
        self,
        dmodel,
        n_heads,
        n_kv_heads,
        use_rope,
        seq_len,
        init_type,
        init_scale,
    ):
        super().__init__()
        self.dmodel = dmodel
        # num_heads set to half of Transformer's #heads
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.enable_gqa = self.n_heads != self.n_kv_heads
        assert self.n_heads % self.n_kv_heads == 0
        self.save_attention_weights = False
        self.attention_weights = None

        self.dhead = dmodel // n_heads
        self.scaling = self.dhead**-0.5
        self.seq_len = seq_len
        self.use_rope = use_rope
        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(
                self.dhead, base=500000.0, interleaved=True
            )
            self.rotary_emb._update_cos_sin_cache(self.seq_len, dtype=torch.float32)

        self.q_proj = Linear(
            dmodel, dmodel, bias=False, init_type=init_type, init_scale=init_scale
        )
        self.k_proj = Linear(
            dmodel,
            self.dhead * n_kv_heads,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.v_proj = Linear(
            dmodel,
            self.dhead * n_kv_heads,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.out_proj = Linear(
            dmodel, dmodel, bias=False, init_type=init_type, init_scale=init_scale
        )

    def forward(
        self,
        x,
        # rel_pos=None,
        # attn_mask=None,
    ):
        bsz, _, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, self.seq_len, self.n_heads, self.dhead)
        k = k.view(bsz, self.seq_len, self.n_kv_heads, self.dhead)
        v = v.view(bsz, self.seq_len, self.n_kv_heads, self.dhead)

        if self.use_rope:
            assert self.rotary_emb._cos_cached.dtype == torch.float32
            rel_pos = (
                self.rotary_emb._cos_cached.to(x.device),
                self.rotary_emb._sin_cached.to(x.device),
            )
            q = apply_rotary_emb(q.to(dtype=torch.float32), *rel_pos, interleaved=True).to(x)
            k = apply_rotary_emb(k.to(dtype=torch.float32), *rel_pos, interleaved=True).to(x)

        q = q.reshape(bsz, self.seq_len, self.n_heads, self.dhead)
        k = k.reshape(bsz, self.seq_len, self.n_kv_heads, self.dhead)
        if self.save_attention_weights:
            attn, attn_scores = manual_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                causal=True,
                enable_gqa=self.enable_gqa,
            )
            attn = attn.transpose(1, 2)
            if False and self.block_number == 0:
                reference_attn = flash_attn_func(
                    q.to(dtype=torch.bfloat16),
                    k.to(dtype=torch.bfloat16),
                    v.to(dtype=torch.bfloat16),
                    causal=True,
                )
                assert torch.allclose(
                    attn, reference_attn, atol=1e-3
                ), f"Manual attn does not match reference attn: {attn-reference_attn}"

            self.attention_weights = attn_scores
        else:
            attn = flash_attn_func(
                q,
                k,
                v,
                causal=True,
            )

        attn = attn.reshape(bsz, self.seq_len, self.n_heads * self.dhead).to(x)

        attn = self.out_proj(attn)
        return attn
