import torch
import torch.nn as nn
import torch.nn.functional as F

from lizrd.core.llm import Attention, LLM
from lizrd.core.misc import LoggingLayer


def muP_attention_mechanism(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dhead: int,
    causal: bool,
    attn_scale: float = None,
):
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_math=False, enable_mem_efficient=False
    ):
        output = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
            is_causal=causal,
            scale=attn_scale,
        )
    return output


class muP_AttentionMechanism(nn.Module):
    def __init__(self, mode: bool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mode = mode

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dhead: int,
        causal: bool,
        *args,
        **kwargs,
    ):
        return muP_attention_mechanism(
            query=query,
            key=key,
            value=value,
            dhead=dhead,
            causal=causal,
            mode=self.mode,
        )


# if key ,or key_flash works as good as dhead mode, implement changes in forward only
class muP_Attention(Attention):
    def __init__(
        self,
        dmodel,
        heads,
        causal,
        init_type: str,
        init_scale: float,
        dhead=None,
        flash=False,
        mode="vanilla",
    ):
        super(muP_Attention, self).__init__(
            dmodel,
            heads,
            causal,
            init_type,
            init_scale,
            dhead=dhead,
            flash=flash,
        )
        self.attention_mechanism = muP_AttentionMechanism(mode=mode)


class nonResidual(LoggingLayer):
    def __init__(self, layer, alpha=1.0, m_d=1.0):
        super(nonResidual, self).__init__()
        self.layer = layer
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.register_buffer("m_d", torch.tensor(m_d, dtype=torch.float32))

    def forward(self, x):
        out = self.layer(x)
        out *= self.alpha / self.m_d  # muP scaling
        self.update_cache_for_logging("update", out)
        return out

    def log_heavy(self):
        updates = self.logging_cache["update"]

        update_norms = torch.norm(updates, dim=-1)

        update_norms_mean = torch.mean(update_norms)
        update_norms_std = torch.std(update_norms)

        return {
            "update_norms/mean": update_norms_mean,
            "update_norms/std": update_norms_std,
        }


class muP_LLM(LLM):
    def __init__(self, embedding_layer, encoder_tower, head, mup_config: dict = None):
        super(muP_LLM, self).__init__(embedding_layer, encoder_tower, head)

        alpha_in = 1.0
        alpha_out = 1.0
        m_d = 1.0

        self.mup = False
        if mup_config is not None:
            self.mup = True
            # Register alpha_in and alpha_out as buffers to make them non-trainable
            alpha_in = mup_config["alpha_in"]
            alpha_out = mup_config["alpha_out"]
            m_d = mup_config["m_d"]

        self.embedding_layer = nonResidual(
            self.embedding_layer, alpha=alpha_in, m_d=1.0
        )
        self.head = nonResidual(self.head, alpha=alpha_out, m_d=m_d)

    def forward(self, *args, **kwargs):
        x = self.embedding_layer(*args, **kwargs)
        x = self.encoder(x)
        x = self.head(x)
        return x
