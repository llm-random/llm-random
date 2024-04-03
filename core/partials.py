from functools import partial
from torch import nn
from lizrd.text import tokenizers
from core import layers


def build_embedding_partial_functions(args):
    return {
        "vanilla": lambda: layers.EmbeddingLayer(
            layers.TokenEmbedding(
                (
                    tokenizers.BertTokenizer.VOCAB_SIZE
                    if args.model_type == "bert"
                    else tokenizers.GPTTokenizer.VOCAB_SIZE
                ),
                args.dmodel,
                init_type=args.init_type,
                init_scale=args.init_scale,
            ),
            layers.PositionalEmbedding(
                args.seq_length,
                args.dmodel,
                init_type=args.init_type,
                init_scale=args.init_scale,
            ),
        )
    }


def get_norm_class(norm_class):
    if norm_class == "layer_norm":
        return nn.LayerNorm
    else:
        raise NotImplementedError(f"Norm type {norm_class} not implemented")


def build_norm_block_partial_functions(dmodel, norm_class_name):
    norm_class = get_norm_class(norm_class_name)
    return {
        "pre_norm": partial(layers.PreNormBlock, dmodel=dmodel, norm_class=norm_class),
    }


def build_attention_partial_functions(args):
    causal = args.model_type == "gpt"
    return {
        "vanilla": lambda: layers.Attention(
            dmodel=args.dmodel,
            heads=args.n_att_heads,
            causal=causal,
            dhead=args.dhead,
            flash=args.flash_attention_enabled,
            init_type=args.init_type,
            init_scale=args.init_scale,
        )
    }


def build_ff_partial_functions(args):
    return {
        "vanilla": lambda: layers.FeedForward(
            args.dmodel, args.dff, init_type=args.init_type, init_scale=args.init_scale
        ),
    }
