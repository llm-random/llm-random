import datetime
import re

import torch

from lizrd.core import bert
from research.nonlinearities.core import research_bert
from research.nonlinearities.temporary_code import temp_research_bert
from research.nonlinearities.temporary_code.temp_research_bert import (
    FeedForwardMultineckFORCED,
    FeedForwardBottleneckFORCED,
)


def process_and_remove_nan(tensor):
    tensor = tensor.detach().cpu()
    mask = torch.isnan(tensor) | torch.isinf(tensor) | torch.isinf(-tensor)
    nan_freq = mask.sum().item() / mask.numel()
    tensor = tensor[~mask]
    return tensor, nan_freq


def clean_name_for_logging(tag):
    block_name = re.findall("block_[0-9]+", tag)[0].replace("_", " ")
    layer_name = tag.split("logging_")[-1].split(".")[0].replace("_", " ")
    return block_name, layer_name


def sample(tensor, p=0.01):
    "samples elements from tensor with probability p"
    return tensor[torch.rand_like(tensor) < p]


def get_attention_layer(args):
    if args.attention_thinning_coeff != 1.0:
        att_dhead = int(args.dmodel // args.n_att_heads * args.attention_thinning_coeff)
        if att_dhead == 0:
            attention_layer_fun = lambda: None
        else:
            attention_layer_fun = lambda: bert.Attention(
                args.dmodel, args.n_att_heads, att_dhead
            )
    else:
        attention_layer_fun = lambda: bert.Attention(args.dmodel, args.n_att_heads)
    return attention_layer_fun


def make_concise_datetime() -> str:
    now = datetime.datetime.now()
    return str(now.year)[-2:] + "_" + now.strftime("%m-%d_%H:%M:%S")


def get_ff_layer(args):
    mode = args.ff_mode
    if mode == "vanilla":
        ff_layer_type, ff_args = bert.FeedForward, (args.dmodel, args.dff)
    elif mode == "vanilla_einmix":
        ff_layer_type, ff_args = research_bert.LinearEinmix, (args.dmodel, args.dff)
    elif mode == "bottleneck":
        ff_layer_type, ff_args = research_bert.FeedForwardBottleneck, (
            args.dmodel,
            args.exp_rate,
        )
    elif mode == "multilinear":
        ff_layer_type, ff_args = research_bert.FeedForwardMultilinear, (
            args.dmodel,
            args.dff,
            args.n_ff_heads,
        )
    elif mode == "bottleneckFORCED":
        ff_layer_type, ff_args = FeedForwardBottleneckFORCED, (
            args.dmodel,
            args.dbottle,
            args.dff,
        )
    elif mode == "multineck":
        ff_layer_type, ff_args = research_bert.FeedForwardMultineck, (
            args.dmodel,
            args.exp_rate,
            args.n_ff_heads,
            args.multineck_mode,
        )
    elif mode == "inception":
        ff_layer_type, ff_args = research_bert.FeedForwardInceptionNeck, (
            args.dmodel,
            args.exp_rate,
            args.inception_head_sizes,
        )
    elif mode == "choppedneck":
        ff_layer_type, ff_args = research_bert.FeedForwardChoppedNeck, (
            args.dmodel,
            args.n_chunks,
        )
    elif mode == "choppedneck_forced":
        ff_layer_type, ff_args = research_bert.FeedForwardChoppedNeckFORCED, (
            args.dmodel,
            args.n_chunks,
        )
    elif mode == "multineck_forced":
        ff_layer_type, ff_args = FeedForwardMultineckFORCED, (
            args.dmodel,
            args.d_ff_head,
            args.n_ff_heads,
            args.dff,
            args.multineck_mode,
        )
    elif mode == "multibias":
        ff_layer_type, ff_args = temp_research_bert.FeedForwardMultibias, (
            args.dmodel,
            args.dff,
            args.n_bias_copies,
        )
    elif mode == "multineck_shuffle":
        ff_layer_type, ff_args = temp_research_bert.MultineckShuffle, (
            args.dmodel,
            args.d_ff_head,
            args.n_ff_heads,
            args.dff,
        )
    elif mode == "overparametrized":
        ff_layer_type, ff_args = temp_research_bert.OverparametrisedFeedForward, (
            args.dmodel,
            args.dff,
        )
    else:
        raise NotImplementedError(f"ff_mode={mode} is not implemented")

    ff_layer_fun = lambda: ff_layer_type(*ff_args)
    return ff_layer_fun
