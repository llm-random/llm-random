import datetime
import re

import torch

from lizrd.core import bert
from research.nonlinearities.core import research_bert
from research.nonlinearities.temporary_code import temp_research_bert


class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.steps = 0
        self.default_lrs = [
            param_group["lr"] for param_group in self.optimizer.param_groups
        ]

    def step(self):
        self.steps += 1
        lrs = self.get_lrs()
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = lrs[i]

    def get_lrs(self):
        if self.steps < self.warmup_steps:
            alpha = float(self.steps) / float(max(1, self.warmup_steps))
            lrs = [alpha * base_lr for base_lr in self.default_lrs]
        else:
            lrs = self.default_lrs
        return lrs


def divide_model_parameters(model, args):
    "Iterates over named modules of the model, and gathers them into two groups: for modules whose name includes \
    'forward' returns them separately with args.learning_rate_ff, while the rest uses the deafult args.learning_rate"
    params_non_ff = []
    params_ff = []
    if args.learning_rate_ff is None:
        args.learning_rate_ff = args.learning_rate
    for name, param in model.named_parameters():
        if "forward" in name:
            params_ff.append(param)
        else:
            params_non_ff.append(param)
    return [
        {"params": params_non_ff},
        {"params": params_ff, "lr": args.learning_rate_ff},
    ]


def process_and_remove_nan(tensor):
    tensor = tensor.detach().cpu()
    mask = torch.isnan(tensor) | torch.isinf(tensor) | torch.isinf(-tensor)
    nan_freq = mask.sum().item() / mask.numel()
    tensor_clean = tensor[~mask]
    return tensor_clean, nan_freq


def clean_name_for_logging(tag):
    block_name = re.findall("block_[0-9]+", tag)[0].replace("_", " ")
    layer_name = tag.split("logging_")[-1].split(".")[0].replace("_", " ")
    layer_name += " weight" if "weight" in tag else " bias"
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


def map_args(args):
    if args.ff_mode == "multineck_normed_chunked_adjusted_dff":
        args.d_ff_head = int(
            args.dmodel / args.n_ff_heads * args.neck_width_increase_ratio
        )
        args.dff = int(args.dff / args.neck_width_increase_ratio)
        args.ff_mode = "multineck_normed_chunked"
    elif args.ff_mode == "multineck_normed_chunked_adjusted_neck_width":
        args.d_ff_head = int(args.dmodel / args.n_ff_heads)
        args.ff_mode = "multineck_normed_chunked"
    return args


def get_ff_layer(args):
    mode = args.ff_mode
    if mode == "vanilla":
        ff_layer_type, ff_args = bert.FeedForward, (args.dmodel, args.dff)
    elif mode == "vanilla_chunked":
        ff_layer_type, ff_args = temp_research_bert.VanillaChunked, (
            args.dmodel,
            args.dff,
            args.n_chunks,
        )
    elif mode == "vanilla_einmix":
        ff_layer_type, ff_args = temp_research_bert.LinearEinmix, (
            args.dmodel,
            args.dff,
        )
    elif mode == "bottleneck":
        ff_layer_type, ff_args = research_bert.FeedForwardBottleneck, (
            args.dmodel,
            args.exp_rate,
        )
    elif mode == "bottleneck_forced":
        ff_layer_type, ff_args = temp_research_bert.FeedForwardBottleneckFORCED, (
            args.dmodel,
            args.dff,
            args.bottleneck_size,
        )
    elif mode == "multilinear":
        ff_layer_type, ff_args = temp_research_bert.FeedForwardMultilinear, (
            args.dmodel,
            args.dff,
            args.n_ff_heads,
        )
    elif mode == "bottleneckFORCED":
        ff_layer_type, ff_args = temp_research_bert.FeedForwardBottleneckFORCED, (
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
        ff_layer_type, ff_args = temp_research_bert.FeedForwardChoppedNeckFORCED, (
            args.dmodel,
            args.n_chunks,
        )
    elif mode == "multineck_forced":
        ff_layer_type, ff_args = temp_research_bert.FeedForwardMultineckFORCED, (
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
    elif mode == "overparametrized_normed":
        ff_layer_type, ff_args = temp_research_bert.OverparametrisedFeedForwardNormed, (
            args.dmodel,
            args.dff,
        )
    elif mode == "overparametrized_residual":
        (
            ff_layer_type,
            ff_args,
        ) = temp_research_bert.OverparametrisedFeedForwardResidual, (
            args.dmodel,
            args.dff,
        )
    elif mode == "overparametrized_residual_normed":
        (
            ff_layer_type,
            ff_args,
        ) = temp_research_bert.OverparametrisedFeedForwardResidualNormed, (
            args.dmodel,
            args.dff,
        )
    elif mode == "multineck_normed":
        ff_layer_type, ff_args = temp_research_bert.FeedForwardMultineckNormed, (
            args.dmodel,
            args.d_ff_head,
            args.n_ff_heads,
            args.dff,
        )
    elif mode == "multineck_residual":
        ff_layer_type, ff_args = temp_research_bert.FeedForwardMultineckResidual, (
            args.dmodel,
            args.d_ff_head,
            args.n_ff_heads,
            args.dff,
        )
    elif mode == "multineck_residual_normed":
        (
            ff_layer_type,
            ff_args,
        ) = temp_research_bert.FeedForwardMultineckResidualNormed, (
            args.dmodel,
            args.d_ff_head,
            args.n_ff_heads,
            args.dff,
        )
    elif mode == "multineck_chunked":
        (
            ff_layer_type,
            ff_args,
        ) = temp_research_bert.FeedForwardMultineckChunked, (
            args.dmodel,
            args.d_ff_head,
            args.n_ff_heads,
            args.dff,
        )
    elif mode == "multineck_normed_chunked":
        (
            ff_layer_type,
            ff_args,
        ) = temp_research_bert.FeedForwardMultineckNormedChunked, (
            args.dmodel,
            args.d_ff_head,
            args.n_ff_heads,
            args.dff,
        )
    else:
        raise NotImplementedError(f"ff_mode={mode} is not implemented")

    ff_layer_fun = lambda: ff_layer_type(*ff_args)
    return ff_layer_fun
