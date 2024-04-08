from typing import Type, Union
import torch
from functools import partial
from lizrd.core.distributed import wrap_in_fsdp
from lizrd.text import tokenizers
from core import layers
from research.datasets import get_processed_dataset
from lizrd.train.scheduler import get_scheduler
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

######################################################################
######                     PARTIAL FUNCTIONS                    ######
######################################################################
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
        return torch.nn.LayerNorm
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
######################################################################
######                       FSDP HELP                          ######
######################################################################
def get_classes_from_module_names(
    packed_names,
) -> Union[tuple[Type[torch.nn.Module]], None]:
    """
    Unpacks a comma-separated list of module names into a tuple of modules.
    """
    classes = []
    if packed_names is None:
        return None
    for name in packed_names.split(","):
        if name == "Attention":
            classes.append(layers.Attention)
        elif name == "AttentionMechanism":
            classes.append(layers.AttentionMechanism)
        elif name == "FeedForward":
            classes.append(layers.FeedForward)
        elif name == "Residual":
            classes.append(layers.Residual)
        elif name == "TransformerBlock":
            classes.append(layers.TransformerBlock)
        elif name == "TransformerTower":
            classes.append(layers.TransformerTower)
        elif name == "LLM":
            classes.append(layers.LLM)
        elif name == "EmbeddingLayer":
            classes.append(layers.EmbeddingLayer)
        elif name == "PredictionHead":
            classes.append(layers.PredictionHead)
        else:
            raise ValueError(f"Unknown name {name}")
    return tuple(classes)



def get_mixed_precision_ignored_classes(class_names) -> list[Type[torch.nn.Module]]:
    ignored_classes = [
        nn.LayerNorm,
        nn.modules.batchnorm._BatchNorm,
    ]

    selective_precision_modules = get_classes_from_module_names(
        class_names
    )
    if selective_precision_modules is not None:
        ignored_classes += list(selective_precision_modules)

    return ignored_classes


######################################################################
######                     BUILDER CLASS                        ######
######################################################################
class Builder:
    def __init__(self, args, device, data_seeds, rank):
        self.build_model(args)
        self.build_optimizer(args)
        self.build_scheduler(args)
        self.build_dataloaders(args, device, data_seeds, rank)
        #TODO load weights
        if args.fsdp_enabled:
            self.wrap_distributed(args, rank)

    def get_train_artefacts(self):
        return {
            "model": self.model,
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
            "train_dataloader": self.train_dataloader,
        }

    def fetch_embedding_partial_functions(self, args):
        return build_embedding_partial_functions(args)

    def fetch_norm_block_partial_functions(self, args):
        return build_norm_block_partial_functions(args.dmodel, args.norm_class)

    def fetch_attention_partial_functions(self, args):
        return build_attention_partial_functions(args)

    def fetch_ff_partial_functions(self, args):
        return build_ff_partial_functions(args)

    def build_model(self, args):
        embedding_partial_functions = self.fetch_embedding_partial_functions(args)
        norm_block_partial_functions = self.fetch_norm_block_partial_functions(args)
        attention_partial_functions = self.fetch_attention_partial_functions(args)
        ff_partial_functions = self.fetch_ff_partial_functions(args)

        embedding_function = embedding_partial_functions[args.embedding_mode]
        norm_block_function = norm_block_partial_functions[args.residual_mode]
        attention_function = attention_partial_functions[args.attention_mode]
        ff_function = ff_partial_functions[args.ff_mode]

        embedding = embedding_function()
        encoder = layers.TransformerTower(
            args.n_blocks,
            {
                "attention": attention_function,
                "feedforward": ff_function,
            },
            norm_block_function,
        )

        head = layers.PredictionHead(
            args.dmodel,
            (
                tokenizers.BertTokenizer.VOCAB_SIZE
                if args.model_type == "bert"
                else tokenizers.GPTTokenizer.VOCAB_SIZE
            ),
            init_type=args.init_type,
            init_scale=args.init_scale,
        )
        model = layers.LLM(embedding, encoder, head)
        self.model = model

    def build_optimizer(self, args):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.optimizer_weight_decay,
            betas=(args.optimizer_adam_beta1, args.optimizer_adam_beta2),
        )
        self.optimizer = optimizer

    def build_scheduler(self, args):
        scheduler = get_scheduler(
            args.lr_scheduler_type,
            args.lr_scheduler_warmup_steps,
            args.learning_rate,
            args.final_lr_step,
            args.final_lr_fraction,
        )
        self.scheduler = scheduler

    def build_dataloaders(self, args, device, data_seeds, rank):
        batch_size = args.batch_size // args.n_gpus // args.gradient_accumulation_steps if  args.fsdp_enabled  else args.batch_size // args.gradient_accumulation_steps
        common_dataloaders_kwargs = {
            "sequence_length": args.seq_length,
            "device": device,
            "num_workers": args.num_workers,
            "batch_size": batch_size,
            "seed": args.data_seed if data_seeds is None else data_seeds[rank],
            "model_type": args.model_type,
            "dataset_type": args.dataset_type,
            "use_dummy_dataset": args.use_dummy_dataset,
        }

        train_dataloader = get_processed_dataset(
            **common_dataloaders_kwargs,
            dataset_split="train",
            dataset_path=args.train_dataset_path,
        )
        self.train_dataloader = train_dataloader

    def wrap_distributed(self,args, rank):
        fsdp_mixed_precision_ignore_classes = get_mixed_precision_ignored_classes(args.fsdp_selective_precision_modules)
        fsdp_modules_to_wrap = get_classes_from_module_names(args.fsdp_modules_to_wrap)
        precision_dtype = torch.bfloat16 if args.mixed_precision_dtype == "bfloat16" else torch.float32
        self.model = wrap_in_fsdp(
            module=self.model,
            rank=rank,
            param_precision=precision_dtype,
            cast_inputs=True,
            mixed_precision_ignored_classes=fsdp_mixed_precision_ignore_classes,
            offload_params=args.fsdp_offload_params,
            print_model=True,
            min_num_params=args.fsdp_min_num_params,
            modules_to_wrap=fsdp_modules_to_wrap,
            is_logging_process=rank == 0,
        )