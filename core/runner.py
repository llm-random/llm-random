import torch

from lizrd.text import tokenizers
from core import layers
from core.partials import (
    build_attention_partial_functions,
    build_embedding_partial_functions,
    build_ff_partial_functions,
    build_norm_block_partial_functions,
)
from research.datasets import get_processed_dataset
from lizrd.train.scheduler import get_scheduler


class Runner:
    def __init__(self, args, device):
        self.build_model(args)
        self.build_optimizer(args)
        self.build_scheduler(args)
        self.build_dataloaders(args, device)

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
        print("embedding: ", embedding.layers[0].weight[1])
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

    def build_dataloaders(self, args, device):
        common_dataloaders_kwargs = {
            "sequence_length": args.seq_length,
            "device": device,
            "num_workers": args.num_workers,
            "batch_size": args.batch_size,
            "seed": args.data_seed,
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
