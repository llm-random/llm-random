import argparse

import torch

from lizrd.core import misc
from lizrd.support.logging import get_logger
from lizrd.train.train_utils import (
    get_model,
    get_processed_dataset,
)
from research.conditional.utils.conditional_trainer import ConditionalTrainer
from research.conditional.utils.misc_utils import introduce_parser_arguments
from research.conditional.utils.model_utils import get_ff_layer, get_attention_layer

parser = argparse.ArgumentParser()
introduce_parser_arguments(parser)
args = parser.parse_args()


VOCAB_SIZE = 30522 if args.model_type == "bert" else 50257
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
misc.print_available_gpus()

train_dataloader = get_processed_dataset(
    max_total_length=args.cutoff,
    mask_percent=args.mask_percent,
    device=DEVICE,
    num_workers=args.num_workers,
    batch_size=args.batch_size,
    seed=args.data_seed,
    model_type=args.model_type,
)

ff_layer_fun = get_ff_layer(args)
attention_layer_fun = get_attention_layer(args)

model = get_model(
    max_length=args.cutoff,
    vocab_size=VOCAB_SIZE,
    ff_layer_fun=ff_layer_fun,
    attention_layer_fun=attention_layer_fun,
    dm=args.dmodel,
    n_blocks=args.n_blocks,
    device=DEVICE,
    gradient_checkpointing=args.gradient_checkpointing,
)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
logger = get_logger(args, model, VOCAB_SIZE)

trainer = ConditionalTrainer(
    model=model,
    optimizer=optimizer,
    train_dataloader=train_dataloader,
    batch_size=args.batch_size,
    vocab_size=VOCAB_SIZE,
    mask_percent=args.mask_percent,
    mixed_precision=args.mixed_precision,
    logger=logger,
    hack_name=args.hack_name,
    model_type=args.model_type,
    logging_interval_light=args.logging_interval_light,
    logging_interval_heavy=args.logging_interval_heavy,
)

trainer.train(args.n_steps)
