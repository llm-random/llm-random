import argparse

import torch

from lizrd.core import misc
from lizrd.support.logging import get_logger
from lizrd.train.train_utils import (
    get_model,
    get_processed_dataset,
)
from research.conditional.train.trainers import ConditionalTrainer
from research.conditional.train.utils import (
    introduce_parser_arguments,
    get_attention_layer,
    get_ff_layer,
)

parser = argparse.ArgumentParser()
introduce_parser_arguments(parser)
args = parser.parse_args()


VOCAB_SIZE = 30522  # BertTokenizer uses this many words
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
misc.print_available_gpus()

train_dataloader = get_processed_dataset(
    max_total_length=args.cutoff,
    mask_percent=args.mask_percent,
    device=DEVICE,
    num_workers=args.num_workers,
    batch_size=args.batch_size,
    seed=args.data_seed,
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
)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
trainer = ConditionalTrainer(
    model=model,
    optimizer=optimizer,
    train_dataloader=train_dataloader,
    batch_size=args.batch_size,
    vocab_size=VOCAB_SIZE,
    mask_percent=args.mask_percent,
    mixed_precision=args.mixed_precision,
)

logger = get_logger(args, model, VOCAB_SIZE)
trainer.train(args.n_steps)
