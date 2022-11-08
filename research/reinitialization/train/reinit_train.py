import argparse

import torch
import torch.nn.functional as F
import datetime
from torch.utils.tensorboard import SummaryWriter
from clearml import Task

from lizrd.core import misc, bert
from research.reinitialization.core import linears
from research.reinitialization.core import linears_recycle
from research.reinitialization.core.pruner import Pruner
from research.reinitialization.train.train_utils import (
    get_model,
    get_processed_dataset,
    Trainer,
)


parser = argparse.ArgumentParser()

parser.add_argument("--use_clearml", action="store_true")
parser.add_argument("--use_pruner", action="store_true")

parser.add_argument("--batch_size", type=int)
parser.add_argument("--cutoff", type=int)
parser.add_argument("--dm", type=int)
parser.add_argument("--dff", type=int)
parser.add_argument("--n_blocks", type=int)
parser.add_argument("--heads", type=int)
parser.add_argument("--pruner_prob", type=float)
parser.add_argument("--pruner_n_steps", type=int)
parser.add_argument("--project_name", type=str)

parser.add_argument("--ff_layer", type=str, default="regular")
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--mask_loss_weight", type=float, default=1.0)
parser.add_argument("--class_loss_weight", type=float, default=1.0)
parser.add_argument("--mask_percent", type=float, default=0.15)
parser.add_argument("--n_steps", type=int, default=10000001)
parser.add_argument("--n_steps_eval", type=int, default=100)
parser.add_argument("--name", type=str, default="")

args = parser.parse_args()

# constants
VOCAB_SIZE = 30522  # BertTokenizer uses this many words
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.use_clearml:
    task = Task.init(
        project_name=args.project_name,
        task_name=f"{args.name} {datetime.datetime.now()}",
    )
    task.connect(vars(args))

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M")
modelpath = f"runs/wikibooktest/{timestamp}"
writer = SummaryWriter(log_dir=modelpath)

# set pruner if needed
if args.use_pruner:
    pruner = Pruner(
        args.pruner_prob,
        args.pruner_n_steps,
    )

# set ff layer
if args.ff_layer == "regular":
    ff_layer_fun = lambda: bert.FeedForward(args.dm, args.dff)
elif args.ff_layer == "unstruct_prune":
    ff_layer_fun = lambda: linears.UnstructPruneFF(args.dm, args.dff, pruner)
elif args.ff_layer == "struct_prune":
    ff_layer_fun = lambda: linears.StructPruneFF(args.dm, args.dff, pruner)
elif args.ff_layer == "unstruct_magnitude_prune":
    ff_layer_fun = lambda: linears.UnstructMagnitudePruneFF(args.dm, args.dff, pruner)
elif args.ff_layer == "struct_magnitude_prune":
    ff_layer_fun = lambda: linears.StructMagnitudePruneFF(args.dm, args.dff, pruner)
elif args.ff_layer == "unstruct_magnitude_recycle":
    ff_layer_fun = lambda: linears_recycle.UnstructMagnitudeRecycleFF(
        args.dm, args.dff, pruner
    )
elif args.ff_layer == "struct_magnitude_recycle":
    ff_layer_fun = lambda: linears_recycle.StructMagnitudeRecycleFF(
        args.dm, args.dff, pruner
    )

misc.print_available_gpus()
pdataset = get_processed_dataset(
    max_total_length=args.cutoff,
    mask_percent=args.mask_percent,
    device=DEVICE,
)
model = get_model(
    max_length=args.cutoff,
    vocab_size=VOCAB_SIZE,
    ff_layer_fun=ff_layer_fun,
    dm=args.dm,
    n_blocks=args.n_blocks,
    heads=args.heads,
    device=DEVICE,
)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    pdataset=pdataset,
    pruner=pruner,
    batch_size=args.batch_size,
    vocab_size=VOCAB_SIZE,
    mask_percent=args.mask_percent,
    mask_loss_weight=args.mask_loss_weight,
    modelpath=modelpath,
    writer=writer,
)
trainer.train(args.n_steps, args.n_steps_eval)
