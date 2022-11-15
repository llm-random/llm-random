import argparse

import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
from clearml import Task

from lizrd.core import misc
from lizrd.core import bert
from research.reinitialization.core import linears, linears_recycle
from research.reinitialization.core.pruner import VariableProbabilityPruner
from lizrd.train.train_utils import (
    get_model,
    get_processed_dataset,
    LTHTrainer,
)

parser = argparse.ArgumentParser()

parser.add_argument("--use_clearml", action="store_true")

parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--cutoff", type=int, default=128)
parser.add_argument("--dm", type=int, default=256)
parser.add_argument("--dff", type=int, default=1024)
parser.add_argument("--n_blocks", type=int, default=4)
parser.add_argument("--heads", type=int, default=4)
parser.add_argument("--pruner_prob", type=float, default=0.1)
parser.add_argument("--project_name", type=str)

parser.add_argument("--ff_layer", type=str, default="struct_magnitude_prune")
parser.add_argument("--learning_rate", type=float, default=8e-4)
parser.add_argument("--mask_loss_weight", type=float, default=1.0)
parser.add_argument("--class_loss_weight", type=float, default=1.0)
parser.add_argument("--mask_percent", type=float, default=0.15)
parser.add_argument("--n_steps_per_run", type=int, default=10000)
parser.add_argument("--n_steps_eval", type=int, default=100)
parser.add_argument("--target_params", type=float, default=0.1)
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

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S.%f")
modelpath = f"runs/lth/{timestamp}"
writer = SummaryWriter(log_dir=modelpath)

pruner = VariableProbabilityPruner()

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
else:
    raise ValueError("ff_layer not recognized")

misc.print_available_gpus()
pdataset_creator = lambda: get_processed_dataset(
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
optimizer_creator = lambda model: torch.optim.Adam(model.parameters(), lr=args.learning_rate)
trainer = LTHTrainer(
    model=model,
    optimizer_creator=optimizer_creator,
    pdataset_creator=pdataset_creator,
    pruner=pruner,
    batch_size=args.batch_size,
    vocab_size=VOCAB_SIZE,
    mask_percent=args.mask_percent,
    mask_loss_weight=args.mask_loss_weight,
    modelpath=modelpath,
    writer=writer,
    n_steps_per_run=args.n_steps_per_run,
    n_steps_eval=args.n_steps_eval,
    pruning_rate=args.pruner_prob,
    target_params=args.target_params,
)
trainer.train()
