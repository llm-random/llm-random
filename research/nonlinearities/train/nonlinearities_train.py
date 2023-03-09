import argparse
import torch

from lizrd.core import misc
from lizrd.support.logging import get_logger
from lizrd.train.train_utils import (
    get_model,
    get_processed_dataset,
)
from research.nonlinearities.core.trainers import NonlinearityTrainer
from research.nonlinearities.train.utils import (
    get_ff_layer,
    get_attention_layer,
)

parser = argparse.ArgumentParser()

# core hyperparameters, fixed for all experiments; needs a good reason to change

parser.add_argument("--use_clearml", type=bool, default=False)
parser.add_argument("--use_neptune", type=bool, default=False)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--cutoff", type=int, default=128)
parser.add_argument("--dmodel", type=int, default=256)
parser.add_argument("--dff", type=int, default=1024)
parser.add_argument("--n_att_heads", type=int, default=4)
parser.add_argument("--n_blocks", type=int, default=4)
parser.add_argument("--mixed_precision", type=bool, default=False)
parser.add_argument("--log_distributions", type=bool, default=False)
parser.add_argument("--logging_frequency", type=int, default=1000)
parser.add_argument("--mask_loss_weight", type=float, default=1.0)
parser.add_argument("--mask_percent", type=float, default=0.15)
parser.add_argument("--n_steps", type=int, default=100_001)
parser.add_argument("--seed", type=int, default=42)

# parameters usually changed for experiments

parser.add_argument("--ff_mode", type=str, default="vanilla")
parser.add_argument("--project_name", type=str, default="nonlinearities/initial_tests")
parser.add_argument("--name", type=str, default="")
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--tags", nargs="*", type=str, default=None)

# experimental/legacy parameters

parser.add_argument("--exp_rate", type=int, default=4)
parser.add_argument("--bottleneck_size", type=int, default=4)

parser.add_argument("--n_ff_heads", type=int, default=8)
parser.add_argument("--d_ff_head", type=int, default=256)
parser.add_argument("--n_chunks", type=int, default=4)
parser.add_argument("--multineck_mode", type=str, default="none")
parser.add_argument("--inception_head_sizes", nargs="*", type=float)
parser.add_argument("--dbottle", type=int, default=-1)
parser.add_argument("--n_bias_copies", type=int, default=-1)
parser.add_argument("--attention_mode", type=str, default="vanilla")
parser.add_argument("--attention_thinning_coeff", type=float, default=1.0)
parser.add_argument("--n_steps_eval", type=int, default=100)
parser.add_argument("--class_loss_weight", type=float, default=1.0)
parser.add_argument("--save_model_checkpoints", type=bool, default=False)
parser.add_argument("--deterministic", type=bool, default=True)
parser.add_argument("--x_flop", action="store_true")
parser.add_argument("--x_logarithmic", action="store_true")


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
    seed=args.seed,
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
trainer = NonlinearityTrainer(
    model=model,
    optimizer=optimizer,
    train_dataloader=train_dataloader,
    batch_size=args.batch_size,
    vocab_size=VOCAB_SIZE,
    mask_percent=args.mask_percent,
    mixed_precision=args.mixed_precision,
    distribution_logging=args.log_distributions,
    logging_frequency=args.logging_frequency,
)

logger = get_logger(args, model, VOCAB_SIZE)
trainer.train(args.n_steps)
