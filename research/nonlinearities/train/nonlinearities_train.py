import argparse
import time

import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
from clearml import Task

from lizrd.core import misc, bert
from research.nonlinearities.archive.temporary_code.temp_research_bert import FeedForwardMultineckFORCED
from research.nonlinearities.core import research_bert
from lizrd.train.train_utils import (
    get_model,
    get_processed_dataset,)
from research.nonlinearities.core.misc import get_parameter_count
from research.nonlinearities.core.trainers import NonlinearityTrainer

parser = argparse.ArgumentParser()

parser.add_argument("--ff_mode", type=str, default="vanilla")
parser.add_argument("--exp_rate", type=int,default=4)
parser.add_argument("--n_ff_heads", type=int,default=8)
parser.add_argument("--d_ff_head", type=int,default=256)
parser.add_argument("--n_chunks", type=int,default=4)
parser.add_argument("--multineck_mode", type=str, default="none")
parser.add_argument("--inception_head_sizes", nargs='*', type=float)

parser.add_argument("--attention_mode", type=str, default="vanilla")
parser.add_argument("--attention_thinning_coeff", type=float,default=1.0)

parser.add_argument("--use_clearml", action="store_true")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--cutoff", type=int, default=128)
parser.add_argument("--dmodel", type=int, default=512)
parser.add_argument("--dff", type=int, default=2048)
parser.add_argument("--n_att_heads", type=int, default=8)
parser.add_argument("--n_blocks", type=int, default=4)
parser.add_argument("--project_name", type=str, default="nonlinearities/initial_tests")

parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--mask_loss_weight", type=float, default=1.0)
parser.add_argument("--class_loss_weight", type=float, default=1.0)
parser.add_argument("--mask_percent", type=float, default=0.2)
parser.add_argument("--n_steps", type=int, default=100_001)
parser.add_argument("--n_steps_eval", type=int, default=100)
parser.add_argument("--name", type=str, default="")
parser.add_argument("--tags", nargs="*", type=str, default=None)


args = parser.parse_args()

assert args.ff_mode in ["vanilla", "bottleneck", "inception", "multineck", "multineck_forced"], f"ff_mode: {args.ff_mode} must be one of vanilla, bottleneck, inception, multineck, choppedneck"
# constants
VOCAB_SIZE = 30522  # BertTokenizer uses this many words
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M")
modelpath = f"runs/wikibooktest/{timestamp}"
writer = SummaryWriter(log_dir=modelpath)

if args.ff_mode == "vanilla":
    ff_layer_type, ff_args = bert.FeedForward, (args.dmodel, args.dff)
elif args.ff_mode == "bottleneck":
    ff_layer_type, ff_args = research_bert.FeedForwardBottleneck, (args.dmodel, args.exp_rate)
elif args.ff_mode == "multineck":
    ff_layer_type, ff_args = research_bert.FeedForwardMultineck, (args.dmodel, args.exp_rate, args.n_ff_heads, args.multineck_mode)
elif args.ff_mode == "inception":
    ff_layer_type, ff_args = research_bert.FeedForwardInceptionNeck, (args.dmodel, args.exp_rate, args.inception_head_sizes)
elif args.ff_mode == "choppedneck":
    ff_layer_type, ff_args = research_bert.FeedForwardChoppedNeck, (args.dmodel, args.n_chunks)
elif args.ff_mode == "multineck_forced":
    ff_layer_type, ff_args = FeedForwardMultineckFORCED, (args.dmodel, args.d_ff_head, args.n_ff_heads, args.dff, args.multineck_mode)

ff_layer_fun = lambda: ff_layer_type(*ff_args)

if args.attention_mode == "vanilla":
    attention_layer_fun = lambda: bert.Attention(args.dmodel, args.n_att_heads)
elif args.attention_mode == "thin":
    attention_layer_fun = lambda: research_bert.ThinAttention(args.dmodel, args.n_att_heads, thinning_factor=args.attention_thinning_coeff)
elif args.attention_mode == "mute":
    attention_layer_fun = lambda: None
misc.print_available_gpus()

dataset = get_processed_dataset(
    max_total_length=args.cutoff,
    mask_percent=args.mask_percent,
    device=DEVICE,
)
model = get_model(
    max_length=args.cutoff,
    vocab_size=VOCAB_SIZE,
    ff_layer_fun=ff_layer_fun,
    attention_layer_fun=attention_layer_fun,
    dm=args.dmodel,
    n_blocks=args.n_blocks,
    heads=args.n_att_heads,
    device=DEVICE,
)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
trainer = NonlinearityTrainer(
    model=model,
    optimizer=optimizer,
    pdataset=dataset,
    batch_size=args.batch_size,
    vocab_size=VOCAB_SIZE,
    mask_percent=args.mask_percent,
    mask_loss_weight=args.mask_loss_weight,
    modelpath=modelpath,
    writer=writer,
)


dummy_ff_layer = ff_layer_type(*ff_args)
single_ff_layer_parameter_count = get_parameter_count(dummy_ff_layer)
model_parameter_count = get_parameter_count(model)
del dummy_ff_layer

parameter_counts = {
    "single_ff_layer_parameter_count": single_ff_layer_parameter_count,
    "model_parameter_count": model_parameter_count,
}


if args.use_clearml:
    task = Task.init(
        project_name=args.project_name,
        task_name=f"{args.name}",
    )
    task.connect(vars(args))
    task.connect(parameter_counts,"parameter_counts")
    if args.tags:
        task.add_tags(args.tags)
    task.set_random_seed(int(time.time()))

trainer.train(args.n_steps, args.n_steps_eval)
