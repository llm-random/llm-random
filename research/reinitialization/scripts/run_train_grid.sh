#!/bin/bash
#
#SBATCH --partition=common
#SBATCH --qos=16gpu7d
#SBATCH --gres=gpu:rtx2080ti:1


source venv/bin/activate
python3 -m research.reinitialization.train.reinit_train \
    --use_clearml \
    --name=$1 \
    --project_name="jkrajewski/more_stats" \
    --pruner_delay=20000 \
    --ff_layer=$2 \
    --use_pruner \
    --pruner_n_steps=$3 \
    --pruner_prob=$4 \
    --tags $5 \
    --n_steps 100000 \
