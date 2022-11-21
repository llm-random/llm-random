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
    --project_name=$1 \
    --name=$2 \
    --ff_layer=$3 \
    --tags $4 \
    # Next arguments are optional
    $5 \ # "--use_clearml"
    $6 \ # "--use_pruner"
    --pruner_prob=$7 \
    --pruner_n_steps=$8 \
    --pruner_delay=$9 \
