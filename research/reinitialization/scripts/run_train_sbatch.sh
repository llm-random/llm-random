#!/bin/bash
#
#SBATCH --job-name=lizard_train_sparse_pruner
#SBATCH --partition=common
#SBATCH --qos=16gpu7d
#SBATCH --gres=gpu:titanv:1
#SBATCH --output=/home/jkrajewski/sbatchlogs_unstruct_magnitude.txt

source venv/bin/activate
python3 -m research.reinitialization.train.reinit_train \
    --use_clearml \
    --name=lizard_train_sparse_pruner \
    --batch_size=64 \
    --cutoff=128 \
    --dm=256 \
    --dff=1024 \
    --n_blocks=4 \
    --heads=4 \
    --project_name="jkrajewski/reinit"
