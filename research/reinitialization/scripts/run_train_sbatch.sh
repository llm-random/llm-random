#!/bin/bash
#
#SBATCH --job-name=lizard_train_sparse_pruner
#SBATCH --partition=common
#SBATCH --qos=16gpu7d
#SBATCH --gres=gpu:titanv:1
#SBATCH --time=0-02:00:00
#SBATCH --output=/home/jkrajewski/sbatchlogs_no_ff.txt

source venv/bin/activate
python3 -m research.reinitialization.reinit_train NAME=pruning-no-ff
