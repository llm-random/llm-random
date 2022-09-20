#!/bin/bash
#
#SBATCH --job-name=lizard_train_sparse6
#SBATCH --partition=common
#SBATCH --qos=24gpu7d
#SBATCH --gres=gpu:titanv:1
#SBATCH --time=0-04:00:00
#SBATCH --output=/home/jaszczur/sbatchlogs_s6.txt

source venv/bin/activate
python3 bert_train.py
