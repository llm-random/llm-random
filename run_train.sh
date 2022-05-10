#!/bin/bash
#
#SBATCH --job-name=lizard_train
#SBATCH --partition=common
#SBATCH --qos=16gpu3d
#SBATCH --gres=gpu:1
#SBATCH --time=0-05:00:00
#SBATCH --output=/home/sj359674/sbatchlogs.txt

source venv/bin/activate
python3 bert_train.py
