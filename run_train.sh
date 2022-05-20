#!/bin/bash
#
#SBATCH --job-name=lizard_train3
#SBATCH --partition=common
#SBATCH --qos=16gpu3d
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --time=0-10:00:00
#SBATCH --output=/home/sj359674/sbatchlogs5.txt

source venv/bin/activate
python3 bert_train.py
