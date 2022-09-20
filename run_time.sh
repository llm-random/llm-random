#!/bin/bash
#
#SBATCH --job-name=lizard_time
#SBATCH --partition=common
#SBATCH --qos=24gpu7d
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --time=0-01:00:00
#SBATCH --output=/home/jaszczur/sbatchlogs_time1.txt

source venv/bin/activate
python3 bert_time.py
