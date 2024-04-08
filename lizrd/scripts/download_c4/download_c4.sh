#!/bin/bash
#SBATCH --job-name=download_c4
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=125G
#SBATCH --gres=gpu:0

source venv/bin/activate # huggingface datasets needs to be installed
python download_c4.py --dataset_dir "$1" --cache_dir "$2"