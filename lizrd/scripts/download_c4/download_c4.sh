#!/bin/bash


# 

source venv/bin/activate # huggingface datasets needs to be installed
python download_c4.py --dataset_dir "$1" --cache_dir "$2"

