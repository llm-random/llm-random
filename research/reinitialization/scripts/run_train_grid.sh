#!/bin/bash

source venv/bin/activate
python3 -m research.reinitialization.train.reinit_train \
    --use_clearml \
    --name=$1 \
    --batch_size=64 \
    --cutoff=128 \
    --dm=256 \
    --dff=1024 \
    --n_blocks=4 \
    --heads=4 \
    --project_name="jkrajewski/reinit" \
    --pruner_delay=20000 \
    --ff_layer=$2 \
    --use_pruner \
    --pruner_n_steps=$3 \
    --pruner_prob=$4 \
    --tags $5 \
