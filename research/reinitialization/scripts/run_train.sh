#!/bin/bash

source venv/bin/activate

python3 -m research.reinitialization.train.reinit_train \
    --use_pruner \
    --use_clearml \
    --name=lizard_train_sparse_pruner \
    --batch_size=32 \
    --cutoff=32 \
    --dm=256 \
    --dff=1024 \
    --n_blocks=4 \
    --heads=4 \
    --pruner_prob=0.01 \
    --pruner_n_steps=10 \
    --project_name="jkrajewski/reinit"
