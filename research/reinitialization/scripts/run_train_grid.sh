#!/bin/bash

source venv/bin/activate
python3 -m research.reinitialization.train.reinit_train \
    --project_name=$1 \
    --name=$2 \
    --ff_layer=$3 \
    --tags $4 \
    # Next arguments are optional
    $5 \ # "--use_clearml"
    $6 \ # "--use_pruner"
    --pruner_prob=$7 \
    --pruner_n_steps=$8 \
    --pruner_delay=$9 \
