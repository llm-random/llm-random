#!/bin/bash

source ~/sparsity/venv/bin/activate

cd ~/sparsity

python3 -m research.nonlinearities.train.nonlinearities_train \
    --use_clearml \
    --name=new_code_baseline_lower_lr \
    --project_name="nonlinearities/initial_tests" \
    --ff_mode="vanilla" \
    --multineck_mode="none" \
    --n_steps=3000000 \

