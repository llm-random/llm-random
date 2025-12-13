#!/bin/bash

ml CUDA/12.4.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/pd03/plgrid/plgllmefficont2/better-differential/conda-env
which python3
pip freeze
echo "Will run the following command:"
echo "$@"
echo "==============================="
$@