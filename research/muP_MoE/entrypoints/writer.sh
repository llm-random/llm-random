#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate momqa
which python3
pip freeze
echo "Will run the following command:"
echo "$@"
echo "==============================="
$@