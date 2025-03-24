#!/bin/bash -l

source ~/miniconda3/etc/profile.d/conda.sh
echo "Running Entropy H100 Entrypoint"
conda activate /home/mpioro/context-scaling
echo "Will run the following command:"
echo "$@"
echo "==============================="
$@