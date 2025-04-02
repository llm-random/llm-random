#!/bin/bash -l

source ~/miniconda3/bin/activate
echo "Running Entropy H100 Entrypoint"
conda activate llm-random-310
echo "Will run the following command:"
echo "$@"
echo "==============================="
$@