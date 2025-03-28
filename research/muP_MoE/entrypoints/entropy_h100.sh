#!/bin/bash -l

source ~/miniconda3/etc/profile.d/conda.sh
echo "Running Entropy H100 Entrypoint"
conda activate /storage_nvme_1/llm-random/differential-conda
echo "Will run the following command:"
echo "$@"
echo "==============================="
$@