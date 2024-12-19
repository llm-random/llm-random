#!/bin/bash -l

module load ML-bundle/24.06a
echo "Running Custom ENTRYPOINT!!!!!"
source /net/storage/pr3/plgrid/plggllmeffi/momqa/venv/bin/activate
export TRITON_PTXAS_PATH="/net/software/aarch64/el8/CUDA/12.4.0/bin/ptxas"
echo "Will run the following command:"
echo "$@"
echo "==============================="
$@