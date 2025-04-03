#!/bin/bash -l

module load ML-bundle/24.06a
source /net/storage/pr3/plgrid/plggllmeffi/constrained-lm-eval-harness/bin/activate
echo "Will run the following command:"
echo "$@"
echo "==============================="
export HF_DATASETS_TRUST_REMOTE_CODE=1
$@