#!/bin/bash

# module CUDA/11.7.0
echo "Will run the following command:"
echo "$@"
echo "==============================="
# conda activate /scratch/project_465001227/llm-random-group/env
. "/users/ludzieje/miniconda3/etc/profile.d/conda.sh"
conda activate /scratch/project_465001227/llm-random-group/conda3.11
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export PYTHONPATH=$PYTHONPATH:./
# export USE_LIBUV=1
# NUM_TRAINERS
# srun $@
# torchrun --nproc-per-node=2 $@
srun torchrun --standalone --nnodes=1 --nproc-per-node=2 $@

# torchrun
#     --standalone
#     --nnodes=1
#     --nproc-per-node=$NUM_TRAINERS
#     YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)