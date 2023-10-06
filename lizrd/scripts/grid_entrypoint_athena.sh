#!/bin/bash

module CUDA/11.7.0
echo "Will run the following command:"
echo "$@"
echo "==============================="

export MASTER_PORT=12340
export WORLD_SIZE=4


echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

srun $@