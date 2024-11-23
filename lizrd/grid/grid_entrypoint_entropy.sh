#!/bin/bash

module CUDA/11.7.0
source /home/ludziej_a100/ms/venv/bin/activate
echo "MoLE venv!"
echo "Will run the following command:"
echo "$@"
echo "==============================="
$@
