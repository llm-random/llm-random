#!/bin/bash

module CUDA/11.7.0
source /home/ubuntu/mstefaniak/venv/bin/activate
echo "MoLE venv!"
echo "Will run the following command:"
echo "$@"
echo "==============================="
$@
