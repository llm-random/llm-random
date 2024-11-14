#!/bin/bash

module CUDA/11.7.0
source /net/pr2/projects/plgrid/plggllmeffi/plgmstefaniak/venv/bin/activate
echo "Will run the following command:"
echo "$@"
echo "==============================="
$@