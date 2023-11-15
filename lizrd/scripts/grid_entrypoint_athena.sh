#!/bin/bash

module CUDA/11.7.0

export TOKENIZERS_PARALLELISM=true

echo "Will run the following command:"
echo "$@"
echo "==============================="
$@