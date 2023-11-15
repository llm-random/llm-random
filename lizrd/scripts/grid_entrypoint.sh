#!/bin/bash

source ../venv/bin/activate
export TOKENIZERS_PARALLELISM=true

echo "Will run the following command:"
echo "$@"
echo "==============================="
$@