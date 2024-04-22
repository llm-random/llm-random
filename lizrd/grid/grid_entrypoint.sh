#!/bin/bash

conda activate llm-random
echo "Will run the following command:"
echo "$@"
echo "==============================="
$@