#!/bin/bash

source ../../sparsity/venv/bin/activate
echo "Will run the following command:"
echo "$@"
echo "==============================="
$@