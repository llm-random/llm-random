#!/bin/bash

source venv/bin/activate
echo "HALO: $PWD"

echo "Will run the following command:"
echo "$@"
echo "==============================="
$@