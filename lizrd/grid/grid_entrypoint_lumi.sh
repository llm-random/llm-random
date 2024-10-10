#!/bin/bash

echo "Will run the following command:"
echo "$@"
echo "==============================="
export PYTHONPATH=$PYTHONPATH:./

srun $@
