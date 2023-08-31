#!/bin/bash
# needs to be called from sparsity folder to ac
set -e

source venv/bin/activate
# run your python script
python3 -m lizrd.scripts.sync_with_remote --host $1

run_grid_remotely() {
  host=$1
  config=$2
  script="cd ~/sparsity && find . -name $config -exec python3 -m lizrd.scripts.grid '{}' \;"
  ssh $host "$script"
}

# run your bash function
run_grid_remotely $1 $2