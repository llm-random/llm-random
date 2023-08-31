#!/bin/bash
# INSTRUCTIONS:
# 1. needs to be called from sparsity folder
# 2. needs to be called with the host as the first argument (as configured in ~/.ssh/config, e.g. gpu_entropy)
# 3. needs to be called with the config file as the second argument (e.g. "my_config.yaml")

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