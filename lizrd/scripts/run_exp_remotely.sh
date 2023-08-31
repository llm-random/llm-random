#!/bin/bash
# INSTRUCTIONS:
# 1. needs to be called from llm-random folder
# 2. needs to be called with the host as the first argument (as configured in ~/.ssh/config, e.g. gpu_entropy)
# 3. needs to be called with the path to config file as the second argument (e.g. "runs/my_config.yaml")
# EXAMPLE USAGE: bash lizrd/scripts/run_exp_remotely.sh atena quick.json
set -e

source venv/bin/activate
# run your python script
python3 -m lizrd.scripts.sync_with_remote --host $1
base_dir=$(cat base_dir.txt)
rm base_dir.txt

run_grid_remotely() {
  host=$1
  config=$2
  session_name="mysession"

  script="cd $base_dir && tmux new-session -d -s $session_name bash"
  script+="; tmux send-keys -t $session_name 'python3 -m lizrd.scripts.grid $config' C-m"
  script+="; tmux attach -t $session_name"

  ssh -t $host "$script"
}

# run your bash function
run_grid_remotely $1 $2