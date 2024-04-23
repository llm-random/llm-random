#!/bin/bash
# INSTRUCTIONS:
# 1. needs to be called from somewhere in llm-random directory
# 2. needs to be called with the host as the first argument (as configured in ~/.ssh/config, e.g. gpu_entropy)
# 3. needs to be called with the path to config file as the second argument (e.g. "runs/my_config.yaml")
# EXAMPLE USAGE: bash lizrd/scripts/run_exp_remotely.sh atena lizrd/configs/quick.json
set -e

branch_filename="__branch__name__.txt"
experiment_dir_filename="__experiment__dir__.txt"

source venv/bin/activate
python3 -m lizrd.support.sync_code --host $1

run_grid_remotely() {
  host=$1
  config=$2

  # Streamlining the output to some variable and to output at the same time is not possible
  python3 -m submit_experiment --host $host --config $config --clone_only  --save_branch_and_dir
  experiment_branch=$(< $branch_filename)
  experiment_directory=$(< $experiment_dir_filename)
  rm $branch_filename
  rm $experiment_dir_filename

  script="cd $experiment_directory && tmux new-session -d -s $experiment_branch bash"
  script+="; tmux send-keys -t $experiment_branch '"
  if [ -n "$NEPTUNE_API_TOKEN" ]; then
    script+="NEPTUNE_API_TOKEN=$NEPTUNE_API_TOKEN "
  fi
  if [ -n "$WANDB_API_KEY" ]; then
    script+="WANDB_API_KEY=$WANDB_API_KEY "
  fi
  script+="./run_experiment.sh' C-m"
  script+="; tmux attach -t $experiment_branch"
  script+="; echo 'done'" #black magic: without it, interactive sessions like "srun" cannot be detached from without killing the session

  ssh -t $host "$script"
}

for i in "${@:2}"
do
  # run your bash function
run_grid_remotely $1 $i
done

