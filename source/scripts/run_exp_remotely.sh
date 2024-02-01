#!/bin/bash
# INSTRUCTIONS:
# 1. needs to be called from somewhere in SECRET_DIRECTORY directory
# 2. needs to be called with the host as the first argument (as configured in ~/.ssh/config, e.g. gpu_entropy)
# 3. needs to be called with the path to config file as the second argument (e.g. "runs/my_config.yaml")
# EXAMPLE USAGE: bash source/scripts/run_exp_remotely.sh atena source/configs/quick.json
set -e

source venv/bin/activate
# run your python script
python3 -m source.support.sync_and_version --host $1
base_dir=$(cat /tmp/base_dir.txt)
git_branch=$(cat /tmp/git_branch.txt)
rm /tmp/base_dir.txt
rm /tmp/git_branch.txt



run_grid_remotely() {
  host=$1
  config=$2
  session_name=$(date "+%Y_%m_%d_%H_%M_%S")
  echo "Running grid search on $host with config $config"

  script="cd $base_dir && tmux new-session -d -s $session_name bash"
  script+="; tmux send-keys -t $session_name 'python3 -m source.scripts.grid --config_path=$config --git_branch=$git_branch"
  if [ -n "$NEPTUNE_API_TOKEN" ]; then
    script+=" --neptune_key=$NEPTUNE_API_TOKEN"
  fi
  script+="' C-m"
  script+="; tmux attach -t $session_name"
  script+="; echo 'done'" #black magic: without it, interactive sessions like "srun" cannot be detached from without killing the session

  ssh -t $host "$script"
}

for i in "${@:2}"
do
  # run your bash function
run_grid_remotely $1 $i
done

