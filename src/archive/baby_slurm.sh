#!/bin/bash
# $1 refers to tmux session name
# $2 refers to the tmux socket path
# $3 refers to the space-separated list of log files
# $4 refers to the number of machines to reserve
# $5 refers to training command
# $6 refers to the log file



echo -e "Running baby_slurm.sh \n\n\n"
echo "SESSION_NAME=$1"
echo "TMUX_SOCKET_PATH=$2"
echo "LOG_FILES=$3"
echo "NUM_MACHINES=$4"
echo "TRAINING_COMMAND=$5"
echo "LOG_FILE=$6"
echo -e "\n\n\nWAITING FOR RESOURCES \n\n\n"

export CUDA_VISIBLE_DEVICES=$(python3 -m src.scripts.reserve_machines --session_name $1 --socket_path $2 --gpu_log_files "$3" --no_machines $4)
export SESSION=$1
export SOCKET=$2
echo -e "\n\n\nCUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Start a new process group
set -m

# define the cleanup procedure
cleanup() {
    echo -e "\n\n\nCLEANING UP \n\n\n"
    IFS=', ' read -r -a array <<< "$CUDA_VISIBLE_DEVICES"
    for index in "${array[@]}"
    do
        echo "Resetting GPU $index"
        nvidia-smi --gpu-reset -i $index
    done
    echo "Killing all child processes"
    # Kill all processes in the process group except for the cleanup process
    [[ -z "$(jobs -p)" ]] || kill $(jobs -p)
    echo "Killing tmux session $SESSION and exiting baby_slurm"
    tmux -S $SOCKET kill-session -t $SESSION

}

# register the cleanup function to be called on the EXIT signal
trap cleanup EXIT

# the rest of the script
$5 2>&1 | tee -a $6
