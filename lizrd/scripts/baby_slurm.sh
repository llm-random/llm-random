#!/bin/sh
# $1 refers to tmux session name
# $2 refers to the tmux socket path
# $2 refers to the space-separated list of log files
# $3 refers to the number of machines to reserve
# $4 refers to training command
# $5 refers to the log file
echo -e "Running baby_slurm.sh \n\n\n"
echo "SESSION_NAME=$1"
echo "TMUX_SOCKET_PATH=$2"
echo "LOG_FILES=$3"
echo "NUM_MACHINES=$4"
echo "TRAINING_COMMAND=$5"
echo "LOG_FILE=$6"
echo -e "\n\n\nWAITING FOR RESOURCES \n\n\n"
export CUDA_VISIBLE_DEVICES=$(python3 -m lizrd.scripts.reserve_machines --session_name $1 --socket_path $2 --gpu_log_files "$3" --no_machines $4)
echo -e "\n\n\nCUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
$5 2>&1 | tee -a $6
echo "EXITING BABY_SLURM"
tmux -S $2 kill-session -t $1