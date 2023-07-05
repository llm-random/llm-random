#!/bin/sh
# $1 refers to PID
# $2 refers to the space-separated list of log files
# $3 refers to the number of machines to reserve
# $4 refers to training command
# $5 refers to the log file
echo -e "Running baby_slurm.sh \n\n\n"
echo "PID=$1"
echo "LOG_FILES=$2"
echo "NUM_MACHINES=$3"
echo "TRAINING_COMMAND=$4"
echo "LOG_FILE=$5"
echo -e "\n\n\nWAITING FOR RESOURCES \n\n\n"
export CUDA_VISIBLE_DEVICES=$(python3 -m lizrd.scripts.reserve_machines --process_id $1 --gpu_log_files "$2" --no_machines $3)
echo -e "\n\n\nCUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
$4 2>&1 | tee -a $5
exit