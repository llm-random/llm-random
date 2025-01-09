#!/bin/bash -l

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=( "${nodes[@]}" )
head_node=${nodes_array[0]}
head_node_ip=$(getent hosts "$head_node" | awk 'NR==1{print $1}')

# Replace placeholders in command-line arguments
args=()
for arg in "$@"; do
    arg="${arg//__HEAD_NODE_IP__/$head_node_ip}"
    arg="${arg//__RANDOM__/$RANDOM}"
    args+=( "$arg" )
done

export LOGLEVEL=INFO

module load ML-bundle/24.06a
source /net/storage/pr3/plgrid/plggllmeffi/datasets/make_singularity_image/venv/bin/activate
echo "Will run the following command:"
echo "${args[@]}"
echo "==============================="
"${args[@]}"
