#!/bin/bash -l

echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
echo "Hostnames:"
scontrol show hostnames $SLURM_JOB_NODELIST
nodes_array=( "${nodes[@]}" )
head_node=${nodes_array[0]}
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=( "${nodes[@]}" )
head_node=${nodes_array[0]}
echo "Head Node: $head_node"
echo "Output of getent hosts for head node:"
getent hosts "$head_node"
head_node_ip=$(getent hosts "$head_node" | awk 'NR==1{print $1}')
echo Node IP: $head_node_ip
export LOGLEVEL=INFO

# Replace placeholders in command-line arguments
args=()
for arg in "$@"; do
    arg="${arg//__HEAD_NODE_IP__/$head_node_ip}"
    arg="${arg//__RANDOM__/$RANDOM}"
    args+=( "$arg" )
done

module load ML-bundle/24.06a
source /net/storage/pr3/plgrid/plggllmeffi/datasets/make_singularity_image/venv/bin/activate
echo "Will run the following command:"
echo "${args[@]}"
echo "==============================="
"${args[@]}"