#!/bin/bash

for pruner_n_steps in 100 1000 10000
do
    for total_frac_pruned in 0.25 1 4
    do
        name="grid_steps_${pruner_n_steps}_frac_${total_frac_pruned}"

        # calculate pruner_prob
        n_pruning_steps=$(( 100000/$pruner_n_steps ))
        n_steps_prune_all=$( echo $n_pruning_steps / $total_frac_pruned | bc) # n of steps to prune 100% weights
        no_prune_prob=$( echo "e( l(0.01)/$n_steps_prune_all )" | bc -l ) # here nth root is taken
        pruner_prob=$( echo "1 - $no_prune_prob" | bc )

        sbatch \
        --partition=common \
        --qos=16gpu7d \
        --gres=gpu:titanv:1 \
        --job-name=${name} \
        --output=/home/jkrajewski/${name}.txt \
        --time=0-10:00:00 \
        research/reinitialization/scripts/run_train_grid.sh \
        $name \
        struct_magnitude_recycle \
        $pruner_n_steps \
        $pruner_prob
    done
done
