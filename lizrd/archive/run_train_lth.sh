#!/bin/bash
#SBATCH --partition=common                    
#SBATCH --time=168:00:00               # Time limit hrs:min:sec
#SBATCH --qos=16gpu7d
#SBATCH --gres=gpu:titanv:1

python3 -m research.reinitialization.train.lth_train \
            --use_clearml --target_params 0.1 \
            --pruner_prob 0.1 --project_name mpioro/lth \
            --n_steps_per_run 100000 \
            --ff_layer struct_prune