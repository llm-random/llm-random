#!/bin/bash
#SBATCH --job-name=once_again_baseline
#SBATCH --partition=common
#SBATCH --qos=16gpu7d
#SBATCH --gres=gpu:1
#SBATCH --output=/home/simontwice/not_important.txt
#SBATCH --time=0-30:00:00


source ~/sparsity/venv/bin/activate

cd ~/sparsity

#python3 -m research.nonlinearities.train.nonlinearities_train \
#    --use_clearml \
#    --name=thin_attention_experiments \
#    --project_name="nonlinearities/initial_tests/compare_baseline_and_thinner_attention" \
#    --name="thin_005" \
#    --ff_mode="vanilla" \
#    --n_steps=3000000 \
#    --tags mute_out_attention \
#    --attention_mode="thin" \
#    --attention_thinning_coeff=.05 \


python3 -m research.nonlinearities.train.nonlinearities_train \
    --use_clearml \
    --name=seed_variance_experiments \
    --project_name="nonlinearities/initial_tests/compare_different_seeds" \
    --name="seed5" \
    --ff_mode="vanilla" \
    --n_steps=3000000 \
#    --tags match_baseline_size_with_every_multineck_head baseline  \

#
#python3 -m research.nonlinearities.train.nonlinearities_train \
#    --use_clearml \
#    --name=new_code_baseline_lower_lr \
#    --project_name="nonlinearities/initial_tests/compare_baseline_and_fat_multineck" \
#    --name="headsize_256_nheads_2" \
#    --ff_mode="multineck_forced" \
#    --multineck_mode="none" \
#    --n_steps=3000000 \
#    --tags match_baseline_size_with_every_multineck_head  \
#    --n_ff_heads=2 \
#    --d_ff=512 \
#    --dmodel=2048 \
#    --d_ff_head=256 \

