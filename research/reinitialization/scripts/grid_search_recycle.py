import subprocess
import time

for pruner_n_steps in [10, 100, 1000]:
    for total_frac_pruned in [0.25, 1, 4]:
        name = f"grid_steps_{pruner_n_steps}_frac_{total_frac_pruned}"
        tags = f"new_metrics steps_{pruner_n_steps} frac_{total_frac_pruned}"

        # calculate pruner_prob
        n_pruning_steps = 100000 / pruner_n_steps
        n_steps_prune_all = (
            n_pruning_steps / total_frac_pruned
        )  # n of steps to prune 100% weights
        no_prune_prob = 0.01 ** (1 / n_steps_prune_all)  # here nth root is taken
        pruner_prob = 1 - no_prune_prob

        sbatch_args = [
            "--job-name",
            name,
            "--output",
            f"/home/jkrajewski/{name}.out",
            "--time",
            "10:00:00",
        ]
        script_path = "research/reinitialization/scripts/run_train_grid.sh"
        script_args = [
            name,
            "struct_magnitude_recycle",
            str(pruner_n_steps),
            str(pruner_prob),
            tags,
        ]

        subprocess.check_call(["sbatch", *sbatch_args, script_path, *script_args])
        time.sleep(1)
