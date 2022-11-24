import subprocess

pruner_n_steps = ""
for learning_rate in [1e-5, 2e-5, 3e-5]:
    name = f"new_grid_script_lr_{learning_rate}"
    tags = f"TESTING baseline"

    sbatch_args = [
        "--job-name",
        name,
        "--output",
        f"/home/simontwice/{name}.out",
        "--time",
        "20:00:00",
    ]
    script_path = "research/nonlinearities/scripts/run_train_grid.sh"
    script_args = [
        name,
        "struct_magnitude_recycle",
        tags,
    ]

    subprocess.check_call(["sbatch", *sbatch_args, script_path, *script_args])
