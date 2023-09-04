"""
Script to grid search in recycle layers. Run this script from the root of the project:
$ python3 research/reinitialization/scripts/grid.py
Remember to set RUNNER and PARAMS in the script or add an argument parser.
"""

import datetime
import os
import pprint
import subprocess
import sys
import yaml
from time import sleep

from lizrd.scripts.grid_utils import (
    create_grid,
    get_train_main_function,
    multiply_grid,
    timestr_to_minutes,
    get_machine_backend,
    MachineBackend,
    get_grid_entrypoint,
    get_setup_args_with_defaults,
    translate_to_argparse,
)
from lizrd.support.code_versioning_support import copy_and_version_code

if __name__ == "__main__":
    CLUSTER_NAME = get_machine_backend()
    PROCESS_CALL_FUNCTION = lambda args, env: subprocess.run(
        [str(arg) for arg in args if arg is not None], env=env
    )

    try:
        path = sys.argv[1]
    except IndexError:
        raise ValueError("No config path specified. Aborting...")

    if path.endswith(".yaml"):
        with open(path) as f:
            configs = list(yaml.safe_load_all(f))
    else:
        raise ValueError("config path point to a .yaml")

    for config in configs:
        config["params"]["path_to_config"] = sys.argv[1]

    interactive_options_per_config = [
        config.get("interactive_debug", False) for config in configs
    ]

    assert (
        len(set(interactive_options_per_config)) == 1
    ), f"`interactive_debug` must be the same for all configs"

    interactive_debug_session = interactive_options_per_config[0]

    # list of pairs: a dictionary of training_args and a dictionary of setup_args
    grid = []
    total_no_experiments = 0
    total_minutes = 0

    for i, grid_args in enumerate(configs):
        print(f"\nProcessing config {i}...")
        pprint.pprint(grid_args)
        single_exp_training_args_grid = create_grid(grid_args["params"])

        setup_args = get_setup_args_with_defaults(grid_args, CLUSTER_NAME)
        single_exp_training_args_grid = multiply_grid(
            single_exp_training_args_grid, setup_args["runs_multiplier"]
        )
        no_experiments = len(single_exp_training_args_grid)

        grid += list(zip(single_exp_training_args_grid, [setup_args] * no_experiments))

        total_no_experiments += no_experiments
        minutes_per_exp = timestr_to_minutes(setup_args["time"])
        total_minutes_from_this_grid = no_experiments * minutes_per_exp
        total_minutes += total_minutes_from_this_grid

    if CLUSTER_NAME == MachineBackend.LOCAL and len(grid) > 1:
        raise ValueError(
            f"Running more than one experiment locally is not supported (you are trying to run {len(grid)} experiments). Aborting..."
        )

    if not CLUSTER_NAME == MachineBackend.LOCAL:
        if not interactive_debug_session:
            user_input = input(
                f"Will run {total_no_experiments} experiments, using up {total_minutes} minutes, i.e. around {round(total_minutes / 60)} hours\n"
                f"Continue? [Y/n]"
                f"Will run {total_n_experiments} experiments, using up {total_minutes} minutes, i.e. around {round(total_minutes / 60)} hours\n"
                f"Continue? [Y/n]"
            )
        else:
            user_input = input(
                f"Will run an INTERACTIVE experiment, which will be the first one from the supplied configs. \nContinue? [Y/n]"
            )
        if user_input.lower() not in ("", "y", "Y"):
            print("Aborting...")
            exit(1)

    if not (interactive_debug_session or CLUSTER_NAME == MachineBackend.LOCAL):
        first_exp_training_args, _ = grid[0]
        exp_name = first_exp_training_args["name"]
        name_for_branch = (
            f"{exp_name}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        )
        copy_and_version_code(name_for_branch, name_for_branch, False)
    else:
        print(
            f"Running in debug mode or locally, skip copying code to a new directory."
        )

    slurm_command = "srun" if interactive_debug_session else "sbatch"

    for i, (training_args, setup_args) in enumerate(grid):
        job_name = training_args["name"]
        training_args["n_gpus"] = setup_args["n_gpus"]

        env = None
        runner_params = translate_to_argparse(training_args)

        if CLUSTER_NAME == MachineBackend.ENTROPY:
            subprocess_args = [
                slurm_command,
                "--partition=common",
                "--qos=16gpu7d",
                f"--gres={setup_args['gres']}",
                f"--job-name={job_name}",
                f"--time={setup_args['time']}",
                get_grid_entrypoint(CLUSTER_NAME),
                "python3",
                "-m",
                setup_args["runner"],
                *runner_params,
            ]
        elif CLUSTER_NAME == MachineBackend.ATHENA:
            subprocess_args = [
                slurm_command,
                f"--gres=gpu:{setup_args['n_gpus']}",
                "--partition=plgrid-gpu-a100",
                f"--cpus-per-gpu={setup_args['cpus_per_gpu']}",
                "--account=plgplggllmeffi-gpu-a100",
                f"--job-name={job_name}",
                f"--time={setup_args['time']}",
                get_grid_entrypoint(CLUSTER_NAME),
                "singularity",
                "run",
                "--bind=/net:/net",
                f"--env",
                f"HF_DATASETS_CACHE={setup_args['hf_datasets_cache']}",
                f"-B={os.getcwd()}:/sparsity,{setup_args['hf_datasets_cache']}:{setup_args['hf_datasets_cache']}",
                "--nv",
                setup_args["singularity_image"],
                "python3",
                "-m",
                setup_args["runner"],
                *runner_params,
            ]
        elif CLUSTER_NAME == MachineBackend.IDEAS:
            subprocess_args = [
                slurm_command,
                f"--gres=gpu:{setup_args['n_gpus']}",
                f"--cpus-per-gpu={setup_args['cpus_per_gpu']}",
                f"--job-name={job_name}",
                f"--time={setup_args['time']}",
                "--mem=32G",
                setup_args["nodelist"],
                get_grid_entrypoint(CLUSTER_NAME),
                "singularity",
                "run",
                f"--env",
                f"HF_DATASETS_CACHE={setup_args['hf_datasets_cache']}",
                f"-B={os.getcwd()}:/sparsity,{setup_args['hf_datasets_cache']}:{setup_args['hf_datasets_cache']}",
                "--nv",
                setup_args["singularity_image"],
                "python3",
                "-m",
                setup_args["runner"],
                *runner_params,
            ]
        elif CLUSTER_NAME == MachineBackend.ENTROPY_GPU:
            if setup_args["cuda_visible"] is not None:
                env = os.environ.copy()
                env.update({"CUDA_VISIBLE_DEVICES": setup_args["cuda_visible"]})
            subprocess_args = [
                "singularity",
                "run",
                f"--env",
                f"HF_DATASETS_CACHE={setup_args['hf_datasets_cache']}",
                f"-B={os.getcwd()}:/sparsity,{setup_args['hf_datasets_cache']}:{setup_args['hf_datasets_cache']}",
                "--nv",
                setup_args["singularity_image"],
                "python3",
                "-m",
                setup_args["runner"],
                *runner_params,
            ]
        elif CLUSTER_NAME == MachineBackend.LOCAL:
            # We run the experiment directly, not through a grid entrypoint script
            # because we want to be able to debug it
            runner_main_function = get_train_main_function(setup_args["runner"])
            runner_main_function(None, runner_params=runner_params)
            exit(0)
        else:
            raise ValueError(f"Unknown cluster name: {CLUSTER_NAME}")
        print(f"running experiment {i} from {job_name}...")
        PROCESS_CALL_FUNCTION(subprocess_args, env)
        sleep(5)
        if interactive_debug_session:
            print("Ran only the first experiment in interactive mode. Aborting...")
            break
