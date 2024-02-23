"""
Script to grid search in recycle layers. Run this script from the root of the project:
$ python3 research/reinitialization/scripts/grid.py
Remember to set RUNNER and PARAMS in the script or add an argument parser.
"""

import argparse
import datetime
import subprocess
from time import sleep
from lizrd.grid.infrastructure import get_machine_backend, LocalBackend
from lizrd.grid.prepare_configs import prepare_configs
from lizrd.grid.setup_arguments import (
    make_singularity_env_arguments,
)

from lizrd.scripts.grid_utils import (
    get_train_main_function,
    timestr_to_minutes,
    translate_to_argparse,
    check_for_argparse_correctness,
)
from lizrd.scripts.grid_utils import setup_experiments
from lizrd.support.code_copying import copy_code
import yaml


def calculate_experiments_info(grid):
    total_minutes = 0
    total_n_experiments = 0
    for setup_args, training_list in grid:
        minutes_per_exp = timestr_to_minutes(setup_args["time"])
        n_experiments = len(training_list)
        total_n_experiments += n_experiments
        total_minutes = n_experiments * minutes_per_exp
    return total_minutes, total_n_experiments


def create_subprocess_args(
    config_path,
    git_branch,
    neptune_key,
    wandb_key,
    CLUSTER,
    skip_confirmation=False,
    skip_copy_code=False,
):
    configs = prepare_configs(config_path, git_branch, CLUSTER)
    grid = setup_experiments(configs)
    check_for_argparse_correctness(grid)
    interactive_debug_session = grid[0][0]["interactive_debug_session"]

    if not isinstance(CLUSTER, LocalBackend) and not skip_confirmation:
        if not interactive_debug_session:
            total_minutes, total_n_experiments = calculate_experiments_info(grid)
            user_input = input(
                f"Will run {total_n_experiments} experiments, using up {total_minutes} minutes, i.e. around {round(total_minutes / 60)} hours\n"
                f"Continue? [Y/n]"
            )
        else:
            user_input = input(
                "Will run an INTERACTIVE experiment, which will be the first one from the supplied configs. \nContinue? [Y/n]"
            )
        if user_input.lower() not in ("", "y", "Y"):
            print("Aborting...")
            exit(1)

    if not isinstance(CLUSTER, LocalBackend) and (not skip_copy_code):
        _, first_exp_trainings_args = grid[0]
        exp_name = first_exp_trainings_args[0]["name"]
        newdir_name = (
            f"{exp_name}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        )
        copy_code(newdir_name)
    else:
        print("Skip copying code to a new directory.")

    slurm_command = "srun" if interactive_debug_session else "sbatch"
    experiments = []
    for setup_args, trainings_args in grid:
        for i, training_args in enumerate(trainings_args):
            full_config_path = f"full_config{i}.yaml"
            with open(full_config_path, "w") as f:
                yaml.dump({**training_args, **setup_args}, f)
            training_args["all_config_paths"] += f",{full_config_path}"

            singularity_env_arguments = make_singularity_env_arguments(
                hf_datasets_cache_path=setup_args["hf_datasets_cache"],
                neptune_key=neptune_key,
                wandb_key=wandb_key,
            )

            runner_params = translate_to_argparse(training_args)
            if isinstance(CLUSTER, LocalBackend):
                runner_main_function = get_train_main_function(setup_args["runner"])
                return [
                    (runner_main_function, runner_params)
                ], interactive_debug_session

            subprocess_args = CLUSTER.get_subprocess_args(
                slurm_command=slurm_command,
                setup_args=setup_args,
                training_args=training_args,
                singularity_env_arguments=singularity_env_arguments,
                runner_params=runner_params,
            )

            experiments.append((subprocess_args, training_args["name"]))
    return experiments, interactive_debug_session


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--git_branch", type=str, default="")
    parser.add_argument("--neptune_key", type=str, default=None)
    parser.add_argument("--wandb_key", type=str, default=None)
    parser.add_argument("--skip_confirmation", action="store_true")
    parser.add_argument("--skip_copy_code", action="store_true")
    args = parser.parse_args()
    CLUSTER = get_machine_backend()
    experiments, interactive_debug_session = create_subprocess_args(
        args.config_path,
        args.git_branch,
        args.neptune_key,
        args.wandb_key,
        CLUSTER,
        args.skip_confirmation,
        args.skip_copy_code,
    )
    PROCESS_CALL_FUNCTION = lambda args: subprocess.run(
        [str(arg) for arg in args if arg is not None]
    )
    if not isinstance(CLUSTER, LocalBackend):
        for i, experiment in enumerate(experiments):
            subprocess_args, job_name = experiment
            print(f"running experiment {i} from {job_name}...")
            PROCESS_CALL_FUNCTION(subprocess_args)
            sleep(5)
            if interactive_debug_session:
                print("Ran only the first experiment in interactive mode. Aborting...")
                break

    else:
        runner_main_function, runner_params = experiments[0]
        # We run the experiment directly, not through a grid entrypoint script
        # because we want to be able to debug it
        runner_main_function(None, runner_params=runner_params)
        exit(0)
