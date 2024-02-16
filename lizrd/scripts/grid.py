"""
Script to grid search in recycle layers. Run this script from the root of the project:
$ python3 research/reinitialization/scripts/grid.py
Remember to set RUNNER and PARAMS in the script or add an argument parser.
"""

import argparse
import datetime
import pprint
import subprocess
import yaml
from time import sleep
from lizrd.hostname_setup.hostname_setup import get_subprocess_args
from lizrd.hostname_setup.utils import (
    MachineBackend,
    get_machine_backend,
    maybe_set_default_datasets_paths,
)

from lizrd.scripts.grid_utils import (
    create_grid,
    get_train_main_function,
    multiply_grid,
    timestr_to_minutes,
    get_setup_args_with_defaults,
    translate_to_argparse,
    check_for_argparse_correctness,
)
from lizrd.support.code_copying import copy_code
from lizrd.support.misc import load_with_inheritance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--git_branch", type=str, default="")
    parser.add_argument("--neptune_key", type=str, default=None)
    parser.add_argument("--wandb_key", type=str, default=None)
    args = parser.parse_args()
    CLUSTER_NAME = get_machine_backend()

    if args.config_path.endswith(".yaml"):
        configs, all_config_paths = load_with_inheritance(args.config_path)
    else:
        raise ValueError("config path point to a .yaml")

    for config in configs:
        config["params"]["git_branch"] = args.git_branch
        config["params"]["path_to_entry_config"] = args.config_path
        config["params"]["all_config_paths"] = ",".join(all_config_paths)

    interactive_options_per_config = [
        config.get("interactive_debug", False) for config in configs
    ]

    assert (
        len(set(interactive_options_per_config)) == 1
    ), "`interactive_debug` must be the same for all configs"

    interactive_debug_session = interactive_options_per_config[0]

    # list of pairs: a dictionary of training_args and a dictionary of setup_args
    grid = []
    total_n_experiments = 0
    total_minutes = 0

    for i, config in enumerate(configs):
        print(f"\nProcessing config {i}...")
        pprint.pprint(config)
        single_exp_training_args_grid = create_grid(config["params"])

        setup_args = get_setup_args_with_defaults(config, CLUSTER_NAME)
        single_exp_training_args_grid = multiply_grid(
            single_exp_training_args_grid, setup_args["runs_multiplier"]
        )
        n_experiments = len(single_exp_training_args_grid)

        grid += list(zip(single_exp_training_args_grid, [setup_args] * n_experiments))

        total_n_experiments += n_experiments
        minutes_per_exp = timestr_to_minutes(setup_args["time"])
        total_minutes_from_this_grid = n_experiments * minutes_per_exp
        total_minutes += total_minutes_from_this_grid

    check_for_argparse_correctness(grid)

    if not CLUSTER_NAME == MachineBackend.LOCAL:
        if not interactive_debug_session:
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

    if CLUSTER_NAME != MachineBackend.LOCAL:
        first_exp_training_args, _ = grid[0]
        exp_name = first_exp_training_args["name"]
        newdir_name = (
            f"{exp_name}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        )
        copy_code(newdir_name)
    else:
        print("Running locally, skip copying code to a new directory.")

    slurm_command = "srun" if interactive_debug_session else "sbatch"
    maybe_set_default_datasets_paths(grid, CLUSTER_NAME)

    for i, (training_args, setup_args) in enumerate(grid):
        full_config_path = f"full_config{i}.yaml"
        with open(full_config_path, "w") as f:
            yaml.dump({**training_args, **setup_args}, f)
        training_args["all_config_paths"] += f",{full_config_path}"

        job_name = training_args["name"]
        training_args["n_gpus"] = setup_args["n_gpus"]

        env = None
        runner_params = translate_to_argparse(training_args)

        if CLUSTER_NAME == MachineBackend.LOCAL:
            # We run the experiment directly, not through a grid entrypoint script
            # because we want to be able to debug it
            runner_main_function = get_train_main_function(setup_args["runner"])
            runner_main_function(None, runner_params=runner_params)
            exit(0)

        print(f"running experiment {i} from {job_name}...")
        subprocess_args = get_subprocess_args(
            CLUSTER_NAME,
            slurm_command,
            setup_args,
            training_args,
            args.neptune_key,
            args.wandb_key,
        )

        PROCESS_CALL_FUNCTION = lambda args, env: subprocess.run(
            [str(arg) for arg in args if arg is not None], env=env
        )
        PROCESS_CALL_FUNCTION(subprocess_args, env)

        sleep(5)
        if interactive_debug_session or CLUSTER_NAME == MachineBackend.LOCAL:
            print(
                "Ran only the first experiment in (interactive mode or local run). Aborting..."
            )
            break
