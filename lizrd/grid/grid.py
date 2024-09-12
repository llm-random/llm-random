import datetime
import pathlib
from lizrd.grid.infrastructure import LocalBackend
from lizrd.grid.prepare_configs import prepare_configs
from lizrd.grid.setup_arguments import (
    make_singularity_env_arguments,
)

from lizrd.grid.utils import (
    get_train_main_function,
    timestr_to_minutes,
    translate_to_argparse,
    check_for_argparse_correctness,
)
from lizrd.grid.utils import setup_experiments
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
                if setup_args["repeater_mode"]: #dev check
                    training_args["save_weights_path"] = str(pathlib.Path(training_args["save_weights_path"])/f"{i}")
                    training_args["load_weights_path"] = str(pathlib.Path(training_args["load_weights_path"])/f"{i}")
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
            cuda_visible = setup_args.get("cuda_visible")
            experiments.append((subprocess_args, training_args["name"], cuda_visible))
    return experiments, interactive_debug_session
