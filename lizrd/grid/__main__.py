"""
Script to grid search. Run this script from the root of the project:
$ python3 -m lizrd.grid --git_branch name_of_branch --config_path path/to/config.yaml
"""

import argparse
import os
from time import sleep
from lizrd.grid.grid import create_subprocess_args

from lizrd.grid.infrastructure import LocalBackend, get_machine_backend
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--git_branch", type=str, default="")
    parser.add_argument(
        "--neptune_key", type=str, default=os.environ.get("NEPTUNE_API_TOKEN")
    )
    parser.add_argument(
        "--wandb_key", type=str, default=os.environ.get("WANDB_API_KEY")
    )
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
    PROCESS_CALL_FUNCTION = lambda args, env: subprocess.run(
        [str(arg) for arg in args if arg is not None], env=env
    )
    if not isinstance(CLUSTER, LocalBackend):
        for i, experiment in enumerate(experiments):
            subprocess_args, job_name, cuda_visible = experiment
            print(f"running experiment {i} from {job_name}...")
            env = os.environ.copy()
            if cuda_visible is not None:
                env.update({"SINGULARITYENV_CUDA_VISIBLE_DEVICES": cuda_visible})

            PROCESS_CALL_FUNCTION(subprocess_args, env)
            sleep(5)
            
            if interactive_debug_session:
                print("Ran only the first experiment in interactive mode. Aborting...")
                break
        print("Successfully ran all experiments.")

    else:
        runner_main_function, runner_params = experiments[0]
        # We run the experiment directly, not through a grid entrypoint script
        # because we want to be able to debug it
        runner_main_function(None, runner_params=runner_params)
        exit(0)
