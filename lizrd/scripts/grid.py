"""
Script to grid search in recycle layers. Run this script from the root of the project:
$ python3 research/reinitialization/scripts/grid.py
Remember to set RUNNER and PARAMS in the script or add an argument parser.
"""

import argparse
from time import sleep
from lizrd.hostname_setup.hostname_setup import get_subprocess_args
from lizrd.hostname_setup.utils import (
    MachineBackend,
    get_machine_backend,
)
from lizrd.preparation.prepare_params import PrepareParams

from lizrd.scripts.grid_utils import (
    get_train_main_function,
    translate_to_argparse,
)
from lizrd.submitter.job_submitter import JobSubmitter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--git_branch", type=str, default="")
    parser.add_argument("--neptune_key", type=str, default=None)
    parser.add_argument("--wandb_key", type=str, default=None)
    parser.add_argument("--run_locally", action="store_true")
    args = parser.parse_args()
    CLUSTER_NAME = get_machine_backend()
    CLUSTER_NAME = MachineBackend.IDEAS
    run_locally = args.run_locally or CLUSTER_NAME == MachineBackend.LOCAL

    preparator = PrepareParams(args.config_path, args.git_branch)
    (
        grid,
        total_n_experiments,
        total_minutes,
        interactive_debug_session,
    ) = preparator.prepare_configs(CLUSTER_NAME)

    slurm_command = "srun" if interactive_debug_session else "sbatch"

    for i, (training_args, setup_args) in enumerate(grid):
        if run_locally:
            # We run the experiment directly, not through a grid entrypoint script
            # because we want to be able to debug it
            runner_params = translate_to_argparse(training_args)
            runner_main_function = get_train_main_function(setup_args["runner"])
            runner_main_function(None, runner_params=runner_params)
        else:
            print(f"running experiment {i} from {training_args['name']}...")
            subprocess_args = get_subprocess_args(
                CLUSTER_NAME,
                slurm_command,
                setup_args,
                training_args,
                args.neptune_key,
                args.wandb_key,
            )
            job_submitter = JobSubmitter(
                setup_args,
                training_args["name"],
                run_locally,
                total_n_experiments,
                total_minutes,
                interactive_debug_session,
            )
            job_submitter.prepare()
            job_submitter.submit(subprocess_args, training_args)

        if interactive_debug_session or run_locally:
            print(
                "Ran only the first experiment in (interactive mode or local run). Aborting..."
            )
            break
