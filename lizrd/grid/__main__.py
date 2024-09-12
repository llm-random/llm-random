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

            # if '--repeater_mode' in subprocess_args:
            #     arg_time_to_replace = None
            #     arg_time_to_replace_with = None
            #     total_exp_time = None
            #     for arg in subprocess_args:
            #         if "--time=" in arg:
            #             max_exp_time = arg.replace("--time=", "")
            #             hours, minutes, seconds = map(int, max_exp_time.split(':'))
            #             total_exp_time = hours*60*60 + minutes*60 + seconds
            #             arg_time_to_replace = arg
            #             break
               
            #     if CLUSTER.max_exp_time < total_exp_time:
            #         n_job_repetitions = int(total_exp_time/CLUSTER.max_exp_time)

            #         arg_time_to_replace_with = f"--time={int(CLUSTER.max_exp_time/(60*60))}:{int((CLUSTER.max_exp_time%(60*60))/60)}:{int(CLUSTER.max_exp_time%60)}"
            #         subprocess_args.remove(arg_time_to_replace)
            #         subprocess_args.append(arg_time_to_replace_with)
            #         subprocess_args.append(f"--array=0-{n_job_repetitions-1}%1")                

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
