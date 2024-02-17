import datetime
import subprocess
from lizrd.scripts.grid_utils import translate_to_argparse
from lizrd.support.code_copying import copy_code


class JobSubmitter:
    def __init__(
        self,
        setup_args,
        experiment_name,
        use_local,
        total_n_experiments,
        total_minutes,
        interactive_debug_session=False,
    ):
        self.setup_args = setup_args
        self.use_local = use_local
        self.interactive_debug_session = interactive_debug_session
        self.total_n_experiments = total_n_experiments
        self.total_minutes = total_minutes
        self.exp_directory = self.create_exp_directory(experiment_name)

    def create_exp_directory(self, experiment_name):
        return (
            f"{experiment_name}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        )

    def prepare(self):
        if not self.use_local:
            interactive_session_msg = f"Will run {self.total_n_experiments} experiments, using up {self.total_minutes} minutes, i.e. around {round(self.total_minutes / 60)} hours\n"
            normal_session_msg = "Will run an INTERACTIVE experiment, which will be the first one from the supplied configs. \n"
            message = f"{interactive_session_msg if self.interactive_debug_session else normal_session_msg}\nContinue? [Y/n]"

            user_input = input(message)
            if user_input.lower() not in ("", "y", "Y"):
                print("Aborting...")
                exit(1)

            copy_code(self.exp_directory)
        else:
            print("Running locally, no preparation needed.")

    def submit(self, subprocess_args, training_args):
        PROCESS_CALL_FUNCTION = lambda args, env: subprocess.run(
            [str(arg) for arg in args if arg is not None], env=env
        )
        run_arguments = subprocess_args
        run_arguments += ["python", "-m", self.setup_args["runner"]]
        runner_params = translate_to_argparse(training_args)
        run_arguments += runner_params
        PROCESS_CALL_FUNCTION(run_arguments, None)
