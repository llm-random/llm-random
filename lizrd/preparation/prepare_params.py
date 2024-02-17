import pprint
from lizrd.hostname_setup.utils import maybe_set_default_datasets_paths
from lizrd.scripts.grid_utils import (
    create_grid,
    get_setup_args_with_defaults,
    multiply_grid,
    timestr_to_minutes,
)
from lizrd.support.misc import load_with_inheritance


class PrepareParams:
    def __init__(self, config_path, git_branch):
        self.config_path = config_path
        self.git_branch = git_branch

    def prepare_configs(self, CLUSTER_NAME):
        configs, all_config_paths = load_with_inheritance(self.config_path)

        for config in configs:
            config["params"]["git_branch"] = self.git_branch
            config["params"]["path_to_entry_config"] = self.config_path
            config["params"]["all_config_paths"] = ",".join(
                sorted(list(all_config_paths))
            )

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

            grid += list(
                zip(single_exp_training_args_grid, [setup_args] * n_experiments)
            )

            total_n_experiments += n_experiments
            minutes_per_exp = timestr_to_minutes(setup_args["time"])
            total_minutes_from_this_grid = n_experiments * minutes_per_exp
            total_minutes += total_minutes_from_this_grid

        for _, (training_args, setup_args) in enumerate(grid):
            training_args["n_gpus"] = setup_args["n_gpus"]
        maybe_set_default_datasets_paths(grid, CLUSTER_NAME)

        return grid, total_n_experiments, total_minutes, interactive_debug_session
