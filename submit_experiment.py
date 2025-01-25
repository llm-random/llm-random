import argparse
import os

from lizrd.grid.infrastructure import (
    resolve_get_machine_backend_function,
)
from lizrd.support.code_versioning import version_code

from contextlib import contextmanager
import copy
import getpass
from typing import Generator
from fabric import Connection
import paramiko.ssh_exception

from lizrd.support.code_versioning import version_code


CEMETERY_REPO_URL = "git@github.com:llm-random/llm-random-cemetery.git"  # TODO(crewtool) move to constants
BRANCH_FILENAME = "__branch__name__.txt"
EXPERIMENT_DIR_FILENAME = "__experiment__dir__.txt"

_SSH_HOSTS_TO_PASSPHRASES = {}

@contextmanager
def ConnectWithPassphrase(*args, **kwargs) -> Generator[Connection, None, None]:
    """Connect to a remote host using a passphrase if the key is encrypted. The passphrase is preserved for subsequent connections to the same host."""
    try:
        connection = Connection(*args, **kwargs)
        connection.run('echo "Connection successful."')
        yield connection
    except paramiko.ssh_exception.PasswordRequiredException as e:
        if connection.host not in _SSH_HOSTS_TO_PASSPHRASES:
            passphrase = getpass.getpass(
                f"SSH key encrypted, provide the passphrase ({connection.host}): "
            )
            _SSH_HOSTS_TO_PASSPHRASES[connection.host] = passphrase
        else:
            passphrase = _SSH_HOSTS_TO_PASSPHRASES[connection.host]
        kwargs["connect_kwargs"] = copy.deepcopy(
            kwargs.get("connect_kwargs", {})
        )  # avoid modifying the original connect_kwargs
        kwargs["connect_kwargs"]["passphrase"] = passphrase
        connection = Connection(*args, **kwargs)
        yield connection
    finally:
        connection.close()


def submit_experiment(
    hostname,
    experiment_branch_name,
    experiment_config_path,
    clone_only,
    save_branch_and_dir,
    custom_backends_module,
):
    if experiment_branch_name is None:
        assert custom_backends_module is None
        experiment_branch_name, custom_backends_module = version_code(
            experiment_config_path
        )

    get_machine_backend = resolve_get_machine_backend_function(custom_backends_module)

    with ConnectWithPassphrase(hostname) as connection:
        result = connection.run("uname -n", hide=True)
        node = result.stdout.strip()
        cluster = get_machine_backend(node, connection)

        cemetery_dir = cluster.get_cemetery_directory()
        connection.run(f"mkdir -p {cemetery_dir}")
        experiment_directory = f"{cemetery_dir}/{experiment_branch_name}"

        if "NEPTUNE_API_TOKEN" in os.environ:
            connection.config["run"]["env"]["NEPTUNE_API_TOKEN"] = os.environ[
                "NEPTUNE_API_TOKEN"
            ]

        if "WANDB_API_KEY" in os.environ:
            connection.config["run"]["env"]["WANDB_API_KEY"] = os.environ[
                "WANDB_API_KEY"
            ]

        if connection.run(f"test -d {experiment_directory}", warn=True).failed:
            print(f"Cloning {experiment_branch_name} to {experiment_directory}...")
            connection.run(
                f"git clone --depth 1 -b {experiment_branch_name} {CEMETERY_REPO_URL} {experiment_directory}"
            )
            print(f"Cloned.")
        else:
            print(
                f"Experiment {experiment_branch_name} already exists on {node}. Skipping."
            )

        connection.run(f"chmod +x {experiment_directory}/run_experiment.sh")
        if not clone_only:
            connection.run(f"cd {experiment_directory} && ./run_experiment.sh")

        if save_branch_and_dir:
            with open(BRANCH_FILENAME, "w") as f:
                f.write(f"{experiment_branch_name}")
            with open(EXPERIMENT_DIR_FILENAME, "w") as f:
                f.write(f"{experiment_directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        type=str,
        help="Hostname as in ~/.ssh/config",
        required=True,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--experiment",
        type=str,
        help="[Optional] Name of the existing branch on cemetery to run experiment.",
    )
    group.add_argument(
        "--config",
        type=str,
        help="[Optional] Path to experiment config file.",
    )
    parser.add_argument(
        "--clone_only",
        action="store_true",
        help="Only clone the experiment, do not run it.",
    )
    parser.add_argument(
        "--save_branch_and_dir",
        action="store_true",
        help="This flag will save the branch name and the directory to a file. This is only for `run_exp_remotely.sh` to use.",
    )
    parser.add_argument(
        "--custom_backends_module",
        type=str,
        default=None,
        help="Allows you to define custom backend module, which should contain a function `get_machine_backend` that returns a `MachineBackend` object.",
    )

    args = parser.parse_args()
    submit_experiment(
        args.host,
        args.experiment,
        args.config,
        args.clone_only,
        args.save_branch_and_dir,
        args.custom_backends_module,
    )
