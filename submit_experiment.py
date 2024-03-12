import argparse
import datetime
import os
from lizrd.grid.infrastructure import get_machine_backend
from lizrd.grid.prepare_configs import load_with_inheritance
from lizrd.support.code_versioning import version_code

from contextlib import contextmanager
import copy
import getpass
from typing import Generator
from fabric import Connection
import paramiko.ssh_exception

from lizrd.support.code_versioning import version_code

CEMETERY_REPO_URL = "git@github.com:llm-random/llm-random-cemetery.git"  # TODO(crewtool) move to constants

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


def submit_experiment(hostname, experiment_branch_name, experiment_config_path):
    if experiment_branch_name is None:
        configs, paths_to_all_configs = load_with_inheritance(experiment_config_path)
        job_name = configs[0]["params"]["name"]
        experiment_branch_name = (
            f"{job_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        version_code(
            experiment_branch_name,
            experiment_config_path,
            files_to_force_add=paths_to_all_configs,
        )

    with ConnectWithPassphrase(hostname) as connection:
        result = connection.run("uname -n", hide=True)
        node = result.stdout.strip()
        cluster = get_machine_backend(node)

        cemetery_dir = cluster.get_cemetery_directory()
        connection.run(f"mkdir -p {cemetery_dir}")
        experiment_directory = f"{cemetery_dir}/{experiment_branch_name}"

        if "NEPTUNE_API_TOKEN" in os.environ:
            connection.config["run"]["env"] = {
                "NEPTUNE_API_TOKEN": os.environ["NEPTUNE_API_TOKEN"]
            }
        if "WANDB_API_KEY" in os.environ:
            connection.config["run"]["env"] = {
                "WANDB_API_KEY": os.environ["WANDB_API_KEY"]
            }

        connection.run(
            f"git clone --depth 1 -b {experiment_branch_name} {CEMETERY_REPO_URL} {experiment_directory}"
        )
        print(f"Cloned {experiment_branch_name} to {experiment_directory}")
        connection.run(f"chmod +x {experiment_directory}/run_experiment.sh")
        connection.run(f"cd {experiment_directory} && ./run_experiment.sh")


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
    args = parser.parse_args()
    submit_experiment(args.host, args.experiment, args.config)
