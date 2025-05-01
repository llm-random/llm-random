#!/usr/bin/env python
import datetime
import logging
import os
import time
from git import Repo
from contextlib import contextmanager
import copy
import getpass
from typing import Generator, Optional
from fabric import Connection
import hydra
from omegaconf import OmegaConf
import paramiko.ssh_exception

from resolver import get_cluster_name

logger = logging.getLogger(__name__)

_SSH_HOSTS_TO_PASSPHRASES = {}


def ensure_remote_config_exist(repo: Repo, remote_name: str, remote_url: str):
    for remote in repo.remotes:
        if remote.name == remote_name:
            if remote.url != remote_url:
                old_remote_url = remote.url
                remote.set_url(remote_url)
                print(
                    f"Updated url of '{remote_name}' remote from '{old_remote_url}' to '{remote_url}'"
                )
            return

    repo.create_remote(remote_name, url=remote_url)
    print(f"Added remote '{remote_name}' with url '{remote_url}'")


def commit_pending_changes(repo: Repo):
    if len(repo.index.diff("HEAD")) > 0:
        repo.git.commit(m="Versioning code", no_verify=True)


def reset_to_original_repo_state(
    repo: Repo,
    original_branch: str,
    original_branch_commit_hash: str,
    versioning_branch: str,
):
    repo.git.checkout(original_branch, "-f")
    if versioning_branch in repo.branches:
        repo.git.branch("-D", versioning_branch)
    repo.head.reset(original_branch_commit_hash, index=True)
    print("Successfully restored working tree to the original state!")


def version_code(
    remote_name: str,
    remote_url: str,
    experiment_config_path: Optional[str] = None,
    job_name: Optional[str] = None,
) -> str:
    repo = Repo(".", search_parent_directories=True)

    experiment_branch_name = (
        f"{job_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    original_branch = repo.active_branch.name
    original_branch_commit_hash = repo.head.object.hexsha

    ensure_remote_config_exist(repo, remote_name, remote_url)
    repo.git.add(experiment_config_path, force=True)
    repo.git.add(all=True)

    try:
        commit_pending_changes(repo)

        repo.git.checkout(b=experiment_branch_name)
        print(
            f"Pushing experiment code to {experiment_branch_name} '{remote_name}' remote..."
        )
        repo.git.push(remote_name, experiment_branch_name)
        print(f"Pushed.")
    finally:
        reset_to_original_repo_state(
            repo, original_branch, original_branch_commit_hash, experiment_branch_name
        )

    return experiment_branch_name


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


def get_experiment_components(
    hydra_config: OmegaConf,
) -> str:
    # this is a workaround as hydra does not provide a way to get the config path
    # https://github.com/facebookresearch/hydra/discussions/2750
    config_name = hydra_config.job.config_name
    config_path = [
        path["path"]
        for path in hydra_config.runtime.config_sources
        if path["schema"] == "file"
    ][0]
    return config_path, config_name


@hydra.main(version_base=None, config_path=".", config_name="experiment")
def submit_experiment(
    cfg: OmegaConf,
):
    hydra_config = hydra.utils.HydraConfig.get()
    config_path, config_name = get_experiment_components(hydra_config)
    experiment_config_path = f"{config_path}/{config_name}.yaml"

    experiment_branch_name = version_code(
        cfg.git.remote_name,
        cfg.git.remote_url,
        experiment_config_path,
        hydra_config.job.name,
    )

    missing_keys: set[str] = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Got missing keys in config:\n{missing_keys}")

    with ConnectWithPassphrase(host=cfg.server, inline_ssh_env=True) as connection:
        result = connection.run("uname -n", hide=True)
        hostname = result.stdout.strip()
        username = connection.user

        cluster_name = get_cluster_name(hostname, username)

        cluster_config = OmegaConf.load(f"configs/clusters/{cluster_name}.yaml")

        cemetery_dir = cluster_config.cemetery_experiments_dir
        connection.run(f"mkdir -p {cemetery_dir}")

        if "NEPTUNE_API_TOKEN" in os.environ:
            connection.config["run"]["env"]["NEPTUNE_API_TOKEN"] = os.environ[
                "NEPTUNE_API_TOKEN"
            ]

        if "WANDB_API_KEY" in os.environ:
            connection.config["run"]["env"]["WANDB_API_KEY"] = os.environ[
                "WANDB_API_KEY"
            ]

        experiment_dir = f"{cemetery_dir}/{experiment_branch_name}"
        if connection.run(f"test -d {experiment_dir}", warn=True).failed:
            print(f"Cloning {experiment_branch_name} to {experiment_dir}...")
            connection.run(
                f"git clone --depth 1 -b {experiment_branch_name} {cfg.git.remote_url} {experiment_dir}"
            )
            print(f"Cloned.")
        else:
            print(
                f"Experiment {experiment_branch_name} already exists on {hostname}. Skipping."
            )

        try:
            connection.run(f"tmux new -d -s {experiment_branch_name}")
            connection.run(
                f'tmux send -t {experiment_branch_name}.0 "cd {experiment_dir}/nano" ENTER'
            )
            connection.run(
                f'tmux send -t {experiment_branch_name}.0 "source {cfg.experiment_prepare_venv_path}" ENTER'
            )
            pwd = os.getcwd()
            relative_path = os.path.relpath(config_path, pwd)
            connection.run(
                f'tmux send -t {experiment_branch_name}.0 "python main.py --config-path={relative_path} --config-name={config_name}" ENTER'
            )
            connection.run(
                f'tmux send -t {experiment_branch_name}.0 "sbatch exp.job" ENTER'
            )
            logger.info("=" * 38 + "TMUX" + "=" * 38)
            time.sleep(3)
            output = connection.run(
                f"tmux capture-pane -t {experiment_branch_name}.0 -p", hide=True
            ).stdout
            logger.info(output)
        except Exception as e:
            print("Exception while running an experiment: ", e)


if __name__ == "__main__":
    submit_experiment()
