import argparse
import datetime
import os
from typing import Optional, Tuple
from git import Repo

from lizrd.grid.prepare_configs import load_with_inheritance

REMOTE_NAME = "cemetery"  # TODO(crewtool) move to constants file
REMOTE_URL = "git@github.com:llm-random/llm-random-cemetery.git"  # TODO(crewtool) move to constants


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


def create_run_experiment_script(
    experiment_config_path,
    experiment_branch_name,
    file_path,
    custom_backends_module,
):
    script_text = """#!/bin/bash
python3 -m lizrd.grid --config_path={} --git_branch={} --skip_copy_code""".format(
        experiment_config_path, experiment_branch_name
    )
    if custom_backends_module is not None:
        script_text += f" --custom_backends_module={custom_backends_module}"

    with open(file_path, "w") as f:
        f.write(script_text)


def delete_run_experiment_script(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def version_code(
    experiment_config_path: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    repo = Repo(".", search_parent_directories=True)
    configs, paths_to_all_configs = load_with_inheritance(experiment_config_path)
    job_name = configs[0]["params"]["name"]
    custom_backends_module = configs[0].get("backends_module")
    assert custom_backends_module is None or isinstance(custom_backends_module, str)
    experiment_branch_name = (
        f"{job_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    script_run_experiment_path = os.path.join(
        repo.working_tree_dir, "run_experiment.sh"
    )

    if experiment_config_path is not None:
        create_run_experiment_script(
            experiment_config_path=experiment_config_path,
            experiment_branch_name=experiment_branch_name,
            file_path=script_run_experiment_path,
            custom_backends_module=custom_backends_module,
        )

    original_branch = repo.active_branch.name
    original_branch_commit_hash = repo.head.object.hexsha

    try:
        ensure_remote_config_exist(repo, REMOTE_NAME, REMOTE_URL)

        repo.git.add(all=True)
        if paths_to_all_configs is not None:
            repo.git.add(paths_to_all_configs, force=True)
        commit_pending_changes(repo)

        repo.git.checkout(b=experiment_branch_name)
        print(
            f"Pushing experiment code to {experiment_branch_name} '{REMOTE_NAME}' remote..."
        )
        repo.git.push(REMOTE_NAME, experiment_branch_name)
        print(f"Pushed.")
    finally:
        reset_to_original_repo_state(
            repo, original_branch, original_branch_commit_hash, experiment_branch_name
        )
        if experiment_config_path is not None:
            delete_run_experiment_script(script_run_experiment_path)
    return experiment_branch_name, custom_backends_module


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to experiment config file.",
        required=True,
    )
    args = parser.parse_args()
    _ = version_code(args.config)
