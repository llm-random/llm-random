import argparse
import os
from typing import Optional, List
from git import Repo

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
    experiment_config_path, experiment_branch_name, file_path
):
    script_text = f"#!/bin/bash\npython3 -m lizrd.grid.grid --config_path={experiment_config_path} --git_branch={experiment_branch_name} --skip_copy_code"
    with open(file_path, "w") as f:
        f.write(script_text)


def delete_run_experiment_script(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def version_code(
    versioning_branch: str,
    experiment_config_path: Optional[str] = None,
    files_to_force_add: Optional[List[str]] = None,
    repo_path: Optional[str] = None,
):
    repo = Repo(repo_path, search_parent_directories=True)
    script_run_experiment_path = os.path.join(
        repo.working_tree_dir, "run_experiment.sh"
    )

    if experiment_config_path is not None:
        create_run_experiment_script(
            experiment_config_path, versioning_branch, script_run_experiment_path
        )

    original_branch = repo.active_branch.name
    original_branch_commit_hash = repo.head.object.hexsha

    try:
        ensure_remote_config_exist(repo, REMOTE_NAME, REMOTE_URL)

        repo.git.add(all=True)
        if experiment_config_path is not None:
            repo.git.add(experiment_config_path, force=True)
        commit_pending_changes(repo)

        repo.git.checkout(b=versioning_branch)
        repo.git.push(REMOTE_NAME, versioning_branch)
        print(
            f"Code versioned successfully to remote branch {versioning_branch} on '{REMOTE_NAME}' remote!"
        )
    finally:
        reset_to_original_repo_state(
            repo, original_branch, original_branch_commit_hash, versioning_branch
        )
        if experiment_config_path is not None:
            delete_run_experiment_script(script_run_experiment_path)


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
        "--branch", type=str, help="Name of a branch to version code", required=True
    )
    parser.add_argument(
        "--experiment_config",
        type=str,
        help="Path to experiment config file. [Optional]",
        required=False,
    )
    parser.add_argument(
        "--repository_path",
        type=str,
        help="Path of the repository which we want to version. If unspecified current working directory will be used. [Very, very rare cases.]",
        required=False,
    )
    args = parser.parse_args()
    version_code(args.branch, args.experiment_config, args.repository_path)
