from typing import Optional
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
            else:
                return

    repo.create_remote(remote_name, url=remote_url)
    print(f"Added remote '{remote_name}' with url '{remote_url}'")


def commit_pending_changes(repo: Repo):
    if len(repo.index.diff("HEAD")) > 0:
        repo.git.commit(m="Versioning code", no_verify=True)


def version_code(versioning_branch: str, repo_path: Optional[str] = None ):
    repo = Repo(repo_path, search_parent_directories=True)

    original_branch = repo.active_branch.name
    original_branch_commit_hash = repo.head.object.hexsha

    try:
        ensure_remote_config_exist(repo, REMOTE_NAME, REMOTE_URL)
        repo.git.add(all=True)
        commit_pending_changes(repo)

        repo.git.checkout(b=versioning_branch)
        repo.git.push(REMOTE_NAME, versioning_branch)
        print(
            f"Code versioned successfully to remote branch {versioning_branch} on '{REMOTE_NAME}' remote!"
        )
    finally:
        reset_to_original_repo_state(repo, original_branch, original_branch_commit_hash, versioning_branch)


def reset_to_original_repo_state(repo: Repo, original_branch: str, original_branch_commit_hash: str, versioning_branch: str):
    repo.git.checkout(original_branch, "-f")
    if versioning_branch in repo.branches:
        repo.git.branch("-D", versioning_branch)
    repo.head.reset(original_branch_commit_hash, index=True)
    print("Successfully restored working tree to the original state!")


if __name__ == "__main__":
    version_code( "to_branch_test")
