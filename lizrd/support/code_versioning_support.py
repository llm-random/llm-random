import shutil
import os
import subprocess
from typing import Union


class CodeVersioningDaemon:
    def __init__(self, remote_name, remote_url, name_for_branch):
        self.remote_name: str = remote_name
        self.remote_url: str = remote_url
        self.name_for_branch: str = name_for_branch
        self.original_branch: Union[str, None] = None
        self.current_branch: Union[str, None] = None
        self.original_branch_commit_hash: Union[str, None] = None

        self.repo: Repo = Repo(find_git_root())
        self.revert_status: int = 0
        self.stash_present: bool = False
        self.stash_message: Union[str, None] = None

    def version_code(self):
        """
        Versions the code by stashing uncommited changes, creating a branch copy that is up to date with main, unstashing them and pushing it to a remote repo.
        """
        try:
            # Record current branch
            self.original_branch = self.repo.active_branch.name
            self.original_branch_commit_hash = self.repo.head.object.hexsha

            # reject if there are unpushed commits
            commits_behind = list(
                self.repo.iter_commits(f"origin/{self.original_branch}..HEAD")
            )
            if len(commits_behind) > 0:
                raise Exception(
                    f"Unpushed commits detected. Push them first. Aborting..."
                )

            self.check_if_cemetery_exists()

            self.stash_if_necessary()
            self.revert_status = 1

            self.repo.git.checkout(b=self.name_for_branch)
            self.revert_status = 2

            self.repo.git.checkout("main", "-f")
            self.revert_status = 3

            self.repo.git.clean("-f", "-d")
            self.repo.git.pull()
            self.revert_status = 4

            self.repo.git.checkout(self.name_for_branch)
            self.revert_status = 5

            self.repo.git.merge("main")
            self.revert_status = 6

            self.unstash_if_necessary()
            self.revert_status = 7

            self.repo.git.add(u=True)
            self.repo.git.commit(m="Versioning code", no_verify=True)
            self.revert_status = 7.5

            self.repo.git.push(self.remote_name, self.name_for_branch)
            self.revert_status = 8

            self.repo.git.checkout(self.original_branch)
            self.revert_status = 9

            self.unstash_if_necessary()
            self.repo.git.branch("-D", self.name_for_branch)

        except GitCommandError:
            self.handle_failure()
            raise Exception("Failed to version code. Aborting...")

    def handle_failure(self):
        if self.revert_status == 0:
            pass
        elif self.revert_status == 1:
            self.unstash_if_necessary()
        else:
            if self.revert_status == 4:
                self.repo.git.checkout("main", "-f")
                self.repo.git.reset("--hard", "HEAD")
            elif self.revert_status == 6:
                print(
                    f"[CHANGES NEEDED AT USER'S SIDE] Merging of latest main branch into current branch failed. Please merge manually and try again."
                )
            elif self.revert_status == 7.5:
                self.repo.git.reset("--hard", "HEAD~1")
            self.clean_up_new_branch()
            self.reset_to_original_branch_and_commit()

    def clean_up_new_branch(self):
        self.repo.git.checkout("main", "-f")
        self.repo.git.branch("-D", self.name_for_branch)

    def reset_to_original_branch_and_commit(self):
        self.repo.git.checkout(self.original_branch, "-f")
        self.repo.head.reset(
            self.original_branch_commit_hash, index=True, working_tree=True
        )
        self.unstash_if_necessary()

    def check_if_cemetery_exists(self):
        for remote in self.repo.remotes:
            if remote.name == self.remote_name:
                if remote.url == self.remote_url:
                    return
                else:
                    raise Exception(
                        f"Wrong url under remote repo {self.remote_name}: {self.repo.remotes[self.remote_name].url.strip()}, should be {self.remote_url}"
                    )
        self.repo.create_remote(self.remote_name, url=self.remote_url)
        return

    def stash_if_necessary(self):
        if self.repo.is_dirty():
            self.stash_message = f"versioning_{self.name_for_branch}"
            try:
                self.repo.git.stash("save", "--message", self.stash_message)
                self.stash_present = True
            except GitCommandError:
                raise GitCommandError(
                    "Failed to stash changes. Reverting changes. If anything goes wrong, consult local history. Aborting..."
                )

    def unstash_if_necessary(self):
        if self.stash_present:
            try:
                stash_id = self.find_stash_by_message(self.stash_message)
                self.repo.git.stash("apply", stash_id)
            except GitCommandError as e:
                print(
                    "Error encountered while applying stashed changes.",
                    e,
                    "Trying to merge, favoring stash...",
                )
                try:
                    subprocess.run(["git", "checkout", "--theirs", "."], check=True)
                except GitCommandError as e:
                    print(
                        "Error encountered while resolving conflicts in favor of stash.",
                        e,
                    )
                    raise GitCommandError(
                        "Unstashing changes on rebased branch failed; conflicts occurred. Could not resolve them automatically."
                    )

    def find_stash_by_message(self, stash_message):
        result = subprocess.run(["git", "stash", "list"], stdout=subprocess.PIPE)
        stashes = result.stdout.decode("utf-8").split("\n")

        for stash in stashes:
            if stash_message in stash:
                return stash.split(":")[0]  # return stash name (ex: stash@{0})
        raise GitCommandError(f"Could not find stash with message {stash_message}")


def version_code(
    code_path,
    name_for_branch,
    remote_name="cemetery",
    remote_url="git@github.com:Simontwice/llm-random-cemetery.git",
):
    """
    Stashes the current code, adds all changes, commits them, pushes them to a remote repo, and returns to the original branch.
    Then copies all code to a new directory.
    Prerequisite: the user needs to be able to push to the remote repo from the command line without entering a password.
    If not met, the user needs to set up ssh keys.
    """
    os.chdir(code_path)
    # Create versioning daemon
    version_daemon = CodeVersioningDaemon(remote_name, remote_url, name_for_branch)
    # Version code
    version_daemon.version_code()
    print(
        f"Versioned successfully to git branch {name_for_branch} on remote repo {remote_name}"
    )


def copy_code(newdir_name):
    # Copy code
    root_dir = find_git_root()
    newdir_path = f"{os.path.dirname(root_dir)}/code_cemetery_sparsity/{newdir_name}"
    ignore_patterns_file = os.path.join(root_dir, ".versioningignore")
    versioning_ignore_patterns = make_ignore_patterns(ignore_patterns_file)
    print(f"Copying code to {newdir_path}...")
    # Copy the project root directory to a new directory, ignoring files described in versioning_ignore_patterns
    shutil.copytree(root_dir, newdir_path, ignore=versioning_ignore_patterns)
    print(f"Code copied successfully to {newdir_path}")
    # Change to the new directory
    os.chdir(newdir_path)


def find_git_root():
    current_dir = os.getcwd()
    while True:
        git_dir = os.path.join(current_dir, ".git")
        if os.path.exists(git_dir):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise Exception("Not in a Git repository.")
        current_dir = parent_dir


def make_ignore_patterns(filepath):
    # Set up ignore patterns
    with open(filepath) as f:
        patterns = f.read().splitlines()
        patterns = [
            pattern for pattern in patterns if pattern != "" and pattern[0] != "#"
        ]
        patterns = [p.strip() for p in patterns]
        patterns = shutil.ignore_patterns(*patterns)
    return patterns


if __name__ == "__main__":
    import argparse
    from git import Repo, GitCommandError

    parser = argparse.ArgumentParser()
    parser.add_argument("code_path", type=str, default="")
    parser.add_argument("name_for_branch", type=str, default="")
    args = parser.parse_args()
    version_code(code_path=args.code_path, name_for_branch=args.name_for_branch)
