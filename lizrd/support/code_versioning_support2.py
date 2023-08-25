import shutil
import os
import subprocess
from typing import Union

from git import Repo, GitCommandError


class CodeVersioningDaemon:
    def __init__(self, remote_name, remote_url, name_for_branch):
        self.remote_name: str = remote_name
        self.remote_url: str = remote_url
        self.name_for_branch: str = name_for_branch
        self.original_branch: Union[str, None] = None
        self.current_branch: Union[str, None] = None

        self.repo: Repo = Repo(find_git_root())
        self.revert_status: int = 0
        self.stash_present: bool = False
        self.stash_message: Union[str, None] = None

    def version_code(self):
        try:
            # Record current branch
            self.original_branch = self.repo.active_branch.name

            # reject if there are unpushed commits
            commits_behind = list(
                self.repo.iter_commits(f"origin/{self.original_branch}..HEAD")
            )
            if len(commits_behind) > 0:
                raise Exception(
                    f"Unpushed commits detected. Push them first. Aborting..."
                )

            self.check_remote_cemetery()

            self.stash_if_necessary()
            self.revert_status = 1

            self.repo.git.checkout(b=self.name_for_branch)
            self.revert_status = 2

            self.repo.git.checkout("main")
            self.revert_status = 3

            self.repo.git.pull()
            self.revert_status = 4

            self.repo.git.checkout(self.name_for_branch)
            self.revert_status = 5

            self.repo.git.merge("main")  # if there are conflicts,
            # this should fail and prompt the user to merge manually and try again
            self.revert_status = 6

            self.unstash_if_necessary()
            self.revert_status = 7

            self.repo.git.push(self.remote_name, self.name_for_branch)
            self.revert_status = 8

            self.repo.git.checkout(self.original_branch)
        except GitCommandError:
            self.handle_failure()
            raise Exception("Failed to version code. Aborting...")

    def handle_failure(self):
        pass

    def check_remote_cemetery(self):
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
            self.stash_present = True
            self.stash_message = f"versioning_{self.name_for_branch}"
            try:
                self.repo.git.stash("save", "--message", self.stash_message)
            except GitCommandError:
                raise GitCommandError(
                    "Failed to stash changes. Reverting changes. If anything goes wrong, consult local history. Aborting..."
                )

    def unstash_if_necessary(self):
        if self.stash_present:
            try:
                stash_id = self.find_stash_by_message(self.stash_message)
                self.repo.git.stash("apply", stash_id)
                self.repo.git.stash("drop", stash_id)
            except subprocess.CalledProcessError as e:
                print(
                    "Error encountered while applying stashed changes.",
                    e,
                    "Trying to merge, favoring stash...",
                )
                try:
                    subprocess.run(["git", "checkout", "--theirs", "."], check=True)
                except subprocess.CalledProcessError as e:
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


def version_and_copy_code(
    newdir_name,
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
    # Create versioning daemon
    version_daemon = CodeVersioningDaemon(remote_name, remote_url, name_for_branch)
    # Version code
    version_daemon.version_code()

    # Copy code
    root_dir = find_git_root()
    newdir_path = f"{os.path.dirname(root_dir)}/sparsity_code_cemetery/{newdir_name}"

    ignore_patterns_file = os.path.join(root_dir, ".versioningignore")
    versioning_ignore_patterns = make_ignore_patterns(ignore_patterns_file)

    print(f"Copying code to {newdir_path}...")
    # Copy the project root directory to a new directory, ignoring files described in versioning_ignore_patterns
    shutil.copytree(root_dir, newdir_path, ignore=versioning_ignore_patterns)
    print(f"Code copied successfully to {newdir_path}")

    # Change to the new directory
    os.chdir(newdir_path)


def rebase_on_new_main(name_for_branch, current_branch, repo):
    # Check for changes in the current workspace
    should_unstash = False
    if repo.is_dirty():
        should_unstash = True
        # Changes exist, so let's stash them
        try:
            repo.git.stash()
        except GitCommandError:
            raise GitCommandError(
                "Failed to stash changes. Not sure what to do. Aborting..."
            )

    # try to checkout main and pull
    try:
        # Switch to the 'main' branch
        repo.git.checkout("main")
    except GitCommandError:
        if should_unstash:
            repo.git.stash("pop")
        raise GitCommandError(
            "Failed to checkout main. Make sure you have a main branch."
        )
    try:
        # Perform git pull
        repo.git.pull()
    except GitCommandError:
        assert repo.active_branch.name == "main"
        repo.git.reset("--hard", "HEAD")
        repo.git.checkout(current_branch)
        if should_unstash:
            repo.git.stash("pop")
        raise GitCommandError(
            "Failed to pull from main. Make sure you have a main branch."
        )
    try:
        repo.git.checkout(b=name_for_branch)

    except GitCommandError:
        assert repo.active_branch.name == "main"
        repo.git.reset("--hard", "HEAD")
        repo.git.checkout(current_branch)
        if should_unstash:
            repo.git.stash("pop")
        raise GitCommandError("Failed to create new branch.")

    try:
        if should_unstash:
            try:
                # Use 'subprocess' module to call git stash pop
                subprocess.run(["git", "stash", "pop"], check=True)
            except subprocess.CalledProcessError as err:
                print(
                    "Error encountered while applying stashed changes.",
                    err,
                    "Trying to merge, favoring stash...",
                )
                try:
                    subprocess.run(["git", "checkout", "--theirs", "."], check=True)
                except subprocess.CalledProcessError as err:
                    print(
                        "Error encountered while resolving conflicts in favor of stash.",
                        err,
                    )
                    raise GitCommandError(
                        "Rebasing on new main failed, conflicts occurred. Could not resolve them automatically. "
                    )

    except GitCommandError:
        # In case of conflict or any other git error, reset back to the initial state
        repo.git.reset("--hard", "HEAD")
        # Reset back to the initial recorded state
        repo.git.checkout(current_branch)
        if should_unstash:
            # Unstash the changes
            repo.git.stash("pop")
        raise GitCommandError(
            "An error occurred. There are conflicts between newest main and your local changes. Resolve them locally (e.g. by merging the newest main to your current branch) and try again. \nResetting back to initial state..."
        )


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
