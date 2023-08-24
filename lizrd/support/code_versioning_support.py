import subprocess
import shutil
import os


def get_stash_id(stash_message):
    # Get the list of stashes:
    stash_list = (
        subprocess.check_output(["git", "stash", "list"]).decode("utf-8").split("\n")
    )
    # Parse the output and check if our stash_message exists
    for stash in stash_list:
        if stash_message in stash:
            # Stash details are in form stash@{N}: On <branch_name>: <stash_message>
            # We need to get 'stash@{N}' from this.
            return stash.split(":")[0].strip()
    raise Exception("No stash found with given stash message")


def run_subprocess(command, error_message, branch_to_return_to=None):
    try:
        output = subprocess.check_output(command, shell=True).strip().decode("utf-8")
        return output
    except subprocess.CalledProcessError as e:
        if branch_to_return_to:
            # Try to revert changes and return to original branch
            try:
                subprocess.check_output(
                    f"git checkout {branch_to_return_to}", shell=True
                )
                subprocess.check_output("git reset --hard HEAD~1", shell=True)
            except subprocess.CalledProcessError as err:
                # If reverting fails, raise another exception
                raise Exception(
                    f"POTENTIALLY CHANGE-MAKING ERROR: Failed to revert changes, error occurred during {command}: {str(err)}. The staging process has been interrupted and the system did not manage to return to its original state."
                )
        raise Exception(f"{error_message}: {str(e)}")


def add_remote_if_not_present(
    remote_name: str = "cemetery", url: str = "git@github.com:Simontwice/sparsity.git"
):
    try:
        # Check if the remote exists
        remotes = (
            subprocess.check_output(["git", "remote"])
            .strip()
            .decode("utf-8")
            .split("\n")
        )

        if remote_name not in remotes:
            # Ask and ddd remote if it does not exist
            response = input(
                f"The remote {remote_name} does not exist. Do you want to add it? Y/n "
            )

            # If yes, try to add the remote
            if response.lower() in ["", "y", "yes"]:
                subprocess.check_call(["git", "remote", "add", remote_name, url])
            else:
                raise Exception(
                    f"Remote {remote_name} does not exist and was chosen not to be added. The experiment will not be run"
                )
            # Check if the URL is correct
            remote_url = (
                subprocess.check_output(["git", "remote", "get-url", remote_name])
                .decode("utf-8")
                .strip()
            )
            if remote_url != url:
                raise Exception(
                    f"Wrong url under the remote repo {remote_name}: {remote_url}. Should be: {url} instead"
                )

        # Check if the URL is correct if the remote already exists
        remote_url = (
            subprocess.check_output(["git", "remote", "get-url", remote_name])
            .decode("utf-8")
            .strip()
        )
        if remote_url != url:
            raise Exception(
                f"Wrong url under the remote repo {remote_name}: {remote_url}"
            )

    except subprocess.CalledProcessError as e:
        raise Exception("Failed to handle git remote: " + str(e))


def version_code(name_for_branch, remote_name, remote_url):
    # Record current branch
    current_branch = run_subprocess(
        "git rev-parse --abbrev-ref HEAD", "Failed to get current branch"
    )

    # Add remote if it does not exist
    add_remote_if_not_present(remote_name, remote_url)

    # Stage all changes
    run_subprocess("git add .", "Failed to stage changes")

    # Commit changes
    run_subprocess(
        f'git commit --no-verify -m "commit before experiment"', "Commit failed"
    )

    # Check out a new branch
    run_subprocess(
        f"git checkout -b {name_for_branch}",
        "Failed to create new branch",
        current_branch,
    )

    # Push changes
    run_subprocess(
        f"git push --no-verify -u {remote_name} {name_for_branch}",
        "Failed to push changes",
        current_branch,
    )

    run_subprocess(
        f"git checkout {current_branch}",
        "Failed to return to original branch",
        current_branch,
    )

    # Return to original state
    run_subprocess(
        f"git reset --hard HEAD~1", "Failed to reset changes", current_branch
    )


def version_and_copy_code(
    newdir_name,
    name_for_branch,
    remote_name="cemetery",
    remote_url="git@github.com:Simontwice/sparsity.git",
):
    """
    Stashes the current code, adds all changes, commits them, pushes them to a remote repo, and returns to the original branch.
    Then copies all code to a new directory.
    Prerequisite: the user needs to be able to push to the remote repo from the command line without entering a password.
    If not met, the user needs to set up ssh keys.

    :param name_for_branch:
    :param newdir_name:
    :param remote_name:
    :param remote_url:
    :return: None
    """
    # Version code
    version_code(name_for_branch, remote_name, remote_url)

    # Copy code

    # Find git root directory
    root_dir = find_git_root()
    newdir_path = f"{os.path.dirname(root_dir)}/sparsity_code_cemetery/{newdir_name}"

    # Set up ignore patterns
    with open(os.path.join(root_dir, ".versioningignore")) as f:
        versioning_ignore_patterns = f.read().splitlines()
        versioning_ignore_patterns = [
            pattern
            for pattern in versioning_ignore_patterns
            if pattern != "" and pattern[0] != "#"
        ]
        versioning_ignore_patterns = [p.strip() for p in versioning_ignore_patterns]
    versioning_ignore_patterns = shutil.ignore_patterns(*versioning_ignore_patterns)

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
