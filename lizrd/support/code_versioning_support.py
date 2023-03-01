import re
import subprocess
import shutil
import os


def version_code(
    branch_name, newdir_name, remote_url="git@github.com:Simontwice/sparsity.git"
):
    """Copies the current code to a new directory, and pushes the code to a remote repo.
    NOTE: it is assumed that this function is called from inside the project.
    Prerequisite: the user needs to be able to push to the remote repo from the command line without entering a password.
    If not met, the user needs to set up ssh keys."""

    # Find git root directory
    root_dir = find_git_root()
    if "sparsity_code_cemetery" in root_dir:
        newdir_path = f"{os.path.dirname(root_dir)}/{newdir_name}"
    else:
        newdir_path = (
            f"{os.path.dirname(root_dir)}/sparsity_code_cemetery/{newdir_name}"
        )

    # Set up ignore patterns
    with open(os.path.join(root_dir, ".versioningignore")) as f:
        gitignore_patterns = f.read().splitlines()
        gitignore_patterns = [
            pattern
            for pattern in gitignore_patterns
            if pattern != "" and pattern[0] != "#"
        ]
        gitignore_patterns = [p.strip() for p in gitignore_patterns]
    gitignore_patterns = shutil.ignore_patterns(*gitignore_patterns)

    # Copy the project root directory to a new directory, ignoring files described in .gitignore
    shutil.copytree(root_dir, newdir_path, ignore=gitignore_patterns)

    # Change to the new directory
    os.chdir(newdir_path)

    # Push the code to the remote repo
    push_code_to_url(branch_name, remote_url)


def push_code_to_url(
    branch_name,
    remote_url,
):
    remote_url = remote_url.strip()

    # Check if remote_url is already among remote repos
    check_remote = subprocess.run(
        ["git", "remote", "-v"], capture_output=True, text=True
    )
    check_remote_output = check_remote.stdout
    remote_present = False
    lines = check_remote_output.split("\n")
    remote_name = "code_image_cemetery"
    for line in lines:
        line = line.strip()
        if line == "":
            continue
        nickname, url, _ = re.split("[\t ]", line)
        if url == remote_url:
            remote_name = nickname
            remote_present = True
            break

    if not remote_present:
        # Add the repo as a remote
        subprocess.run(
            ["git", "remote", "add", "code_image_cemetery", remote_url],
            capture_output=True,
            text=True,
        )
    # Create a new branch
    subprocess.run(
        ["git", "checkout", "-b", branch_name], capture_output=True, text=True
    )
    # Push the current code to the remote repo
    subprocess.run(
        ["git", "push", remote_name, branch_name], capture_output=True, text=True
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
