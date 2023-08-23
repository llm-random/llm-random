import re
import subprocess
import shutil
import os

from lizrd.support.misc import generate_random_string


def run_subprocess(command, error_message, branch=None):
    try:
        output = subprocess.check_output(command, shell=True).strip().decode("utf-8")
        return output
    except subprocess.CalledProcessError as e:
        if branch:
            # Try to revert changes and return to original branch
            try:
                subprocess.check_output("git reset --hard", shell=True)
                subprocess.check_output(f"git checkout {branch}", shell=True)
                subprocess.check_output(
                    'git stash pop "stash_for_experiment_versioning"', shell=True
                )
            except subprocess.CalledProcessError as err:
                # If reverting fails, raise another exception
                raise Exception(
                    f"Failed to revert changes, error occurred during {command}: {str(err)}"
                )
        raise Exception(f"{error_message}: {str(e)}")


def version_code_and_copy(name_for_branch, remote_url):
    # Record current branch
    branch = run_subprocess(
        "git rev-parse --abbrev-ref HEAD", "Failed to get current branch"
    )

    # Stash any uncommitted changes with a custom message
    run_subprocess(
        'git stash save "stash_for_experiment_versioning"',
        "Failed to stash changes",
        branch,
    )

    # Stage all changes
    run_subprocess("git add .", "Failed to stage changes", branch)

    # Commit changes
    run_subprocess(
        'git commit --no-verify -m "commit before experiment"', "Commit failed", branch
    )

    # Push changes
    run_subprocess(
        f"git push --no-verify A {name_for_branch}", "Failed to push changes", branch
    )

    # Return to original branch and apply stashed changes (using stash name)
    run_subprocess(
        f"git git reset --hard", "Failed to reset git back to before changes", branch
    )
    run_subprocess(
        f"git checkout {branch}", "Failed to checkout original branch", branch
    )
    run_subprocess(
        'git stash pop "stash_for_experiment_versioning"',
        "Failed to apply stashed changes to original branch",
        branch,
    )


def copy_and_version_code(
    name_for_branch,
    newdir_name,
    push_to_git,
    remote_url="git@github.com:Simontwice/sparsity.git",
):
    """Copies the current code to a new directory, and pushes the code to a remote repo.
    NOTE: it is assumed that this function is called from inside the project.
    Prerequisite: the user needs to be able to push to the remote repo from the command line without entering a password.
    If not met, the user needs to set up ssh keys."""

    # Find git root directory
    root_dir = find_git_root()
    newdir_path = f"{os.path.dirname(root_dir)}/sparsity_code_cemetery/{newdir_name}"
    random_string = generate_random_string(10)

    # Set up ignore patterns
    with open(os.path.join(root_dir, ".versioningignore")) as f:
        versioning_ignore_patterns = f.read().splitlines()
        versioning_ignore_patterns = [
            pattern
            for pattern in versioning_ignore_patterns
            if pattern != "" and pattern[0] != "#"
        ]
        versioning_ignore_patterns = [p.strip() for p in versioning_ignore_patterns] + [
            ".git"
        ]

    versioning_ignore_patterns = shutil.ignore_patterns(*versioning_ignore_patterns)

    print(f"Copying code to {newdir_path}...")
    # Copy the project root directory to a new directory, ignoring files described in versioning_ignore_patterns
    shutil.copytree(root_dir, newdir_path, ignore=versioning_ignore_patterns)
    print(f"Code copied successfully to {newdir_path}")

    # Change to the new directory
    os.chdir(newdir_path)

    if push_to_git:
        tmp_dir = f"{root_dir}/tmp/{random_string}"
        tmp_git_dir = f"{tmp_dir}/.git"
        subprocess.run(
            ["mkdir", "-p", tmp_dir],
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["cp", "-r", f".git", tmp_dir],
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["ln", "-s", tmp_git_dir, ".git"], capture_output=True, text=True
        )
        print(f"Creating branch {name_for_branch}")
        # Push the code to the remote repo
        push_code_to_url(name_for_branch, remote_url)
        print(
            f"Code pushed successfully to {remote_url} under branch {name_for_branch}"
        )
        subprocess.run(["rm", "-rf", tmp_dir], capture_output=True, text=True)
        subprocess.run(["rm", ".git"], capture_output=True, text=True)
    return newdir_path


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
        run_subprocess_verbose(
            ["git", "remote", "add", "code_image_cemetery", remote_url]
        )

    # Pull (this is a fix for >fatal: you are on a branch yet to be born)
    run_subprocess_verbose(["git", "pull"])

    # Create a new branch
    run_subprocess_verbose(["git", "checkout", "-b", branch_name])

    # Stage the changes
    run_subprocess_verbose(["git", "add", "."])

    # Commit the changes
    run_subprocess_verbose(
        ["git", "commit", "--no-verify", "-m", "Committing local changes"]
    )

    # Push the current code to the remote repo
    run_subprocess_verbose(["git", "push", remote_name, branch_name])


def run_subprocess_verbose(argument_list):
    prc = subprocess.run(
        argument_list,
        capture_output=True,
        text=True,
    )

    if prc.returncode != 0:
        print(f"Error: {prc.stderr}")
        raise Exception("Error: Git operation was not successful")


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
