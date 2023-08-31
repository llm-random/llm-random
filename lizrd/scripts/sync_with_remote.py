import os
from fabric import Connection
from argparse import ArgumentParser
from git import Repo
from lizrd.support.code_versioning_support import find_git_root


def go_to_sparsity():
    git_root = find_git_root()
    repo = Repo(git_root)
    assert repo.remotes.origin.url in [
        "git@github.com:llm-random/llm-random.git",
        "git@github.com:sebastianjaszczur/sparsity.git",
    ], "You're not in the right repo! Move to sparsity. Aborting..."
    os.chdir(git_root)


def rsync_to_remote(host, local_dir, remote_dir):
    try:
        with Connection(host) as c:
            rsync_command = (
                f"rsync -rlp -e ssh {local_dir} {c.user}@{c.host}:{remote_dir}"
            )
            c.local(rsync_command)
            print(f"Successfully synced {local_dir} to {remote_dir} on {host}")
    except Exception as e:
        raise Exception(f"[RSYNC ERROR]: An error occurred during rsync: {str(e)}")


def run_remote_script(host, script):
    try:
        with Connection(host) as c:
            result = c.run(script)
            print(f"Successfully ran script ({result.command})")
    except Exception as e:
        raise Exception(f"An error occurred while running the script: {str(e)}")


def run_grid_remotely(host, config):
    script = (
        f"cd ~/sparsity && find -name {config}"
        + " -exec python3 -m lizrd.scripts.grid {} \;"
    )
    run_remote_script(host, script)


def set_up_permissions(host):
    # if it turns out at some point that we need to add more permissions for files, we can add them here
    pass


if __name__ == "__main__":
    # set up argparse for hostname, config
    parser = ArgumentParser()
    parser.add_argument("--host", type=str)
    # create parser
    args = parser.parse_args()

    go_to_sparsity()
    working_dir = os.getcwd()
    rsync_to_remote(args.host, working_dir + "/lizrd", "~/sparsity/lizrd ")
    rsync_to_remote(args.host, working_dir + "/research", "~/sparsity/research ")
    set_up_permissions(args.host)
