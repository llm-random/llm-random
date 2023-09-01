import os
from fabric import Connection
from argparse import ArgumentParser
from git import Repo
from lizrd.support.code_versioning import find_git_root, version_code
from lizrd.support.misc import generate_random_string


def go_to_llm_random():
    git_root = find_git_root()
    repo = Repo(git_root)
    assert repo.remotes.origin.url in [
        "git@github.com:llm-random/llm-random.git",
    ], "You're not in the right repo! Move to the llm-random folder, and make sure your origin is the llm-random repo. Aborting..."
    os.chdir(git_root)


def rsync_to_remote(host, local_dir):
    try:
        with Connection(host) as connection:
            base_dir = get_base_directory(connection)
            proxy_command = get_proxy_command(connection)
            rsync_command = f"rsync -zrlp -e {proxy_command} {local_dir} {connection.user}@{connection.host}:{base_dir}"
            print(f"Syncing {local_dir} to {connection.host}:{base_dir}...")
            connection.local(
                rsync_command,
            )
            print("Sync complete.")
            return base_dir
    except Exception as e:
        raise Exception(f"[RSYNC ERROR]: An error occurred during rsync: {str(e)}")


def get_base_directory(connection):
    if connection.host == "athena.cyfronet.pl":
        base_dir = (
            f"/net/pr2/projects/plgrid/plggllmeffi/{connection.user[3:]}/llm-random"
        )
    else:
        base_dir = f"~/llm-random"
    return base_dir


def get_proxy_command(connection):
    if connection.host == "4124gs01":
        cc = Connection(connection.ssh_config["proxyjump"])
        proxy_command = f"'ssh -A -J {cc.user}@{cc.host}'"
    else:
        proxy_command = "ssh"
    return proxy_command


def run_remote_script(host, script):
    try:
        with Connection(host) as c:
            result = c.run(script)
    except Exception as e:
        raise Exception(f"An error occurred while running the script: {str(e)}")


def set_up_permissions(host):
    # if it turns out at some point that we need to add more permissions for files, we can add them here
    pass


if __name__ == "__main__":
    # set up argparse for hostname, config
    parser = ArgumentParser()
    parser.add_argument("--host", type=str)
    # create parser
    args = parser.parse_args()

    go_to_llm_random()
    working_dir = os.getcwd()
    base_dir = rsync_to_remote(args.host, working_dir + "/lizrd")
    _ = rsync_to_remote(args.host, working_dir + "/research")
    _ = rsync_to_remote(args.host, working_dir + "/.versioningignore")
    # WRITE base_dir to temp file
    with open("base_dir.txt", "w") as f:
        f.write(base_dir)
    set_up_permissions(args.host)
    name_for_branch = "exp_hash_" + generate_random_string(10)
    version_code(name_for_branch)
