from git import Repo

REMOTE_NAME = "cemetery"  # TODO(crewtool) move to constants file
REMOTE_URL = "git@github.com:llm-random/llm-random-cemetery.git"  # TODO(crewtool) move to constants


class CodeVersioningAgent:
    def __init__(
        self,
        versioning_branch: str,
        remote_name: str = REMOTE_NAME,
        remote_url: str = REMOTE_URL,
        repo_path: str = ".",
    ):
        self.versioning_branch = versioning_branch
        self.remote_name = remote_name
        self.remote_url = remote_url

        self.repo = Repo(repo_path, search_parent_directories=True)

        self.original_branch = self.repo.active_branch.name
        self.original_branch_commit_hash = self.repo.head.object.hexsha

    def ensure_remote_config_exist(self):
        for remote in self.repo.remotes:
            if remote.name == self.remote_name:
                if remote.url != self.remote_url:
                    old_remote_url = remote.url
                    remote.set_url(self.remote_url)
                    print(
                        f"Updated url of '{self.remote_name}' remote from '{old_remote_url}' to '{self.remote_url}'"
                    )
                    return
                else:
                    return

        self.repo.create_remote(self.remote_name, url=self.remote_url)
        print(f"Added remote '{self.remote_name}' with url '{self.remote_url}'")

    def commit_pending_changes(self):
        if len(self.repo.index.diff("HEAD")) > 0:
            self.repo.git.commit(m="Versioning code", no_verify=True)

    def version_code(self):
        try:
            self.ensure_remote_config_exist()
            self.repo.git.add(all=True)
            self.commit_pending_changes()

            self.repo.git.checkout(b=self.versioning_branch)
            self.repo.git.push(self.remote_name, self.versioning_branch)
            print(
                f"Code versioned successfully to remote branch {self.versioning_branch} on '{self.remote_name}' remote!"
            )
        finally:
            self.reset_to_original_repo_state()

    def reset_to_original_repo_state(self):
        self.repo.git.checkout(self.original_branch, "-f")
        if self.versioning_branch in self.repo.branches:
            self.repo.git.branch("-D", self.versioning_branch)
        self.repo.head.reset(self.original_branch_commit_hash, index=True)
        print("Succesfuly restored working tree to original state!")


if __name__ == "__main__":
    # version_code("to_jest_test")

    version_daemon = CodeVersioningAgent(
        "to_branch2", repo_path="/Users/crewtool/llm-random/tojesttest"
    ).version_code()
