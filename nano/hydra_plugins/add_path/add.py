from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
import os

root_dir_name = "nano"


class MainSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Appends the search path for this plugin to the end of the search path

        current_dir = os.getcwd()
        path_parts = current_dir.split(os.sep)

        if root_dir_name in path_parts:
            index = path_parts.index(root_dir_name)
            result_path = os.sep.join(path_parts[: index + 1])

            search_path.append(
                provider="add-nano-root-path", path=f"file://{result_path}"
            )
