import argparse
import os
import hydra
import yaml
from grid_generator.generate_configs import create_grid_config
from grid_generator.sbatch_builder import generate_sbatch_script
from model import run
import resolver as _  # I should be able to ignore this line by linter, but ~ things like # ignore did not work
from hydra import initialize, compose
import logging
from omegaconf import OmegaConf

def dump_grid_configs(configs_grid, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    class CustomDumper(yaml.SafeDumper):
        def write_line_break(self, data=None):
            super().write_line_break(data)
            if len(self.indents) == 1:  # Check if we're at the root level
                super().write_line_break()

    for idx, (cfg_dict, overrides_list) in enumerate(configs_grid):
        cfg_dict["overrides"] = overrides_list
        cfg_dict["_run_"] = True

        out_path = os.path.join(output_folder, f"config_{idx}.yaml")
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg_dict, f, Dumper=CustomDumper, sort_keys=True)


logger = logging.getLogger(__name__)
@hydra.main(version_base=None, config_path=".", config_name="experiment")
def main(config):

    if config.get("_run_"):
        run(config)
        return

    configs_grid = create_grid_config(config)
    output_folder = "generated_configs" # TODO parametrize
    dump_grid_configs(configs_grid, output_folder)

    generate_sbatch_script(config.slurm, output_folder, len(configs_grid), config.venv_path)

    if config.get("_debug_"):
        training_config, overrides = configs_grid[0]
        training_config["overrides"] = overrides
        training_config = OmegaConf.create(training_config)
        run(training_config)

if __name__ == "__main__":
    main()
