import hydra
from token_reduction.model import run
import resolver as _  # I should be able to ignore this line by linter, but ~ things like # ignore did not work
import token_reduction.model as _


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(job_config):
    hydra_config = hydra.utils.HydraConfig.get()
    run(job_config, hydra_config)


if __name__ == "__main__":
    main()
