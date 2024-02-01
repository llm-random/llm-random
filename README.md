## Development (WIP)
### Getting started
In the root directory run `./start-dev.sh`. This will create a virtual environment, install requirements and set up git hooks.

## Running Experiments (WIP)

### Experiments config
Use the baseline configuration as a template, which is in `research/conditional/train/configs/test/test_baseline.yaml`. Based on this template, create a new experiment config and put it in `source/scripts/run_configs`.

### Running Locally
`python -m source.scripts.grid path/to/config`

### Running Remotely
`bash source/scripts/run_exp_remotely.sh <remote_cluster_name> source/scripts/run_configs/<your_config>`

# License

This project is licensed under the terms of the Apache License, Version 2.0.

    Copyright 2023 Authors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.