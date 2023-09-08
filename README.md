# llm-random



## Getting started
In root directory run `./start-dev.sh`. This will create a virtual environment, install requirements and setup git hooks.


## Experiments config
Use the baseline configuration as a template, which is in `research/conditional/train/configs/test/test_baseline.yaml`. Based on this template, create a new experiment config and put it in `lizrd/scripts/run_configs`.

## Run exeperiment

### Locally
`python -m lizrd.scripts.grid path/to/config`

### Remotely
`bash lizrd/scripts/run_exp_remotely.sh <remote_cluster_name> lizrd/scripts/run_configs/<your_config>`

## Code description

By directories:
* `lizrd` - main library
  * `core` - core code
    * `bert.py` - main layers, bert model
      * in the future, this should be split into layers, models, and some of it should be moved to research
    * `misc.py` - miscellanous functions
    * `nn.py` - base Module layer, and classes from torch.nn
  * `datasets` - data processing
    * `wikibookdata.py` - data processing of standard BERT training datasets
  * `scripts` - scripts for running experiments
    * `gen_run_trains.py` - generate shell scripts for running experiments
    * `run_train.sh` - shell script for running a single experiment
  * `support` - support code
    * `ash.py` - Assert Shapes on tensors
    * `metrics.py` - logging metrics during training
    * `profile.py` - for profiling models and layers
    * `test_utils.py` - testing utilities
  * `train` - training code
    * `bert_train.py` - main training script
* `research` - experimental code
    * `archived_code` - probably should be removed
    * `initialization` - research on better initialization scheme
    * `timing` - tests with profiling the main code
    * `conditional` - research on conditional computation (to be split from `core`)
    * `nonlinearities` - research on smaller neurons
    * `reinitialization` - research on recycling neurons
 
# License

This project is licensed under the terms of the Apache License, Version 2.0.

    Copyright 2023 LLM-Random Authors
    
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    
        http://www.apache.org/licenses/LICENSE-2.0
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

