# LiZRD - Library for Zipping through Research Designs

Name suggestions are very welcome.


## Getting started
After cloning the repo, `cd` to repo directory and run `./start-dev.sh`. This will create a virtual environment, install requirements and setup git hooks.

## Installing requirements

Running `start-dev.sh` will install requirements. If you wish to setup your environment manually, `python3 -m pip install -r requirements.txt` should work. Be sure to use a virtualenv.

## Usage
Run a single local experiment with `python3 -m lizrd.train.bert_train TESTING`. The flag `TESTING` will disable ClearML, and run a smaller model.
If you use ClearML, you must have clearml config in your home directory. Note that if you don't use srun.sbatch on entropy, you won't have access to GPU.

To run a single experiment on entropy, write your shell script based on `run_train.sh`, and run it either by:
`srun --partition=common --qos=24gpu7d --gres=gpu:titanv:1 run_time.sh`
or `sbatch lizrd/scripts/run_train.sh`.

To run multiple experiment, modify gen_run_trains.py and run:
`python3 -m lizrd.scripts.gen_run_trains.py --prefix=SWPB --real --sleep=70`

To just generate configs without scheduling jobs, run:
`python3 -m lizrd.scripts.gen_run_trains.py --prefix=SWP --test --sleep=0`

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

