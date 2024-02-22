# llm-random
We are LLM-Random, a research group at [IDEAS NCBR](https://ideas-ncbr.pl/en/) (Warsaw, Poland). We develop this repo and use it to conduct research. To learn more about us and our research, check out our blog, [llm-random.github.io](https://llm-random.github.io/).

## Publications, preprints and blogposts
- MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts ([arxiv](https://arxiv.org/abs/2401.04081), [blogpost](https://llm-random.github.io/posts/moe_mamba/))
- Mixture of Tokens: Efficient LLMs through Cross-Example Aggregation ([arxiv](https://arxiv.org/abs/2310.15961), [blogpost](https://llm-random.github.io/posts/mixture_of_tokens/))



## Development (WIP)
### Getting started
In the root directory run `./start-dev.sh`. This will create a virtual environment, install requirements and set up git hooks.

## Running Experiments (WIP)

### Experiments config
Use the baseline configuration as a template, which is in `configs/test/test_baseline.yaml`. Based on this template, create a new experiment config and put it in `lizrd/scripts/run_configs`.

### Running Locally
`python -m lizrd.scripts.grid path/to/config`

### Running Remotely
`bash lizrd/scripts/run_exp_remotely.sh <remote_cluster_name> lizrd/scripts/run_configs/<your_config>`

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

