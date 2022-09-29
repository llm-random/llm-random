# sparsity

# Installing requirements

Simple `pythin3 -m pip install -r requirements.txt` should work. Be sure to use a virtualenv.

## Usage
Run a single local experiment with `python3 bert_train.py TESTING`. The flag `TESTING` will disable ClearML, and run a smaller model.
If you use ClearML, you must have clearml config in your home directory. Note that if you don't use srun.sbatch on entropy, you won't have access to GPU.

To run a single experiment on entropy, write your shell script based on `run_train.sh`, and run it either by:
`srun --partition=common --qos=24gpu7d --gres=gpu:titanv:1 run_time.sh`
or `sbatch run_time.sh`.

To run multiple experiment, modify gen_run_trains.py and run:
`python3 gen_run_trains.py --prefix=SWPB --real --sleep=70`

To just generate configs without scheduling jobs, run:
`python3 gen_run_trains.py --prefix=SWP --test --sleep=0`
