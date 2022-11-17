import os
import sys
import datetime

template = """#!/bin/bash
#
#SBATCH --job-name=lizard_train{JOB_ID}
#SBATCH --partition=common
#SBATCH --qos=24gpu7d
#SBATCH --gres=gpu:titanv:1
#SBATCH --time=0-08:00:00
#SBATCH --output=/home/jaszczur/logs/t{TIMESTAMP}/sbatchlogs{JOB_ID}.txt

source venv/bin/activate
python3 -m lizrd.train.bert_train {ARGS}
"""

SLEEPTIME = 3

# create a directory name based on timestamp
timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")

# create a directory for the logs
log_dir = "logs/t" + timestamp
print("log_dir:", log_dir)
os.makedirs("/home/jaszczur/" + log_dir)

jobs = []
REAL_RUN = None

# # for lr in [0.016, 0.008, 0.004, 0.002, 0.001, 0.0005]:
# for lr in [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1]:
#     jobs.append((f'LEARNING_RATE={lr}', 'FIXED'))
#
# # for lr in [0.016, 0.008, 0.004, 0.002, 0.001, 0.0005]:
# for lr in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
#     jobs.append((f'LEARNING_RATE={lr}', 'STANDARD'))


NEXPERTS = 32
SPARSITY = 8
EXPERTSIZE = 64


PREFIX = None

DFF = 2048

# for expertsize in [32, 64, 128, 256, 512, 1024, 2048]:
for expertsize in [512, 1024, 2048]:
    for sparsity in [2, 4, 8, 16, 32]:
        jobs.append(
            (
                "SPARSE",
                f"NEXPERTS={DFF//expertsize}",
                f"SPARSITY={sparsity}",
                f"EXPERTSIZE={expertsize}",
                f"NAME=sp{sparsity}_es{expertsize}",
            )
        )
    jobs.append(("DENSE", f"NAME=dense"))

jobs = jobs[:3]

REMOVEAFTER = False

for arg in sys.argv[1:]:
    if arg.startswith("--prefix="):
        PREFIX = arg[arg.index("=") + 1 :]
    elif arg.startswith("--real"):
        REAL_RUN = True
    elif arg.startswith("--test"):
        REAL_RUN = False
    elif arg.startswith("--removeafter"):
        REMOVEAFTER = True
    elif arg.startswith("--sleep="):
        SLEEPTIME = int(arg[arg.index("=") + 1 :])
    else:
        print(f"Unknown argument: {arg}")
        sys.exit(1)

if PREFIX is None:
    raise ValueError("--prefix=<prefix> is required")
if REAL_RUN is None:
    raise ValueError("--real or --test is required")


CLEARMLDIR = f"jaszczur/init/batched/{PREFIX}"

# make directory named "train_scripts"
train_scripts_dir = "train_scripts"
if not os.path.exists(train_scripts_dir):
    os.makedirs(train_scripts_dir)

# remove files from 'train_scripts' directory
for f in os.listdir(train_scripts_dir):
    os.remove(os.path.join(train_scripts_dir, f))

for job_id, job in enumerate(jobs):
    job_id = f"{PREFIX}{job_id}"
    args = " ".join(job) + f" CLEARMLDIR={CLEARMLDIR}"
    filetext = template.format(JOB_ID=job_id, ARGS=args, TIMESTAMP=timestamp)
    with open(f"{train_scripts_dir}/train{job_id}.sh", "w") as f:
        f.write(filetext)

# run all scripts in 'train_scripts' directory
for f in os.listdir(train_scripts_dir):
    if REAL_RUN:
        # sleep for 3 secs
        os.system(f"sleep {SLEEPTIME}")
        os.system(f"sbatch {train_scripts_dir}/{f}")
    else:
        print(f"Would run: sbatch {train_scripts_dir}/{f}")
    print(f"Submitted {f}")

if REMOVEAFTER:
    # remove files from 'train_scripts' directory and directory itself
    for f in os.listdir(train_scripts_dir):
        os.remove(os.path.join(train_scripts_dir, f))
    os.rmdir(train_scripts_dir)
    print(f"Removed {train_scripts_dir}")
