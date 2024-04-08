1. Upload this folder (download_c4) to target cluster (`rsync -r lizrd/scripts/download_c4 athena:./`)
2. Login to cluster (`ssh athena`)
3. `cd download_c4`
4. Create venv and install huggingface datasets (`python3 -m venv venv && source venv/bin/activate && pip install 'datasets==<insert current version from requirements.txt>'`)
5. Match slurm preamble in `download_c4.sh` to your cluster
6. Run `./download_c4.sh [target dataset path] [cache directory]`
7. Make sure everyone can read target: `chmod -R ugo+r [target dataset path]`
8. (Optionally) cleanup: `rm -r venv && rm [cache directory]`