1. Upload this folder (download_c4) to target cluster
1. `cd download_c4`
2. Create venv and install huggingface datasets 
3. Match slurm preamble in `download_c4.sh` to your cluster
3. Run `./download_c4.sh [target dataset path] [cache directory]`