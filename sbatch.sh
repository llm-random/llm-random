#!/bin/bash

#SBATCH --job-name=multinode_job        # Job name
#SBATCH --output=output.txt          # Output file
#SBATCH --error=error.txt            # Error file
#SBATCH --nodes=4                       # Number of nodes
#SBATCH --time=12:00:00                 # Time limit (1 hour)
#SBATCH --partition=all             # Partition name
#SBATCH --ntasks-per-node=1


#SBATCH --mail-type=END,FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=your_email@example.com # Email address

# # Load necessary modules
# module load python/3.8
# module load mpi/openmpi-x86_64

# # Activate virtual environment if needed
# source /path/to/your/venv/bin/activate

# # Run the Python script with MPI
# mpirun -np $SLURM_NTASKS python your_script.py

# # Deactivate virtual environment if used
# deactivate

ls
source /home/mp/venv/bin/activate
cd /home/mp/llm-random
export PJRT_DEVICE=TPU
# export PT_XLA_DEBUG_LEVEL=2
srun python3 -m lizrd.grid --use_tpu --config_path configs/experiments/subtoken/tpu.yaml

#SBATCH --mail-type=END,FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=your_email@example.com # Email address
