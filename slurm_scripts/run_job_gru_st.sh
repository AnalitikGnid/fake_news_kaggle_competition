#!/bin/bash
#SBATCH --job-name=python_job
#SBATCH --output=output_gs.txt
#SBATCH --error=error_gs.txt
#SBATCH --time=01:00:00
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

echo "Running on $(hostname)"

# Set environment variables to use 4 CPU cores for numpy/OpenBLAS/MKL
# export OPENBLAS_NUM_THREADS=4
# export OMP_NUM_THREADS=4
# export MKL_NUM_THREADS=4

python "gru_st.py"
