#!/bin/bash
#SBATCH --job-name=python_job
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --time=00:00:01
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

echo "Running on $(hostname)"
python py_script.py
