#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=icelake
#SBATCH -J test_access
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -o logs/test_data_access.o%j
#SBATCH -e logs/test_data_access.e%j
#SBATCH --mail-user=mlee@flatironinstitute.org
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00

# Quick test job to verify data access and module imports
# Run this first before the full pipeline

set -e

echo "=============================================="
echo "Test: Data Access Verification"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="

module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate
cd /mnt/home/mlee1/hydro_replace2

mkdir -p logs

python scripts/test_data_access.py

echo "=============================================="
echo "End time: $(date)"
echo "=============================================="
