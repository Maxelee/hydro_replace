#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=icelake
#SBATCH -J prep_data
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -o logs/01_prepare_data.o%j
#SBATCH -e logs/01_prepare_data.e%j
#SBATCH --mail-user=mlee@flatironinstitute.org
#SBATCH --mail-type=ALL
#SBATCH -t 02:00:00

# Stage 1: Data Preparation
# - Load halo catalogs
# - Perform bijective matching
# - Save matched catalog
#
# Expected runtime: ~1-2 hours
# Memory: ~16-32 GB for loading full catalogs

set -e

echo "=============================================="
echo "Stage 1: Data Preparation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="

# Load modules
module load python openmpi python-mpi hdf5

# Activate environment
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

# Change to project directory
cd /mnt/home/mlee1/hydro_replace2

# Create output directories
mkdir -p /mnt/home/mlee1/ceph/hydro_replace/matching
mkdir -p logs

# Run script
python scripts/01_prepare_data.py \
    --config config/simulation_paths.yaml \
    --min-mass 1e12 \
    --max-mass 1e15 \
    --snapshot 99 \
    --log-level INFO

echo "=============================================="
echo "End time: $(date)"
echo "=============================================="
