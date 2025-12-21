#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=icelake
#SBATCH -J halo_match
#SBATCH -n 16
#SBATCH --ntasks-per-node=16
#SBATCH --mem=200G
#SBATCH -t 02:00:00
#SBATCH -o logs/halo_match.o%j
#SBATCH -e logs/halo_match.e%j
#SBATCH --mail-user=mlee@flatironinstitute.org
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Compute Halo Matches with MPI Parallelization
# =============================================================================
# This script runs the BijectiveMatcher to find matched halos between
# DMO and hydro simulations. Results are cached for subsequent pipeline runs.
#
# Usage:
#   sbatch batch/compute_matches.sh
#   sbatch --export=RES=2500,SNAP=99 batch/compute_matches.sh
# =============================================================================

set -e

# Load modules
module purge
module load python openmpi python-mpi hdf5

# Activate virtual environment
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

# Move to project directory
cd /mnt/home/mlee1/hydro_replace2

# Create directories
mkdir -p logs

# Parameters
RES=${RES:-625}
SNAP=${SNAP:-99}
MASS_MIN=${MASS_MIN:-1e13}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/home/mlee1/ceph/hydro_replace}

echo "========================================================================"
echo "HALO MATCHING - MPI Parallel"
echo "========================================================================"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Resolution: ${RES}^3"
echo "Snapshot: $SNAP"
echo "Mass min: $MASS_MIN Msun/h"
echo "MPI tasks: $SLURM_NTASKS"
echo "========================================================================"

# Run with MPI
srun --mpi=pmix python -u scripts/compute_halo_matches.py \
    --resolution $RES \
    --snapshot $SNAP \
    --mass-min $MASS_MIN \
    --output-dir $OUTPUT_DIR

echo "========================================================================"
echo "Matching complete: $(date)"
echo "========================================================================"
