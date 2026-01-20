#!/bin/bash
#SBATCH --job-name=baryonic_response
#SBATCH --output=logs/baryonic_response_%j.out
#SBATCH --error=logs/baryonic_response_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=34
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH -p cca
#SBATCH --mem=120G

# =============================================================================
# Compute baryonic response statistics for all convergence maps
# =============================================================================
#
# This runs the driver_baryonic_response_mpi.py which:
#   1. Computes statistics (C_ell, peaks, minima) for all 34 models
#   2. Computes response fractions and tile responses
#
# The script uses 34 MPI ranks (one per model) for Stage 1 parallelization.
# Stage 2 (analysis) runs on rank 0 only.
#
# Usage:
#   sbatch run_baryonic_response.sh           # Run both stages
#   sbatch run_baryonic_response.sh 1         # Stage 1 only (compute stats)
#   sbatch run_baryonic_response.sh 2         # Stage 2 only (analysis)
#
# =============================================================================

set -e

# Load modules
module purge
module load python openmpi/4.1.8 python-mpi hdf5/mpi-1.12.3
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

# Work directory
cd /mnt/home/mlee1/hydro_replace2/notebooks

echo "=============================================="
echo "Running baryonic response analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "Started at: $(date)"
echo "=============================================="

# Create output directories
mkdir -p statistics_cache
mkdir -p response_analysis_output
mkdir -p figures

# Get stage argument (default: both)
STAGE=${1:-both}

echo "Running stage: $STAGE"

# Run the MPI-parallelized driver
srun -n 34 /mnt/home/mlee1/venvs/hydro_replace/bin/python -u driver_baryonic_response_mpi.py $STAGE

echo ""
echo "=============================================="
echo "Job complete!"
echo "Finished at: $(date)"
echo "=============================================="
