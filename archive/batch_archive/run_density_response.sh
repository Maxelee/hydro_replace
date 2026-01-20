#!/bin/bash
#SBATCH --job-name=density_response
#SBATCH --output=logs/density_response_%j.out
#SBATCH --error=logs/density_response_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=34
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH -p cca
#SBATCH --mem=180G

# =============================================================================
# Compute baryonic response for 2D density planes (lenspots)
# =============================================================================
#
# This runs driver_density_response_mpi.py which:
#   1. Computes statistics (power spectrum, PDF, moments) for all 34 models
#   2. Computes response fractions for each Replace configuration
#
# The density planes are the raw projected matter fields before ray-tracing,
# allowing direct measurement of baryonic effects on the matter distribution.
#
# Statistics computed:
#   - Power spectrum P(k)
#   - PDF of log(1 + delta)
#   - Variance, Skewness, Kurtosis
#
# Usage:
#   sbatch run_density_response.sh           # Run both stages
#   sbatch run_density_response.sh 1         # Stage 1 only (compute stats)
#   sbatch run_density_response.sh 2         # Stage 2 only (analysis)
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
echo "Running density plane baryonic response analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "Started at: $(date)"
echo "=============================================="

# Create output directories
mkdir -p density_statistics_cache
mkdir -p density_response_output
mkdir -p figures

# Get stage argument (default: both)
STAGE=${1:-both}

echo "Running stage: $STAGE"

# Back up and clear old cache to force recomputation
if [ -d "density_statistics_cache" ]; then
    BACKUP_DIR="density_statistics_cache_backup_$(date +%Y%m%d_%H%M%S)"
    echo "Backing up old cache to $BACKUP_DIR"
    mv density_statistics_cache "$BACKUP_DIR"
fi
mkdir -p density_statistics_cache

# Run the MPI-parallelized driver with --force-recompute
srun -n 34 /mnt/home/mlee1/venvs/hydro_replace/bin/python -u driver_density_response_mpi.py $STAGE --force-recompute

echo ""
echo "=============================================="
echo "Job complete!"
echo "Finished at: $(date)"
echo "=============================================="
